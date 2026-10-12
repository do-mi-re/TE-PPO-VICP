"""Config-driven PPO training entrypoint for the RSU leader.

Exactness label: implementation convenience only; mathematical semantics live
in the environment and its wrappers.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize

from src.evaluation.provenance import config_digest
from src.rl.checkpoint_selection import (
    LAST_TRAIN_VECNORMALIZE,
    SB3_LAST_MODEL,
    SB3_SELECTED_MODEL,
    SELECTED_VECNORMALIZE,
    cpu_validation_ranges,
)
from src.rl.eval import evaluate_model, write_episode_rows_csv
from src.rl.policy import ActorNetworkConfig, price_logit_for_fraction
from src.rl.ppo import (
    NormalizationConfig,
    RewardConfig,
    build_callbacks,
    build_ppo_model,
    make_vectorized_eval_env,
    make_vectorized_training_env,
)
from src.rl.progress import TrainingProgress
from src.rl.run_metadata import write_run_metadata
from src.rl.training_config import resolve_effective_training_config
from src.rl.value import CriticNetworkConfig
from src.sim.scenario import ScenarioConfig

# Keys in the algorithm config block that are consumed directly by the PPO
# builder; everything else is forwarded as algorithm_options to the env stack.
ALGORITHM_OPTION_KEYS: frozenset[str] = frozenset({
    "name",
    "total_timesteps",
    "learning_rate",
    "n_envs",
    "n_steps",
    "batch_size",
    "n_epochs",
    "gamma",
    "gae_lambda",
    "clip_range",
    "clip_range_vf",
    "ent_coef",
    "vf_coef",
    "max_grad_norm",
    "target_kl",
    "action_logit_bound",
    "initial_price_fraction",
    "device",
})


def load_training_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML training config."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("training config must deserialize to a mapping")
    return data


def seed_everything(seed: int, *, strict: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible training.

    The default mode configures cuDNN for deterministic convolutions and sets
    ``CUBLAS_WORKSPACE_CONFIG`` so cuBLAS reductions are reproducible. This is
    enough to make same-seed, same-config runs bit-equal for the TE-PPO stack.

    ``strict=True`` additionally enables ``torch.use_deterministic_algorithms``
    which refuses to dispatch any non-deterministic op. That catches the
    remaining sources of nondeterminism, at the cost of failing noisily when
    an underlying library picks a non-deterministic kernel — opt in only for
    reproducibility audits.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # cuDNN is the dominant source of nondeterminism in SB3's feature extractor.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Affects cuBLAS contexts initialised *after* this call (SubprocVecEnv
    # workers and direct training calls). ``run.py`` sets the same
    # default before any torch import for the main process.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    if strict:
        torch.use_deterministic_algorithms(True, warn_only=False)


def resolve_device(device_str: str) -> torch.device:
    """Return a ``torch.device`` honoring the config's ``device`` string.

    ``"auto"`` prefers CUDA on the GPU server, falls back to Apple Silicon
    MPS for local macOS development, then CPU. MPS numerics differ slightly
    from CUDA (float32 only, different dispatch for some ops) so it's a
    dev-loop accelerator, not a substitute for CUDA in final results.
    """

    if device_str != "auto":
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _run_dir(config: dict[str, Any], output_root: str | Path | None) -> Path:
    artifacts = config.get("artifacts", {})
    run_name = str(artifacts.get("run_name", "ppo_run"))
    if output_root is not None:
        return Path(output_root) / run_name
    root = Path(str(artifacts.get("root_dir", "artifacts/ppo")))
    return root / run_name


def _evaluation_seeds(
    eval_config: dict[str, Any],
    *,
    default_seed_base: int,
    default_count: int,
) -> list[int]:
    """Resolve the evaluation seed list.

    When ``evaluation.seed_list`` is explicitly given in the config, use it
    verbatim. Otherwise auto-generate a contiguous block
    ``[default_seed_base, default_seed_base + default_count)``.
    """

    explicit = eval_config.get("seed_list")
    if explicit is not None:
        if not isinstance(explicit, list) or not explicit:
            raise ValueError("evaluation.seed_list must be a non-empty list when provided")
        return [int(value) for value in explicit]
    return [default_seed_base + index for index in range(default_count)]


def train_from_config(
    config: dict[str, Any],
    *,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Train PPO from a fully materialized config and return artifact metadata."""

    config = resolve_effective_training_config(config)
    algorithm_cfg = dict(config.get("algorithm", {}))
    algorithm_name = str(algorithm_cfg.get("name", "proposed")).lower()

    # Dispatch to non-PPO algorithms.
    if algorithm_name == "mappo":
        from src.rl.mappo import train_from_config as mappo_train

        return mappo_train(
            config,
            output_root=output_root,
        )
    if algorithm_name == "jdls":
        # JDLS runs on a custom PPO loop (see src.baselines.jdls) so its
        # Categorical assignment head contributes to policy gradient.
        from src.baselines.jdls import train_from_config as jdls_train

        return jdls_train(
            config,
            output_root=output_root,
        )
    if algorithm_name == "dbas":
        return _run_dbas_evaluation(config, output_root=output_root)

    seed = int(config.get("seed", 123))
    seed_everything(seed)

    scenario_config = ScenarioConfig(**config.get("scenario", {}))
    reward_config = RewardConfig(**config.get("reward", {}))
    actor_config = ActorNetworkConfig(**config.get("policy", {}).get("actor", {}))
    critic_config = CriticNetworkConfig(**config.get("policy", {}).get("critic", {}))
    algo_config = algorithm_cfg
    artifact_config = config.get("artifacts", {})
    eval_config = config.get("evaluation", {})
    normalization_config = NormalizationConfig(**config.get("normalization", {}))

    run_dir = _run_dir(config, output_root)
    run_name = str(artifact_config.get("run_name", run_dir.name))
    checkpoints_dir = run_dir / "checkpoints"
    eval_log_dir = run_dir / "eval"
    log_dir = run_dir / "logs"
    for directory in (run_dir, checkpoints_dir, eval_log_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # Snapshot the user-supplied config plus the effective algorithm section,
    # so reproducers can see the actual default values used (e.g.
    # ``gae_lambda`` = 1.0 when the yaml omits it).
    with (run_dir / "config_snapshot.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    n_envs = int(algo_config.get("n_envs", 1))
    action_logit_bound = float(algo_config.get("action_logit_bound", 8.0))
    eval_seed_list = _evaluation_seeds(
        eval_config,
        default_seed_base=seed + 20_000,
        default_count=int(eval_config.get("n_eval_episodes", 3)),
    )

    algorithm_options = {
        key: value
        for key, value in algorithm_cfg.items()
        if key not in ALGORITHM_OPTION_KEYS
    }
    train_env = make_vectorized_training_env(
        scenario_config=scenario_config,
        reward_config=reward_config,
        n_envs=n_envs,
        normalization_config=normalization_config,
        action_logit_bound=action_logit_bound,
        monitor_path=str(log_dir / "train_monitor.csv"),
        algorithm_name=algorithm_name,
        algorithm_options=algorithm_options,
    )
    train_env.seed(seed)
    train_env.reset()
    validation_ranges = cpu_validation_ranges(eval_config)
    eval_scenarios = (
        [
            replace(
                scenario_config,
                cpu_cycles_per_s_range=band,
                cpu_cycles_per_s_regimes=None,
                cpu_regime_probabilities=None,
            )
            for band in validation_ranges
        ]
        if validation_ranges else [scenario_config]
    )
    eval_envs = [
        make_vectorized_eval_env(
            scenario_config=eval_scenario,
            reward_config=reward_config,
            normalization_config=normalization_config,
            action_logit_bound=action_logit_bound,
            monitor_path=str(log_dir / f"eval_monitor_cell{cell_index}.csv"),
            algorithm_name=algorithm_name,
            algorithm_options=algorithm_options,
        )
        for cell_index, eval_scenario in enumerate(eval_scenarios)
    ]
    for cell_index, candidate_env in enumerate(eval_envs):
        candidate_env.seed(seed + 10_000 + cell_index)
        candidate_env.reset()
    eval_env = eval_envs[0]

    model = build_ppo_model(
        env=train_env,
        actor_config=actor_config,
        critic_config=critic_config,
        learning_rate=float(algo_config.get("learning_rate", 3e-4)),
        n_steps=int(algo_config.get("n_steps", 64)),
        batch_size=int(algo_config.get("batch_size", 64)),
        n_epochs=int(algo_config.get("n_epochs", 10)),
        gamma=float(algo_config["gamma"]),
        gae_lambda=float(algo_config.get("gae_lambda", 1.0)),
        clip_range=float(algo_config["clip_range"]),
        clip_range_vf=(
            None
            if algo_config.get("clip_range_vf") is None
            else float(algo_config.get("clip_range_vf"))
        ),
        ent_coef=float(algo_config.get("ent_coef", 0.0)),
        vf_coef=float(algo_config.get("vf_coef", 0.5)),
        max_grad_norm=float(algo_config.get("max_grad_norm", 0.5)),
        target_kl=(
            None
            if algo_config.get("target_kl") is None
            else float(algo_config.get("target_kl"))
        ),
        seed=seed,
        device=str(algo_config.get("device", "cpu")),
    )
    initial_price_fraction = float(algo_config.get("initial_price_fraction", 0.5))
    initial_price_logit = price_logit_for_fraction(initial_price_fraction)
    action_bias = getattr(model.policy.action_net, "bias", None)
    if action_bias is None or action_bias.numel() < scenario_config.num_tasks:
        raise RuntimeError("PPO policy does not expose the expected price-head bias")
    with torch.no_grad():
        action_bias[: scenario_config.num_tasks].fill_(initial_price_logit)
    print(f"[train] PPO device: {model.device}", flush=True)
    # Keep complete machine-readable diagnostics without flooding the terminal
    # with one SB3 table per rollout.  TrainingProgress below owns the concise
    # human-facing display.
    model.set_logger(configure(str(log_dir), ["csv", "json"]))

    total_timesteps = int(algo_config.get("total_timesteps", 1_000))
    progress = TrainingProgress(total=total_timesteps, label=run_name)
    fingerprint = config_digest(config)
    if fingerprint is None:
        raise RuntimeError("training configuration is not digestible")

    callbacks = build_callbacks(
        checkpoint_dir=checkpoints_dir,
        run_dir=run_dir,
        eval_log_dir=eval_log_dir,
        eval_env=eval_env,
        checkpoint_freq=int(artifact_config.get("checkpoint_freq", 100)),
        interval_env_steps=int(eval_config["interval_env_steps"]),
        config_fingerprint=fingerprint,
        force_terminal_evaluation=bool(
            eval_config.get("force_terminal_evaluation", True)
        ),
        eval_seeds=eval_seed_list,
        progress=progress,
        cpu_eval_envs=eval_envs if validation_ranges else None,
        cpu_validation_ranges=validation_ranges,
    )

    progress.phase("formal training")
    try:
        model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=False)
    finally:
        progress.close()

    if isinstance(train_env, VecNormalize):
        train_env.save(str(run_dir / LAST_TRAIN_VECNORMALIZE))

    last_model_path = run_dir / SB3_LAST_MODEL
    model.save(str(last_model_path))
    selected_model_path = run_dir / SB3_SELECTED_MODEL
    if not selected_model_path.exists():
        raise RuntimeError("training produced no selected checkpoint")
    if isinstance(eval_env, VecNormalize):
        selected_norm_path = run_dir / SELECTED_VECNORMALIZE
        if not selected_norm_path.exists():
            raise RuntimeError("selected checkpoint misses its VecNormalize state")
        loaded_eval_envs = []
        for candidate_env in eval_envs:
            if not isinstance(candidate_env, VecNormalize):
                raise RuntimeError("CPU validation normalization environments drifted")
            loaded = VecNormalize.load(str(selected_norm_path), candidate_env.venv)
            loaded.training = False
            loaded.norm_reward = False
            loaded_eval_envs.append(loaded)
        eval_envs = loaded_eval_envs
        eval_env = eval_envs[0]
    model = type(model).load(
        str(selected_model_path), env=eval_env, device=str(model.device)
    )
    print(f"[{run_name}] selected-checkpoint validation", flush=True)
    final_eval_rows: list[dict[str, Any]] = []
    final_rewards: list[float] = []
    for cell_index, candidate_env in enumerate(eval_envs):
        model.set_env(candidate_env)
        cell_eval = evaluate_model(
            model,
            candidate_env,
            seeds=eval_seed_list,
            return_rows=True,
            row_context={
                "policy_name": "te_ppo",
                "train_seed": seed,
                "run_dir": str(run_dir),
                "validation_cell_index": cell_index,
            },
        )
        final_rewards.extend(float(value) for value in cell_eval["episode_rewards"])
        final_eval_rows.extend(cell_eval.pop("episode_rows", []))
    final_eval = {
        "mean_reward": float(np.mean(final_rewards)),
        "std_reward": float(np.std(final_rewards)),
        "episode_rewards": final_rewards,
    }
    final_eval_rows_path = eval_log_dir / "selected_validation_rows.csv"
    write_episode_rows_csv(final_eval_rows_path, final_eval_rows)

    summary = {
        "run_dir": str(run_dir),
        "selected_model_path": str(selected_model_path),
        "last_model_path": str(last_model_path),
        "checkpoints_dir": str(checkpoints_dir),
        "eval_log_dir": str(eval_log_dir),
        "log_dir": str(log_dir),
        "seed": seed,
        "n_envs": n_envs,
        "n_steps": int(algo_config.get("n_steps", 64)),
        "rollout_batch_size": n_envs * int(algo_config.get("n_steps", 64)),
        "torch_num_threads": torch.get_num_threads(),
        "total_timesteps": total_timesteps,
        "learning_rate": float(algo_config.get("learning_rate", 3e-4)),
        "gamma": float(algo_config["gamma"]),
        "gae_lambda": float(algo_config.get("gae_lambda", 1.0)),
        "clip_range": float(algo_config["clip_range"]),
        "eval_seeds": eval_seed_list,
        "selected_validation_rows_path": str(final_eval_rows_path),
        "selected_validation": final_eval,
        # Track whether SubprocVecEnv actually launched or we silently fell
        # back to DummyVecEnv (single-process). The latter is a dramatic
        # throughput regression and must be visible post-hoc.
        "vec_env_mode": getattr(train_env, "vec_env_mode", "unknown"),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    # Reproducibility snapshot (git hash, library versions, CUDA devices, ...).
    # Best-effort: wrapped in a try so training never fails because of it.
    try:
        write_run_metadata(run_dir)
    except Exception as exc:  # pragma: no cover
        print(f"[train] write_run_metadata failed: {exc}")

    train_env.close()
    for candidate_env in eval_envs:
        candidate_env.close()
    return summary


def _run_dbas_evaluation(
    config: dict[str, Any],
    *,
    output_root: str | Path | None,
) -> dict[str, Any]:
    """Run the distance-based heuristic baseline (no training)."""

    from src.baselines.dbas import evaluate_distance_policy
    from src.sim.env import RSULeaderEnv

    seed = int(config.get("seed", 123))
    seed_everything(seed)

    scenario_config = ScenarioConfig(**config.get("scenario", {}))
    eval_cfg = config.get("evaluation", {})
    seed_list = eval_cfg.get("seed_list") or [
        seed + 10_000 + idx for idx in range(int(eval_cfg.get("n_eval_episodes", 3)))
    ]
    fixed_price_fraction = float(
        config.get("algorithm", {}).get("fixed_price_fraction", 0.5)
    )

    run_dir = _run_dir(config, output_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config_snapshot.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    env = RSULeaderEnv(scenario_config=scenario_config)
    rows = evaluate_distance_policy(
        env,
        seeds=[int(s) for s in seed_list],
        fixed_price_fraction=fixed_price_fraction,
    )
    rewards = [float(row["reward"]) for row in rows]

    summary = {
        "run_dir": str(run_dir),
        "algorithm": "dbas",
        "seed": seed,
        "eval_seeds": [int(s) for s in seed_list],
        "eval_episode_rewards": rewards,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    try:
        write_run_metadata(run_dir)
    except Exception as exc:  # pragma: no cover
        print(f"[train/dbas] write_run_metadata failed: {exc}")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train leader PPO on the RSU environment")
    parser.add_argument("--config", required=True, help="Path to a YAML config")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional override for the root directory that stores run artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_training_config(args.config)
    summary = train_from_config(config, output_root=args.output_root)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
