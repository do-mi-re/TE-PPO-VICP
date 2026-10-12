"""Configuration, evaluation, and aggregation helpers for comparison figures.

Exactness label: implementation convenience only.

Evaluates matching algorithm artifacts across parameter grids and produces
per-episode metric rows suitable for aggregation and plotting.
"""

from __future__ import annotations

import copy
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.evaluation.provenance import (
    config_digest,
    validate_run_artifact,
)
from src.rl.checkpoint_selection import (
    SELECTED_VECNORMALIZE,
    selected_model_name,
)
from src.rl.eval import (
    SWEEP_METRIC_NAMES,
    episode_metrics_from_info,
    synchronize_accelerator,
    write_episode_rows_csv,
)
from src.rl.train import ALGORITHM_OPTION_KEYS, load_training_config
from src.sim.env import RSULeaderEnv
from src.sim.scenario import ScenarioConfig

ALGORITHM_NAMES = ("proposed", "jdls", "mappo", "dbas", "ebas", "spis", "rdus")

# Display labels used for plot legends and aggregated CSVs. The internal
# algorithm key stays "mappo" (code, config file, checkpoint dir), but the
# manuscript refers to it as "MAPS".
ALGORITHM_LABELS = {
    "proposed": "TE-PPO (Proposed)",
    "jdls": "JDLS",
    "mappo": "MAPS",
    "dbas": "DBAS",
    "ebas": "EBAS",
    "spis": "SPIS",
    "rdus": "RDUS",
}

EVAL_SEEDS = tuple(range(40_000, 40_008))


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge overrides into base dict (returns new dict)."""
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_algorithm_config(algorithm_name: str) -> dict[str, Any]:
    """Load the algorithm-specific YAML config."""
    config_path = Path(f"configs/algorithms/{algorithm_name}.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Algorithm config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_sweep_config(
    algorithm_name: str,
    *,
    base_config: dict[str, Any] | None = None,
    scenario_overrides: dict[str, Any] | None = None,
    train_seed: int = 123,
    eval_seeds: tuple[int, ...] = EVAL_SEEDS,
    run_name_suffix: str = "",
) -> dict[str, Any]:
    """Build a full training config for one sweep point."""
    if base_config is None:
        base_config = load_algorithm_config(algorithm_name)

    config = copy.deepcopy(base_config)
    config["seed"] = train_seed

    if scenario_overrides:
        config.setdefault("scenario", {}).update(scenario_overrides)

    config.setdefault("evaluation", {})["seed_list"] = list(eval_seeds)

    suffix = f"_{run_name_suffix}" if run_name_suffix else ""
    config.setdefault("artifacts", {})["run_name"] = (
        f"{algorithm_name}_s{train_seed}{suffix}"
    )

    return config


def _evaluate_dbas_with_metrics(
    config: dict[str, Any],
    eval_seeds: tuple[int, ...],
    *,
    scenario_override: dict[str, Any] | None = None,
) -> list[dict[str, object]]:
    """Evaluate DBAS directly and return episode metric rows."""
    from src.baselines.dbas import DistanceBasedPolicy

    scenario_dict = dict(config.get("scenario", {}))
    if scenario_override:
        scenario_dict.update(scenario_override)
    scenario_config = ScenarioConfig(**scenario_dict)
    fixed_price_fraction = float(
        config.get("algorithm", {}).get("fixed_price_fraction", 0.5)
    )
    policy = DistanceBasedPolicy(fixed_price_fraction=fixed_price_fraction)
    env = RSULeaderEnv(scenario_config=scenario_config)

    rows: list[dict[str, object]] = []
    for episode_idx, seed in enumerate(eval_seeds):
        env.reset(seed=int(seed))
        synchronize_accelerator()
        cycle_started = time.perf_counter()
        action = policy.act(env.require_state(), pricing_budget_max=scenario_config.pricing_budget_max)
        _, reward, _, _, info = env.step(action)
        synchronize_accelerator()
        row: dict[str, object] = {
            "episode_index": episode_idx,
            "eval_seed": int(seed),
            "inference_wall_time_ms": (time.perf_counter() - cycle_started) * 1000.0,
        }
        row.update(episode_metrics_from_info(dict(info), episode_reward=float(reward)))
        rows.append(row)
    env.close()
    return rows


def _evaluate_trained_model_with_metrics(
    run_dir: Path,
    eval_seeds: tuple[int, ...],
    algorithm_name: str,
    *,
    scenario_override: dict[str, Any] | None = None,
) -> list[dict[str, object]]:
    """Load a trained model and evaluate to get metric rows."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize

    from src.rl.ppo import (
        NormalizationConfig,
        RewardConfig,
        make_vectorized_eval_env,
    )

    config = load_training_config(run_dir / "config_snapshot.yaml")
    scenario_dict = dict(config.get("scenario", {}))
    if scenario_override:
        scenario_dict.update(scenario_override)
    scenario_config = ScenarioConfig(**scenario_dict)
    reward_config = RewardConfig(**config.get("reward", {}))
    normalization_config = NormalizationConfig(**config.get("normalization", {}))
    action_logit_bound = float(config.get("algorithm", {}).get("action_logit_bound", 8.0))

    env = make_vectorized_eval_env(
        scenario_config=scenario_config,
        reward_config=reward_config,
        normalization_config=normalization_config,
        action_logit_bound=action_logit_bound,
        algorithm_name=algorithm_name,
        algorithm_options={
            key: value
            for key, value in config.get("algorithm", {}).items()
            if key not in ALGORITHM_OPTION_KEYS
        },
    )
    vec_path = run_dir / SELECTED_VECNORMALIZE
    if vec_path.exists() and isinstance(env, VecNormalize):
        base_env = env.venv
        env = VecNormalize.load(str(vec_path), base_env)
        env.training = False
        env.norm_reward = False

    model = PPO.load(
        str(run_dir / selected_model_name(algorithm_name)),
        env=env,
        device=str(config.get("algorithm", {}).get("device", "cpu")),
    )

    from src.rl.eval import evaluate_model

    payload = evaluate_model(
        model, env, seeds=list(eval_seeds), return_rows=True,
    )
    rows = list(payload.get("episode_rows", []))
    env.close()
    return rows


def _evaluate_mappo_with_metrics(
    run_dir: Path,
    eval_seeds: tuple[int, ...],
    *,
    scenario_override: dict[str, Any] | None = None,
) -> list[dict[str, object]]:
    """Load a trained MAPPO model and evaluate to get metric rows."""
    import torch

    from src.rl.mappo import (
        GaussianActor,
        MAPPOLearnedPolicy,
        VehicleHybridActor,
        flatten_global_observation,
        flatten_vehicle_observation,
    )

    config = load_training_config(run_dir / "config_snapshot.yaml")
    from src.rl.train import resolve_device

    scenario_dict = dict(config.get("scenario", {}))
    if scenario_override:
        scenario_dict.update(scenario_override)
    scenario_config = ScenarioConfig(**scenario_dict)
    algo_cfg = config.get("algorithm", {})
    hidden_sizes = tuple(algo_cfg.get("hidden_sizes", (128, 128)))
    device = resolve_device(str(algo_cfg.get("device", "cpu")))

    env = RSULeaderEnv(scenario_config=scenario_config)
    obs, _ = env.reset(seed=0)

    K = scenario_config.num_tasks
    V = scenario_config.num_vehicles
    leader_action_dim = K + 1 + V
    global_dim = flatten_global_observation(obs).shape[0]
    vehicle_obs_dim = flatten_vehicle_observation(
        env.vehicle_observation(0, np.zeros(K), np.zeros(V))
    ).shape[0]

    leader_actor = GaussianActor(global_dim, leader_action_dim, hidden_sizes).to(device)
    vehicle_actor = VehicleHybridActor(vehicle_obs_dim, K, hidden_sizes).to(device)

    checkpoint = torch.load(
        run_dir / selected_model_name("mappo"), map_location=device, weights_only=True
    )
    leader_actor.load_state_dict(checkpoint["leader_actor"])
    vehicle_actor.load_state_dict(checkpoint["vehicle_actor"])

    policy = MAPPOLearnedPolicy(
        leader_actor,
        vehicle_actor,
        env,
        device,
        parameterization=str(
            algo_cfg.get(
                "joint_action_parameterization",
                "candidate_masked_feasible_v1",
            )
        ),
    )

    rows: list[dict[str, object]] = []
    for episode_idx, seed in enumerate(eval_seeds):
        obs_eval, _ = env.reset(seed=int(seed))
        synchronize_accelerator()
        cycle_started = time.perf_counter()
        joint = policy.act(obs_eval)
        _, reward, _, _, info = env.step_joint(joint)
        synchronize_accelerator()
        row: dict[str, object] = {
            "episode_index": episode_idx,
            "eval_seed": int(seed),
            "inference_wall_time_ms": (time.perf_counter() - cycle_started) * 1000.0,
        }
        row.update(episode_metrics_from_info(dict(info), episode_reward=float(reward)))
        rows.append(row)
    env.close()
    return rows


def _evaluate_jdls_with_metrics(
    run_dir: Path,
    eval_seeds: tuple[int, ...],
    *,
    scenario_override: dict[str, Any] | None = None,
) -> list[dict[str, object]]:
    """Load a trained JDLS model and evaluate to get metric rows."""
    import torch

    from src.baselines.jdls import (
        CONSTRAINT_AWARE_JOINT_ACTION_PARAMETERIZATION,
        JDLSCentralizedActor,
        JDLSLearnedPolicy,
    )
    from src.rl.mappo import flatten_global_observation
    from src.rl.train import resolve_device

    config = load_training_config(run_dir / "config_snapshot.yaml")
    scenario_dict = dict(config.get("scenario", {}))
    if scenario_override:
        scenario_dict.update(scenario_override)
    scenario_config = ScenarioConfig(**scenario_dict)
    algo_cfg = config.get("algorithm", {})
    hidden_sizes = tuple(algo_cfg.get("hidden_sizes", (256, 256)))
    device = resolve_device(str(algo_cfg.get("device", "cpu")))

    env = RSULeaderEnv(scenario_config=scenario_config)
    obs, _ = env.reset(seed=0)

    K = scenario_config.num_tasks
    V = scenario_config.num_vehicles
    global_dim = flatten_global_observation(obs).shape[0]

    actor = JDLSCentralizedActor(
        obs_dim=global_dim,
        num_tasks=K,
        num_vehicles=V,
        hidden_sizes=hidden_sizes,
    ).to(device)

    checkpoint = torch.load(
        run_dir / selected_model_name("jdls"), map_location=device, weights_only=True
    )
    actor.load_state_dict(checkpoint["actor"])

    policy = JDLSLearnedPolicy(
        actor,
        env,
        device,
        parameterization=str(
            algo_cfg.get(
                "joint_action_parameterization",
                CONSTRAINT_AWARE_JOINT_ACTION_PARAMETERIZATION,
            )
        ),
    )

    rows: list[dict[str, object]] = []
    for episode_idx, seed in enumerate(eval_seeds):
        obs_eval, _ = env.reset(seed=int(seed))
        synchronize_accelerator()
        cycle_started = time.perf_counter()
        joint = policy.act(obs_eval)
        _, reward, _, _, info = env.step_joint(joint)
        synchronize_accelerator()
        row: dict[str, object] = {
            "episode_index": episode_idx,
            "eval_seed": int(seed),
            "inference_wall_time_ms": (time.perf_counter() - cycle_started) * 1000.0,
        }
        row.update(episode_metrics_from_info(dict(info), episode_reward=float(reward)))
        rows.append(row)
    env.close()
    return rows


def evaluate_run_dir(
    algorithm_name: str,
    run_dir: Path,
    eval_seeds: tuple[int, ...],
    *,
    dbas_config: dict[str, Any] | None = None,
    scenario_override: dict[str, Any] | None = None,
    expected_config: dict[str, Any] | None = None,
) -> list[dict[str, object]]:
    """Load a trained model from ``run_dir`` and return episode metric rows.

    Pure eval — never trains. ``run_dir`` must already contain the artifacts
    the matching ``_evaluate_*_with_metrics`` helper expects (for learning
    algorithms) or be unused (DBAS, which is stateless and takes
    ``dbas_config`` instead).

    ``scenario_override``, when set, is shallow-merged onto the scenario dict
    the run was originally trained with (or onto ``dbas_config["scenario"]``)
    before constructing the eval environment — i.e. it evaluates the existing
    checkpoint/policy under a different scenario than it was trained on,
    without retraining. Used by the occlusion-level sweep, which reuses each
    algorithm's default-scenario checkpoint across occlusion bands instead of
    training one model per grid point.

    Dispatches on ``algorithm_name``:
        proposed/spis/rdus/ebas → _evaluate_trained_model_with_metrics
        mappo                   → _evaluate_mappo_with_metrics
        jdls                    → _evaluate_jdls_with_metrics
        dbas                    → _evaluate_dbas_with_metrics(dbas_config, ...)
    """
    checkpoint_name = selected_model_name(algorithm_name)
    validated_provenance: dict[str, Any] | None = None
    if algorithm_name != "dbas":
        if expected_config is None:
            raise ValueError(
                "evaluate_run_dir: expected_config is required for trained "
                "algorithms so stale or mismatched checkpoints cannot be evaluated"
            )
        validated_provenance = validate_run_artifact(
            run_dir,
            expected_config=expected_config,
            checkpoint_name=checkpoint_name,
        ).as_dict()

    if algorithm_name == "dbas":
        if dbas_config is None:
            raise ValueError(
                "evaluate_run_dir: DBAS is stateless — pass dbas_config with "
                "the scenario overrides you want to score against."
            )
        rows = _evaluate_dbas_with_metrics(
            dbas_config, eval_seeds, scenario_override=scenario_override,
        )
        stateless_config = expected_config or dbas_config
        fingerprint = config_digest(stateless_config)
        if fingerprint is None:
            raise ValueError("DBAS stateless configuration is not digestible")
        provenance = {
            "config_fingerprint": fingerprint,
            "checkpoint_sha256": "stateless",
            "normalization_sha256": None,
            "selection_sha256": "stateless",
            "artifact_kind": "stateless_config",
        }
    elif algorithm_name == "mappo":
        rows = _evaluate_mappo_with_metrics(
            run_dir, eval_seeds, scenario_override=scenario_override,
        )
        provenance = validated_provenance
    elif algorithm_name == "jdls":
        rows = _evaluate_jdls_with_metrics(
            run_dir, eval_seeds, scenario_override=scenario_override,
        )
        provenance = validated_provenance
    else:
        rows = _evaluate_trained_model_with_metrics(
            run_dir, eval_seeds, algorithm_name, scenario_override=scenario_override,
        )
        provenance = validated_provenance
    for row in rows:
        row.update(provenance)
    return rows


def tag_episode_rows(
    rows: list[dict[str, object]],
    *,
    algorithm_name: str,
    train_seed: int,
    scenario_overrides: dict[str, Any] | None = None,
    sweep_param: str | None = None,
    sweep_value: float | None = None,
) -> None:
    """Attach identity + sweep metadata to episode rows in-place.

    Callers build episode rows via ``evaluate_run_dir`` (which returns
    bare metric rows without knowing the scenario context) and then tag
    them with this helper so downstream CSV writers have consistent
    columns regardless of how the row was produced.
    """
    label = ALGORITHM_LABELS.get(algorithm_name, algorithm_name)
    for row in rows:
        row["algorithm"] = algorithm_name
        row["algorithm_label"] = label
        row["train_seed"] = train_seed
        if scenario_overrides:
            row.update({f"sweep_{k}": v for k, v in scenario_overrides.items()})
        if sweep_param is not None:
            row["sweep_param"] = sweep_param
        if sweep_value is not None:
            row["sweep_value"] = sweep_value


def parse_sweep_value(
    sweep_param: str,
    sweep_value: Any,
) -> tuple[dict[str, Any], str, float]:
    """Derive scenario overrides, run-name suffix, and display value.

    Returns (scenario_overrides, suffix, display_value).
    """
    if sweep_param == "num_tasks":
        # K is not a ScenarioConfig field -- num_tasks is a read-only property
        # over urgency_profiles -- so the sweep sets the profile list instead,
        # and widens the task corridors so K > 4 regions fit without overlap.
        # make_urgency_profiles deliberately reuses the high/mid/low labels;
        # see its docstring for why a fourth label would be dropped silently.
        from src.sim.scenario import make_urgency_profiles

        num_tasks = int(sweep_value)
        return (
            {
                "urgency_profiles": make_urgency_profiles(num_tasks),
                "task_region_span_m": 108.0,
            },
            f"k{num_tasks}",
            float(num_tasks),
        )
    overrides = {sweep_param: sweep_value}
    if sweep_param == "num_vehicles":
        suffix = f"v{sweep_value}"
    elif sweep_param == "system_bandwidth_hz":
        bandwidth_mhz = float(sweep_value) / 1e6
        bandwidth_label = f"{bandwidth_mhz:g}".replace(".", "p")
        suffix = f"bw{bandwidth_label}M"
    else:
        suffix = f"{sweep_param}_{sweep_value}"
    return overrides, suffix, float(sweep_value)


def aggregate_sweep_rows(
    rows: list[dict[str, object]],
    *,
    group_fields: tuple[str, ...] = ("algorithm", "sweep_value"),
    metrics: tuple[str, ...] = SWEEP_METRIC_NAMES,
    uncertainty: str = "scenario",
) -> list[dict[str, object]]:
    """Aggregate per-episode rows into per-(algorithm, sweep_value) summaries.

    The point estimate (``{metric}_mean``) is the mean over the full evaluation
    bank (all eval seeds, averaged across whatever train seeds are present).

    The reported uncertainty (``{metric}_std``) is the spread across the
    EVALUATION SCENARIOS: for each ``eval_seed`` we first average across the
    available ``train_seed`` replicates (one value per scenario), then take the
    std across those per-scenario values. This makes the band measure "how much
    does performance vary across the test-scenario distribution", which is
    defined identically for learned policies (multiple train seeds) and for
    deterministic baselines (a single train seed, e.g. DBAS). DBAS therefore
    gets a genuine non-zero band from its 8 eval scenarios instead of the
    degenerate zero that a train-seed-spread definition produced.

    When a metric has fewer than two valid scenarios its std is ``NaN`` (not
    ``0.0``) so the plot layer draws "no band" rather than a zero-width band
    that would read as false certainty. ``replicate_count`` retains the number
    of distinct train seeds for provenance; ``{metric}_replicates`` is the
    number of valid evaluation scenarios contributing to the std.

    TRAIN-SEED SPREAD. Each independent training seed contributes one mean:

        ``{metric}_seed_mean``       mean over per-train-seed values
        ``{metric}_seed_std``        std ACROSS train seeds
        ``{metric}_seed_ci95``       1.96 * seed_std / sqrt(n_seeds)
        ``{metric}_seed_replicates`` number of train seeds contributing

    Each train seed first averages over its own evaluation bank (one value per
    seed), then the std is taken across those per-seed values. Deterministic
    baselines with a single train seed (e.g. DBAS) get ``NaN`` here by
    construction -- there is no retraining variability to measure -- which is
    the honest answer rather than a zero-width band.

    ``uncertainty`` selects which definition the plot-facing ``{metric}_std``
    key carries: ``"scenario"`` (default, preserves existing behaviour) or
    ``"seed"``. Both families are always written, so the choice only affects
    which one the plot layer picks up without further changes.
    """
    if uncertainty not in ("scenario", "seed"):
        raise ValueError(
            f"uncertainty must be 'scenario' or 'seed', got {uncertainty!r}"
        )
    final_groups: dict[tuple, list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(row.get(field, "") for field in group_fields)
        final_groups.setdefault(key, []).append(row)

    # Track how often a metric was missing / NaN so downstream readers can tell
    # "no data" from "zero". See B4 in /plans/sharded-swimming-stardust.md.
    missing_counts: dict[str, int] = {metric: 0 for metric in metrics}

    final_rows: list[dict[str, object]] = []
    for _key, group in final_groups.items():
        record: dict[str, object] = {
            field: group[0].get(field) for field in group_fields
        }
        record["algorithm_label"] = ALGORITHM_LABELS.get(
            str(record.get("algorithm", "")), str(record.get("algorithm", ""))
        )
        record["replicate_count"] = len({row.get("train_seed", "") for row in group})

        # Group this (algorithm, sweep_value) bucket's episodes by scenario
        # (eval_seed) so the std is computed across scenarios, not train seeds.
        by_scenario: dict[object, list[dict[str, object]]] = {}
        for row in group:
            by_scenario.setdefault(row.get("eval_seed", ""), []).append(row)

        # Mirror grouping by train seed: one value per independently trained
        # replicate, which is what seed-level variability asks the error band to describe.
        by_train_seed: dict[object, list[dict[str, object]]] = {}
        for row in group:
            by_train_seed.setdefault(row.get("train_seed", ""), []).append(row)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for metric in metrics:
                episode_values = [float(row.get(metric, float("nan"))) for row in group]
                missing_counts[metric] += sum(1 for v in episode_values if np.isnan(v))

                # One value per scenario = mean across train seeds for that eval seed.
                scenario_means: list[float] = []
                for scenario_rows in by_scenario.values():
                    vals = [float(r.get(metric, float("nan"))) for r in scenario_rows]
                    valid = [v for v in vals if not np.isnan(v)]
                    if valid:
                        scenario_means.append(float(np.mean(valid)))

                if scenario_means:
                    record[f"{metric}_mean"] = float(np.mean(scenario_means))
                    record[f"{metric}_std"] = (
                        float(np.std(scenario_means))
                        if len(scenario_means) > 1
                        else float("nan")
                    )
                else:
                    record[f"{metric}_mean"] = float("nan")
                    record[f"{metric}_std"] = float("nan")
                record[f"{metric}_replicates"] = len(scenario_means)
                record[f"{metric}_valid_count"] = sum(
                    1 for v in episode_values if not np.isnan(v)
                )

                # One value per train seed = mean over that seed's eval bank.
                seed_means: list[float] = []
                for seed_rows in by_train_seed.values():
                    vals = [float(r.get(metric, float("nan"))) for r in seed_rows]
                    valid = [v for v in vals if not np.isnan(v)]
                    if valid:
                        seed_means.append(float(np.mean(valid)))

                if seed_means:
                    record[f"{metric}_seed_mean"] = float(np.mean(seed_means))
                    if len(seed_means) > 1:
                        # ddof=1: these are a sample of the seed population,
                        # not the population itself.
                        seed_std = float(np.std(seed_means, ddof=1))
                        record[f"{metric}_seed_std"] = seed_std
                        record[f"{metric}_seed_ci95"] = float(
                            1.96 * seed_std / np.sqrt(len(seed_means))
                        )
                    else:
                        record[f"{metric}_seed_std"] = float("nan")
                        record[f"{metric}_seed_ci95"] = float("nan")
                else:
                    record[f"{metric}_seed_mean"] = float("nan")
                    record[f"{metric}_seed_std"] = float("nan")
                    record[f"{metric}_seed_ci95"] = float("nan")
                record[f"{metric}_seed_replicates"] = len(seed_means)

                if uncertainty == "seed":
                    record[f"{metric}_std"] = record[f"{metric}_seed_std"]
        final_rows.append(record)

    for metric, count in missing_counts.items():
        if count > 0:
            warnings.warn(
                f"aggregate_sweep_rows: metric '{metric}' was missing/NaN in "
                f"{count} episode row(s); reported means use the remaining samples.",
                RuntimeWarning,
                stacklevel=2,
            )

    return sorted(
        final_rows,
        key=lambda r: (float(r.get("sweep_value", 0)), str(r.get("algorithm", ""))),
    )


def write_sweep_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    """Write sweep result rows to CSV."""
    write_episode_rows_csv(path, rows)
