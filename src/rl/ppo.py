"""SB3-facing wrappers and builders for the leader PPO stack.

Exactness note:
- Paper-exact: the latent action is decoded into the
  leader's direct decision variables as specified by the maintained
  manuscript, and the TE-PPO reward wrapper implements its penalty terms using
  the follower outcomes returned by the environment.
- Paper-consistent approximation: the wrappers adapt the existing single-step
  environment to the interfaces expected by Stable-Baselines3.
- Implementation convenience only: the observation is flattened to a one-level
  dict and cast to ``float32`` for SB3 compatibility.
"""

from __future__ import annotations

import multiprocessing
import os
import time
import warnings
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecMonitor,
    VecNormalize,
    sync_envs_normalization,
)

from src.rl.checkpoint_selection import (
    SB3_SELECTED_MODEL,
    SELECTED_VECNORMALIZE,
    SELECTION_MANIFEST,
    EvaluationSchedule,
    RobustCheckpointScore,
    atomic_write_json,
    robust_checkpoint_score,
    should_replace_checkpoint,
    should_replace_robust_checkpoint,
    validation_grid_digest,
)
from src.rl.common_reward import (
    COMMON_EVALUATION_REWARD_SCHEMA,
    compute_common_evaluation_reward,
)
from src.rl.eval import evaluate_model
from src.rl.obs_flatten import FlattenLeaderObservationWrapper
from src.rl.policy import (
    ActorNetworkConfig,
    build_policy_kwargs,
    decode_leader_action,
    latent_action_dim,
)
from src.rl.progress import TrainingProgress
from src.rl.value import CriticNetworkConfig
from src.sim.env import RSULeaderEnv, ScenarioFactory
from src.sim.scenario import ScenarioConfig

PPO_MONITOR_INFO_KEYWORDS: tuple[str, ...] = (
    "raw_rsu_reward",
    "te_ppo_reward",
    "penalty_fail",
    "penalty_lat",
    "penalty_part",
    "task_failure_count",
    "participant_deficit_total",
    "latency_overshoot_total",
)


def monitor_info_keywords() -> tuple[str, ...]:
    """Return the SB3 Monitor ``info_keywords`` tuple for the TE-PPO family.

    All SB3-path baselines (proposed / rdus / ebas / spis) share the
    :class:`TEPPORewardWrapper` info schema, so this is a single constant. JDLS
    and MAPPO have their own custom PPO loops (see ``src.baselines.jdls`` and
    ``src.rl.mappo``) and do not use SB3's Monitor.
    """

    return PPO_MONITOR_INFO_KEYWORDS


@dataclass(frozen=True)
class RewardConfig:
    """Penalty weights for the paper's TE-PPO reward.

    ``omega_vehicle`` is ignored by :class:`TEPPORewardWrapper` and only
    consumed by joint-learning baselines (JDLS/MAPPO) for their pre-filter
    conditional-utility shaping term.
    """

    omega_fail: float = 2.0
    omega_lat: float = 3.0
    omega_part: float = 3.0
    min_participants: int = 1
    omega_vehicle: float = 0.0
    omega_feasible: float = 0.0
    omega_pre_lat: float = 0.0
    pre_lat_overshoot_cap: float = 10.0


@dataclass(frozen=True)
class NormalizationConfig:
    """Training-only normalization options for SB3 wrappers."""

    norm_obs: bool = True
    norm_reward: bool = True
    clip_obs: float = 10.0
    clip_reward: float = 10.0


# ``FlattenLeaderObservationWrapper`` is re-exported from
# :mod:`src.rl.obs_flatten` so every wrapper stack — TE-PPO, MAPPO, and all
# baselines — consumes the exact same leader-observation layout from one place.


class LeaderLatentActionWrapper(gymnasium.ActionWrapper):
    """Expose the PPO actor's raw logit vector while preserving paper action semantics."""

    def __init__(
        self,
        env: gymnasium.Env[dict[str, object], Any],
        *,
        num_tasks: int,
        num_vehicles: int,
        pricing_budget_max: float,
        action_logit_bound: float = 8.0,
    ) -> None:
        super().__init__(env)
        self.num_tasks = num_tasks
        self.num_vehicles = num_vehicles
        self.pricing_budget_max = pricing_budget_max
        if action_logit_bound <= 0.0:
            raise ValueError("action_logit_bound must be positive")
        self.action_logit_bound = action_logit_bound
        dim = latent_action_dim(num_tasks, num_vehicles)
        self.action_space = spaces.Box(
            low=-action_logit_bound,
            high=action_logit_bound,
            shape=(dim,),
            dtype=np.float32,
        )

    def action(self, action: np.ndarray) -> dict[str, np.ndarray]:
        base_env = self.env.unwrapped
        if not isinstance(base_env, RSULeaderEnv):
            raise TypeError("LeaderLatentActionWrapper requires RSULeaderEnv")
        decoded = decode_leader_action(
            np.asarray(action, dtype=np.float64),
            num_tasks=self.num_tasks,
            num_vehicles=self.num_vehicles,
            pricing_budget_max=self.pricing_budget_max,
            eligible_vehicle_mask=base_env.candidate_vehicle_mask(),
        )
        return {
            "prices": np.asarray(decoded.prices, dtype=np.float64),
            "bandwidth": np.asarray(decoded.bandwidth, dtype=np.float64),
        }


class TEPPORewardWrapper(gymnasium.Wrapper):
    """Compose the paper's TE-PPO penalty reward on top of the raw RSU utility.

    Uses the public ``RSULeaderEnv.require_state()`` accessor to read task
    metadata (task weights :math:`\\beta_k` and deadlines
    :math:`\\tau_{\\max}`) — no private ``_state`` peeking.
    """

    def __init__(
        self,
        env: gymnasium.Env[dict[str, object], Any],
        *,
        reward_config: RewardConfig,
        lean_info: bool = False,
    ) -> None:
        super().__init__(env)
        self.reward_config = reward_config
        # When True (training envs), drop the heavy per-step ``info`` payload
        # (``observation_snapshot`` obs-clone + the full ``follower_responses``
        # V*K TaskEvaluation tree) after the shaping penalties are computed,
        # keeping only the scalar :data:`PPO_MONITOR_INFO_KEYWORDS`. Episodes
        # are single-step, so SB3 deepcopies (DummyVecEnv) / pickles
        # (SubprocVecEnv) this dict on EVERY step — profiling showed that copy
        # is ~35% (deepcopy) / ~40% (IPC) of the leader's per-step cost. Eval
        # envs keep ``lean_info=False`` so ``episode_metrics_from_info`` still
        # sees the full schema.
        self.lean_info = lean_info

    def _task_metadata(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return ``(task_weights, task_deadlines)`` from the base env's public API."""

        state = self.unwrapped.require_state()
        weights = tuple(float(task.beta) for task in state.tasks)
        deadlines = tuple(float(task.tau_max_s) for task in state.tasks)
        return weights, deadlines

    def _compute_penalties(self, info: dict[str, object]) -> dict[str, float]:
        task_weights, task_deadlines = self._task_metadata()
        _, diagnostics = compute_common_evaluation_reward(
            0.0,
            info,
            config=self.reward_config,
            task_weights=task_weights,
            task_deadlines=task_deadlines,
        )
        return diagnostics

    def step(self, action: Any) -> tuple[dict[str, object], float, bool, bool, dict[str, object]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        penalties = self._compute_penalties(info)
        te_ppo_reward = reward - penalties["penalty_fail"] - penalties["penalty_lat"] - penalties["penalty_part"]
        if self.lean_info:
            # Training fast path: emit ONLY the scalar monitor keywords. These
            # are exactly PPO_MONITOR_INFO_KEYWORDS, so VecMonitor/logging is
            # unchanged while the heavy nested payload never reaches SB3's
            # per-step deepcopy/pickle. (See __init__ for the why.)
            lean = {"raw_rsu_reward": float(reward), "te_ppo_reward": float(te_ppo_reward)}
            lean.update({key: float(value) for key, value in penalties.items()})
            return observation, float(te_ppo_reward), terminated, truncated, lean
        info = dict(info)
        info["raw_rsu_reward"] = float(reward)
        info["te_ppo_reward"] = float(te_ppo_reward)
        info.update({key: float(value) for key, value in penalties.items()})
        return observation, float(te_ppo_reward), terminated, truncated, info


class FixedSeedEvalCallback(BaseCallback):
    """Evaluate PPO on a frozen scenario seed bank and save the best checkpoint."""

    def __init__(
        self,
        *,
        eval_env: VecEnv,
        eval_seeds: list[int],
        run_dir: Path,
        eval_log_dir: Path,
        interval_env_steps: int,
        config_fingerprint: str,
        force_terminal_evaluation: bool = True,
        progress: TrainingProgress | None = None,
        cpu_eval_envs: list[VecEnv] | None = None,
        cpu_validation_ranges: tuple[tuple[float, float], ...] = (),
    ) -> None:
        super().__init__(verbose=0)
        if eval_env.num_envs != 1:
            raise ValueError("FixedSeedEvalCallback requires a single-env VecEnv")
        if not eval_seeds:
            raise ValueError("eval_seeds must not be empty")
        self.eval_env = eval_env
        self.eval_envs = list(cpu_eval_envs or [eval_env])
        if any(candidate.num_envs != 1 for candidate in self.eval_envs):
            raise ValueError("every fixed-seed evaluation cell requires one environment")
        if cpu_validation_ranges and len(cpu_validation_ranges) != len(self.eval_envs):
            raise ValueError("CPU validation ranges must match evaluation environments")
        self.cpu_validation_ranges = tuple(cpu_validation_ranges)
        self.eval_seeds = list(eval_seeds)
        self.run_dir = run_dir
        self.eval_log_dir = eval_log_dir
        self.config_fingerprint = config_fingerprint
        self.schedule = EvaluationSchedule(
            interval_env_steps=interval_env_steps,
            force_terminal_evaluation=force_terminal_evaluation,
        )
        self.progress = progress
        self.best_mean_reward = -np.inf
        self.best_robust_score: RobustCheckpointScore | None = None
        self.evaluations_timesteps: list[int] = []
        self.evaluations_results: list[list[float]] = []
        self.evaluations_raw_rsu_reward: list[list[float]] = []
        self.evaluations_common_reward: list[list[float]] = []
        self.evaluations_length: list[list[int]] = []
        self.evaluations_social_welfare: list[list[float]] = []
        self.evaluations_wall_time_s: list[float] = []
        self._training_started = time.monotonic()

    def _on_step(self) -> bool:
        if not self.schedule.periodic_due(int(self.num_timesteps)):
            return True

        self._evaluate_and_select()
        return True

    def _on_training_end(self) -> None:
        if self.schedule.terminal_due(int(self.num_timesteps)):
            self._evaluate_and_select()

    def _evaluate_and_select(self) -> None:

        combined_rewards: list[float] = []
        combined_lengths: list[int] = []
        combined_welfare: list[float] = []
        combined_raw_rsu_reward: list[float] = []
        combined_common_reward: list[float] = []
        rows: list[dict[str, Any]] = []
        for cell_index, candidate_env in enumerate(self.eval_envs):
            if isinstance(self.training_env, VecNormalize) and isinstance(
                candidate_env, VecNormalize
            ):
                sync_envs_normalization(self.training_env, candidate_env)
            result = evaluate_model(
                self.model, candidate_env, seeds=self.eval_seeds, return_rows=True
            )
            combined_rewards.extend(float(value) for value in result["episode_rewards"])
            combined_lengths.extend(int(value) for value in result["episode_lengths"])
            cell_rows = list(result.get("episode_rows", []))
            for row in cell_rows:
                row["validation_cell_index"] = cell_index
            rows.extend(cell_rows)
            combined_welfare.extend(
                float(row.get("social_welfare", 0.0)) for row in cell_rows
            )
            combined_raw_rsu_reward.extend(
                float(row["raw_rsu_reward"]) for row in cell_rows
            )
            combined_common_reward.extend(
                float(row["te_ppo_reward"]) for row in cell_rows
            )
        self.evaluations_timesteps.append(int(self.num_timesteps))
        self.evaluations_results.append(combined_rewards)
        self.evaluations_raw_rsu_reward.append(combined_raw_rsu_reward)
        self.evaluations_common_reward.append(combined_common_reward)
        self.evaluations_length.append(combined_lengths)
        self.evaluations_social_welfare.append(combined_welfare)
        self.evaluations_wall_time_s.append(time.monotonic() - self._training_started)
        np.savez(
            self.eval_log_dir / "evaluations.npz",
            timesteps=np.asarray(self.evaluations_timesteps, dtype=np.int64),
            results=np.asarray(self.evaluations_results, dtype=np.float64),
            raw_rsu_reward=np.asarray(
                self.evaluations_raw_rsu_reward, dtype=np.float64
            ),
            common_evaluation_reward=np.asarray(
                self.evaluations_common_reward, dtype=np.float64
            ),
            reward_schema=np.asarray(COMMON_EVALUATION_REWARD_SCHEMA),
            ep_lengths=np.asarray(self.evaluations_length, dtype=np.int64),
            social_welfare=np.asarray(self.evaluations_social_welfare, dtype=np.float64),
            wall_time_s=np.asarray(self.evaluations_wall_time_s, dtype=np.float64),
            wall_time_schema=np.asarray("monotonic_training_loop_v2"),
            validation_cell_index=np.tile(
                np.repeat(np.arange(len(self.eval_envs)), len(self.eval_seeds)),
                (len(self.evaluations_timesteps), 1),
            ),
            cpu_validation_ranges=np.asarray(
                self.cpu_validation_ranges, dtype=np.float64
            ),
        )
        # Select by validation social welfare, independently of shaping.
        robust_score = None
        if self.cpu_validation_ranges:
            robust_score = robust_checkpoint_score(
                combined_welfare,
                cell_count=len(self.eval_envs),
                seeds_per_cell=len(self.eval_seeds),
            )
            selection_metric = robust_score.minimum_cell_social_welfare
        else:
            selection_metric = float(np.mean(combined_welfare))
        if self.progress is not None:
            self.progress.set_social_welfare(float(selection_metric))
        service_valid = float(np.mean([
            float(row.get("service_valid", 0.0)) for row in rows
        ])) if rows else 0.0
        replace = (
            should_replace_robust_checkpoint(self.best_robust_score, robust_score)
            if robust_score is not None
            else should_replace_checkpoint(
                None if not np.isfinite(self.best_mean_reward) else {
                    "social_welfare": self.best_mean_reward
                },
                social_welfare=float(selection_metric),
            )
        )
        if replace:
            self.best_mean_reward = float(selection_metric)
            self.best_robust_score = robust_score
            self._save_selected_checkpoint(service_valid=service_valid)

    def _save_selected_checkpoint(self, *, service_valid: float) -> None:
        from src.evaluation.provenance import sha256_file

        model_path = self.run_dir / SB3_SELECTED_MODEL
        model_tmp = self.run_dir / f".{SB3_SELECTED_MODEL}.{os.getpid()}.tmp.zip"
        self.model.save(str(model_tmp))
        os.replace(model_tmp, model_path)

        normalization_path: Path | None = None
        normalization_sha256: str | None = None
        normalization_env = self.eval_envs[0]
        if isinstance(normalization_env, VecNormalize):
            normalization_path = self.run_dir / SELECTED_VECNORMALIZE
            norm_tmp = self.run_dir / (
                f".{SELECTED_VECNORMALIZE}.{os.getpid()}.tmp"
            )
            normalization_env.training = False
            normalization_env.norm_reward = False
            normalization_env.save(str(norm_tmp))
            os.replace(norm_tmp, normalization_path)
            normalization_sha256 = sha256_file(normalization_path)

        atomic_write_json(
            self.run_dir / SELECTION_MANIFEST,
            {
                "schema_version": 2 if self.cpu_validation_ranges else 1,
                "selection_metric": "social_welfare",
                "selection_rule": (
                    "maximin_cpu_band_social_welfare"
                    if self.cpu_validation_ranges else "mean_social_welfare"
                ),
                "selected_timestep": int(self.num_timesteps),
                "selected_metric_value": self.best_mean_reward,
                "service_valid": float(service_valid),
                "validation_seeds": self.eval_seeds,
                "cpu_validation_ranges": [
                    list(value) for value in self.cpu_validation_ranges
                ],
                "validation_grid_digest": (
                    validation_grid_digest(self.cpu_validation_ranges)
                    if self.cpu_validation_ranges else None
                ),
                "selected_macro_mean_social_welfare": (
                    self.best_robust_score.macro_mean_social_welfare
                    if self.best_robust_score is not None else self.best_mean_reward
                ),
                "selected_cell_social_welfare_means": (
                    list(self.best_robust_score.cell_social_welfare_means)
                    if self.best_robust_score is not None else [self.best_mean_reward]
                ),
                "config_fingerprint": self.config_fingerprint,
                "model_file": model_path.name,
                "model_sha256": sha256_file(model_path),
                "normalization_file": (
                    normalization_path.name if normalization_path is not None else None
                ),
                "normalization_sha256": normalization_sha256,
            },
        )


class TrainingProgressCallback(BaseCallback):
    """Mirror SB3's absolute environment-step counter to the UI tracker."""

    def __init__(self, progress: TrainingProgress) -> None:
        super().__init__(verbose=0)
        self.progress = progress

    def _on_step(self) -> bool:
        # Updating only at rollout boundaries avoids a Python/tqdm call on
        # every vector step.  _on_training_end performs the final update.
        return True

    def _on_rollout_end(self) -> None:
        self.progress.update(int(self.num_timesteps))

    def _on_training_end(self) -> None:
        self.progress.update(int(self.num_timesteps))


def build_ppo_env(
    base_env: gymnasium.Env[dict[str, object], Any],
    *,
    scenario_config: ScenarioConfig,
    reward_config: RewardConfig,
    action_logit_bound: float,
    options: dict[str, Any],
    lean_info: bool = False,
) -> gymnasium.Env[dict[str, object], Any]:
    """Assemble the TE-PPO / RDUS wrapper stack.

    RDUS and the proposed method share the same SB3 action layout; they
    differ only in scenario config (RDUS sets ``compression_override=1.0``
    via ``scenario.raw_upload_mode``).
    """

    del options  # the TE-PPO path consumes no algorithm-specific options
    env = TEPPORewardWrapper(base_env, reward_config=reward_config, lean_info=lean_info)
    env = LeaderLatentActionWrapper(
        env,
        num_tasks=scenario_config.num_tasks,
        num_vehicles=scenario_config.num_vehicles,
        pricing_budget_max=scenario_config.pricing_budget_max,
        action_logit_bound=action_logit_bound,
    )
    return FlattenLeaderObservationWrapper(env)


# Registry of algorithm -> env builder. Kept here (rather than in a separate
# file) so adding a new baseline means touching exactly one dispatch dict.
# Callers of :func:`_make_training_env_core` should extend this dict; MAPPO
# and DBAS live outside the SB3 PPO path and are dispatched earlier.
def _algorithm_builders() -> dict[str, Any]:
    from src.baselines import ebas as _ebas
    from src.baselines import spis as _spis

    return {
        "proposed": build_ppo_env,
        "rdus": build_ppo_env,
        "ebas": _ebas.build_env,
        "spis": _spis.build_env,
    }


def _make_training_env_core(
    *,
    scenario_config: ScenarioConfig,
    reward_config: RewardConfig,
    scenario_factory: ScenarioFactory | None = None,
    action_logit_bound: float = 8.0,
    algorithm_name: str = "proposed",
    algorithm_options: dict[str, Any] | None = None,
    lean_info: bool = False,
) -> gymnasium.Env[dict[str, object], Any]:
    """Build the paper-faithful env plus RL wrappers, without monitoring.

    ``algorithm_name`` is looked up in :func:`_algorithm_builders`, which
    contains one entry per SB3 PPO-compatible algorithm (proposed / rdus /
    jdls / ebas / spis). MAPPO and DBAS have their own training paths and do
    not pass through this function.

    ``lean_info=True`` (training envs only) drops the heavy per-step ``info``
    payload via :class:`TEPPORewardWrapper`; eval envs leave it ``False`` so
    metric extraction sees the full schema.
    """

    options = dict(algorithm_options or {})
    name = algorithm_name.lower()

    builders = _algorithm_builders()
    builder = builders.get(name)
    if builder is None:
        raise ValueError(f"unsupported algorithm for SB3 PPO path: {algorithm_name}")

    base: gymnasium.Env[dict[str, object], Any] = RSULeaderEnv(
        scenario_config=scenario_config,
        scenario_factory=scenario_factory,
    )
    return builder(
        base,
        scenario_config=scenario_config,
        reward_config=reward_config,
        action_logit_bound=action_logit_bound,
        options=options,
        lean_info=lean_info,
    )


def make_training_env(
    *,
    scenario_config: ScenarioConfig,
    reward_config: RewardConfig,
    scenario_factory: ScenarioFactory | None = None,
    seed: int | None = None,
    monitor_path: str | None = None,
    action_logit_bound: float = 8.0,
    algorithm_name: str = "proposed",
    algorithm_options: dict[str, Any] | None = None,
) -> Monitor:
    """Build one monitored SB3-compatible training env from the paper-faithful base env."""

    monitored = Monitor(
        _make_training_env_core(
            scenario_config=scenario_config,
            reward_config=reward_config,
            scenario_factory=scenario_factory,
            action_logit_bound=action_logit_bound,
            algorithm_name=algorithm_name,
            algorithm_options=algorithm_options,
        ),
        filename=monitor_path,
        info_keywords=monitor_info_keywords(),
    )
    if seed is not None:
        monitored.reset(seed=seed)
        monitored.action_space.seed(seed)
        monitored.observation_space.seed(seed)
    return monitored


def make_vectorized_training_env(
    *,
    scenario_config: ScenarioConfig,
    reward_config: RewardConfig,
    n_envs: int,
    normalization_config: NormalizationConfig | None = None,
    action_logit_bound: float = 8.0,
    monitor_path: str | None = None,
    algorithm_name: str = "proposed",
    algorithm_options: dict[str, Any] | None = None,
) -> VecEnv:
    """Build the PPO training VecEnv for one-step contextual-bandit updates.

    Callers seed the returned VecEnv explicitly before its first reset.
    """

    if n_envs <= 0:
        raise ValueError("n_envs must be positive")

    env_fns = [
        partial(
            _make_training_env_core,
            scenario_config=scenario_config,
            reward_config=reward_config,
            action_logit_bound=action_logit_bound,
            algorithm_name=algorithm_name,
            algorithm_options=algorithm_options,
            # Training only: drop the heavy per-step info payload before it hits
            # SB3's per-step deepcopy/pickle. The eval env builder below keeps
            # the default (full info) for metric extraction.
            lean_info=True,
        )
        for _ in range(n_envs)
    ]
    vec_env_mode = "subproc"
    try:
        _children_before = set(multiprocessing.active_children())
        env: VecEnv = SubprocVecEnv(env_fns)
    except OSError as exc:
        # Terminate worker processes that SubprocVecEnv may have spawned
        # before the error.  They are daemon processes so they would
        # eventually die with the main process, but killing them now
        # avoids wasted memory during a long training session.
        for proc in set(multiprocessing.active_children()) - _children_before:
            proc.terminate()
            proc.join(timeout=5)

        # Opt-in strict mode: a silent subproc->dummy fallback collapses an
        # n_envs-wide run to a single-threaded one (the biggest throughput
        # regression there is). On a long batch run, set TEPPO_STRICT_VECENV=1
        # to fail loudly at startup instead of limping along single-threaded.
        if os.environ.get("TEPPO_STRICT_VECENV", "").strip().lower() not in ("", "0", "false", "no"):
            raise RuntimeError(
                f"SubprocVecEnv construction failed ({exc}) and TEPPO_STRICT_VECENV "
                "is set, so refusing the single-threaded DummyVecEnv fallback. "
                "Unset TEPPO_STRICT_VECENV to allow the fallback, or fix the cause "
                "(often the multiprocessing start method or an fd/shared-memory limit)."
            ) from exc

        warnings.warn(
            f"SubprocVecEnv failed ({exc}); falling back to DummyVecEnv. "
            "Set TEPPO_STRICT_VECENV=1 to make this a hard error instead.",
            stacklevel=2,
        )
        env = DummyVecEnv(env_fns)
        vec_env_mode = "dummy_fallback"
    env = VecMonitor(
        env,
        filename=monitor_path,
        info_keywords=monitor_info_keywords(),
    )
    if normalization_config is not None and (
        normalization_config.norm_obs or normalization_config.norm_reward
    ):
        env = VecNormalize(
            env,
            norm_obs=normalization_config.norm_obs,
            norm_reward=normalization_config.norm_reward,
            clip_obs=normalization_config.clip_obs,
            clip_reward=normalization_config.clip_reward,
            training=True,
        )
    # Expose the chosen execution mode on the outermost env so
    # ``train_from_config`` can record it in ``summary.json`` (a silent
    # subproc→dummy fallback is the single biggest throughput regression we
    # want to catch post-hoc).
    env.vec_env_mode = vec_env_mode  # type: ignore[attr-defined]
    return env


def make_vectorized_eval_env(
    *,
    scenario_config: ScenarioConfig,
    reward_config: RewardConfig,
    normalization_config: NormalizationConfig | None = None,
    action_logit_bound: float = 8.0,
    monitor_path: str | None = None,
    algorithm_name: str = "proposed",
    algorithm_options: dict[str, Any] | None = None,
) -> VecEnv:
    """Build the single-env evaluation VecEnv used by fixed-seed callbacks.

    Callers seed the returned VecEnv explicitly before its first reset.
    """

    env: VecEnv = DummyVecEnv(
        [
            lambda: _make_training_env_core(
                scenario_config=scenario_config,
                reward_config=reward_config,
                action_logit_bound=action_logit_bound,
                algorithm_name=algorithm_name,
                algorithm_options=algorithm_options,
            )
        ]
    )
    env = VecMonitor(
        env,
        filename=monitor_path,
        info_keywords=monitor_info_keywords(),
    )
    if normalization_config is not None and normalization_config.norm_obs:
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=normalization_config.clip_obs,
            clip_reward=normalization_config.clip_reward,
            training=False,
        )
    return env


def build_callbacks(
    *,
    checkpoint_dir: Path,
    run_dir: Path,
    eval_log_dir: Path,
    eval_env: VecEnv,
    eval_seeds: list[int],
    checkpoint_freq: int,
    interval_env_steps: int,
    config_fingerprint: str,
    force_terminal_evaluation: bool = True,
    progress: TrainingProgress | None = None,
    cpu_eval_envs: list[VecEnv] | None = None,
    cpu_validation_ranges: tuple[tuple[float, float], ...] = (),
) -> CallbackList:
    """Build the default checkpoint and evaluation callbacks."""

    callbacks: list[BaseCallback] = [
        CheckpointCallback(
            save_freq=max(1, checkpoint_freq),
            save_path=str(checkpoint_dir),
            name_prefix="rl_model",
        ),
        FixedSeedEvalCallback(
            eval_env=eval_env,
            eval_seeds=eval_seeds,
            run_dir=run_dir,
            eval_log_dir=eval_log_dir,
            interval_env_steps=interval_env_steps,
            config_fingerprint=config_fingerprint,
            force_terminal_evaluation=force_terminal_evaluation,
            progress=progress,
            cpu_eval_envs=cpu_eval_envs,
            cpu_validation_ranges=cpu_validation_ranges,
        ),
    ]
    if progress is not None:
        callbacks.append(TrainingProgressCallback(progress))
    return CallbackList(callbacks)


def build_ppo_model(
    *,
    env: gymnasium.Env[dict[str, object], Any],
    actor_config: ActorNetworkConfig,
    critic_config: CriticNetworkConfig,
    learning_rate: float,
    n_steps: int,
    batch_size: int,
    n_epochs: int,
    gamma: float,
    gae_lambda: float,
    clip_range: float,
    clip_range_vf: float | None,
    ent_coef: float,
    vf_coef: float,
    max_grad_norm: float,
    target_kl: float | None,
    seed: int,
    device: str,
) -> PPO:
    """Construct an SB3 PPO model with explicit actor and critic branches."""

    return PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=build_policy_kwargs(actor=actor_config, critic=critic_config),
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        clip_range_vf=clip_range_vf,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        target_kl=target_kl,
        seed=seed,
        device=device,
        # Terminal progress is handled by TrainingProgressCallback.  Keeping
        # SB3 quiet suppresses repeated metric tables and expected target-KL
        # early-stop notices; CSV/JSON metrics remain enabled by train.py.
        verbose=0,
    )
