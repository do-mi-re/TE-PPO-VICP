"""MAPPO baseline: leader actor + shared vehicle actor + centralized critic.

Implements a lightweight multi-agent PPO trainer that operates on the
:class:`~src.sim.env.RSULeaderEnv`. The leader actor produces the same
``K + 1 + V`` latent vector decoded by the TE-PPO baseline, while a shared
vehicle actor receives a per-vehicle observation conditioned on the leader's
decisions and outputs each vehicle's sensing duration, compression ratio, and
task assignment. The environment then evaluates the assembled joint action via
``RSULeaderEnv.step_joint`` (no follower best-response solver involved).

The episode horizon is one step (contextual bandit), matching the rest of the
project, so the critic just predicts the expected step return and advantages
reduce to ``reward - V(s)``.

Exactness labels
----------------
- Paper-consistent approximation: MAPPO is a learning-based comparison method,
  not part of the proposed analytical follower solution.
- Implementation convenience only: rollout scenarios and all shared-actor
  vehicle decisions are evaluated in batches.  Batching preserves the product
  policy and PPO objective; it only changes the order in which random samples
  are drawn.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.distributions import Categorical, Normal

from src.baselines.joint_learning_reward import (
    JointLearningShapingConfig,
    compute_joint_learning_shaping,
)
from src.evaluation.provenance import config_digest, sha256_file
from src.rl.checkpoint_selection import (
    CUSTOM_LAST_MODEL,
    CUSTOM_SELECTED_MODEL,
    SELECTION_MANIFEST,
    EvaluationSchedule,
    RobustCheckpointScore,
    atomic_write_json,
    cpu_validation_ranges,
    resolve_evaluation_interval,
    robust_checkpoint_score,
    should_replace_checkpoint,
    should_replace_robust_checkpoint,
    validation_grid_digest,
)
from src.rl.common_reward import (
    COMMON_EVALUATION_REWARD_SCHEMA,
    compute_common_evaluation_reward,
)
from src.rl.eval import episode_metrics_from_info
from src.rl.joint_action import (
    CONSTRAINT_AWARE_JOINT_ACTION_PARAMETERIZATION,
    decode_joint_action,
    decode_leader_latent,
)
from src.rl.obs_flatten import flatten_leader_observation_to_vector
from src.rl.policy import price_logit_for_fraction
from src.rl.progress import TrainingProgress
from src.rl.train import resolve_device
from src.rl.train import seed_everything as _set_seed
from src.sim.env import JointAction, RSULeaderEnv
from src.sim.scenario import ScenarioConfig

# Short alias used throughout this module and re-exported for JDLS and sweep
# consumers; the underlying implementation lives in src.rl.obs_flatten.
flatten_global_observation = flatten_leader_observation_to_vector


def flatten_vehicle_observation(obs: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [np.asarray(value, dtype=np.float32).reshape(-1) for value in obs.values()],
        axis=0,
    )


def vehicle_observation_matrix(
    env: RSULeaderEnv,
    prices: np.ndarray,
    bandwidth: np.ndarray,
) -> np.ndarray:
    """Return the ``(V, d_vehicle)`` shared-actor input matrix.

    This is an implementation-only batching helper.  The environment builds
    all rows in one NumPy operation while preserving the scalar observation
    definition and feature order.
    """

    return env.vehicle_observation_matrix(prices, bandwidth)


def _vehicle_rows_by_step(
    *,
    num_steps: int,
    num_vehicles: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return the step-major lookup table for flattened vehicle rows."""

    return torch.arange(
        num_steps * num_vehicles,
        dtype=torch.long,
        device=device,
    ).reshape(num_steps, num_vehicles)


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------


def mlp(input_dim: int, hidden_sizes: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = input_dim
    for size in hidden_sizes:
        layers.append(nn.Linear(last, size))
        layers.append(nn.Tanh())
        last = size
    layers.append(nn.Linear(last, output_dim))
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    """Diagonal Gaussian actor for the leader latent action."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: tuple[int, ...]) -> None:
        super().__init__()
        self.mean_net = mlp(obs_dim, hidden_sizes, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.0))

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.mean_net(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.distribution(obs).log_prob(action).sum(dim=-1)


class VehicleHybridActor(nn.Module):
    """Per-vehicle hybrid actor: 2 Gaussians (tau, eta) + K-way categorical."""

    def __init__(
        self,
        obs_dim: int,
        num_tasks: int,
        hidden_sizes: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.num_tasks = num_tasks
        self.trunk = mlp(obs_dim, hidden_sizes, hidden_sizes[-1])
        self.cont_mean = nn.Linear(hidden_sizes[-1], 2)
        self.cont_log_std = nn.Parameter(torch.full((2,), -1.0))
        self.assign_logits = nn.Linear(hidden_sizes[-1], num_tasks)

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.trunk(obs))

    def sample(
        self,
        obs: torch.Tensor,
        assignment_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cont_mean, cat_logits = self.action_parameters(obs)
        cat_logits = _masked_vehicle_assignment_logits(cat_logits, assignment_mask)
        cont_std = torch.exp(self.cont_log_std).expand_as(cont_mean)
        cont_dist = Normal(cont_mean, cont_std)
        cont_action = cont_dist.rsample()
        cont_logp = cont_dist.log_prob(cont_action).sum(dim=-1)

        cat_dist = Categorical(logits=cat_logits)
        cat_action = cat_dist.sample()
        cat_logp = cat_dist.log_prob(cat_action)

        return cont_action, cat_action, cont_logp + cat_logp

    def log_prob(
        self,
        obs: torch.Tensor,
        cont_action: torch.Tensor,
        cat_action: torch.Tensor,
        assignment_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        log_prob, _ = self.evaluate_actions(
            obs, cont_action, cat_action, assignment_mask=assignment_mask
        )
        return log_prob

    def action_parameters(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return continuous means and assignment logits from one trunk pass."""

        feats = self._features(obs)
        return self.cont_mean(feats), self.assign_logits(feats)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        cont_action: torch.Tensor,
        cat_action: torch.Tensor,
        assignment_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return joint log-probability and means without recomputing features."""

        log_prob, cont_mean, _, _ = self.evaluate_actions_with_entropy(
            obs,
            cont_action,
            cat_action,
            assignment_mask=assignment_mask,
        )
        return log_prob, cont_mean

    def evaluate_actions_with_entropy(
        self,
        obs: torch.Tensor,
        cont_action: torch.Tensor,
        cat_action: torch.Tensor,
        assignment_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return log-probability, means, total entropy, and assignment entropy.

        Exactness label: paper-consistent approximation. MAPPO is a baseline;
        this is the standard entropy bonus for its product Gaussian/categorical
        vehicle policy and changes no deterministic system-model semantics.
        """

        cont_mean, cat_logits = self.action_parameters(obs)
        cat_logits = _masked_vehicle_assignment_logits(cat_logits, assignment_mask)
        cont_std = torch.exp(self.cont_log_std).expand_as(cont_mean)
        cont_dist = Normal(cont_mean, cont_std)
        cont_logp = cont_dist.log_prob(cont_action).sum(dim=-1)
        cat_dist = Categorical(logits=cat_logits)
        cat_logp = cat_dist.log_prob(cat_action)
        assignment_entropy = cat_dist.entropy()
        total_entropy = cont_dist.entropy().sum(dim=-1) + assignment_entropy
        return cont_logp + cat_logp, cont_mean, total_entropy, assignment_entropy


def _masked_vehicle_assignment_logits(
    logits: torch.Tensor,
    assignment_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Mask pruned task choices, with task zero as an inert empty-row placeholder."""

    if assignment_mask is None:
        return logits
    mask = assignment_mask.to(device=logits.device, dtype=torch.bool)
    if mask.shape != logits.shape:
        raise ValueError("vehicle assignment mask shape does not match logits")
    safe = mask.clone()
    empty = ~safe.any(dim=-1)
    safe[empty, 0] = True
    return logits.masked_fill(~safe, torch.finfo(logits.dtype).min)


class Critic(nn.Module):
    """Centralized critic over the global observation."""

    def __init__(self, obs_dim: int, hidden_sizes: tuple[int, ...]) -> None:
        super().__init__()
        self.net = mlp(obs_dim, hidden_sizes, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MAPPOConfig:
    total_timesteps: int = 2_000
    rollout_steps: int = 32
    collection_microbatch_size: int = 32
    n_epochs: int = 4
    minibatch_size: int = 16
    learning_rate: float = 3e-4
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_sizes: tuple[int, ...] = (128, 128)
    device: str = "cpu"
    # Reward shaping lives on :class:`JointLearningShapingConfig` (shared
    # with JDLS); :func:`train_from_config` builds one directly from the
    # yaml ``reward:`` block.
    shaping: JointLearningShapingConfig = field(default_factory=JointLearningShapingConfig)
    # None disables KL early stopping; when set, abort remaining epochs of a
    # PPO update whenever the mean approx_kl exceeds 1.5×target_kl. Same SB3
    # rule used in src/baselines/jdls.py — see that module + jdls.yaml for
    # the mechanism this addresses.
    target_kl: float | None = None


class MAPPOLearnedPolicy:
    """Wrapper exposing a deterministic ``act`` interface for evaluation."""

    def __init__(
        self,
        leader_actor: GaussianActor,
        vehicle_actor: VehicleHybridActor,
        env: RSULeaderEnv,
        device: torch.device,
        parameterization: str = CONSTRAINT_AWARE_JOINT_ACTION_PARAMETERIZATION,
    ) -> None:
        self.leader_actor = leader_actor
        self.vehicle_actor = vehicle_actor
        self.env = env
        self.device = device
        self.parameterization = parameterization

    @torch.no_grad()
    def act(self, observation: dict[str, object]) -> JointAction:
        scenario = self.env.scenario_config
        global_vec = torch.from_numpy(flatten_global_observation(observation)).to(self.device)
        leader_dist = self.leader_actor.distribution(global_vec.unsqueeze(0))
        leader_latent_t = leader_dist.mean
        leader_latent = leader_latent_t.squeeze(0).cpu().numpy()
        prices, bandwidth = decode_leader_latent(
            leader_latent,
            num_tasks=scenario.num_tasks,
            pricing_budget_max=scenario.pricing_budget_max,
            eligible_vehicle_mask=self.env.candidate_vehicle_mask(),
        )
        vehicle_obs = vehicle_observation_matrix(self.env, prices, bandwidth)
        vehicle_obs_t = torch.from_numpy(vehicle_obs).to(self.device)
        cont_t, cat_logits_t = self.vehicle_actor.action_parameters(vehicle_obs_t)
        cat_logits_t = _masked_vehicle_assignment_logits(
            cat_logits_t,
            torch.from_numpy(self.env.candidate_pair_mask()).to(self.device),
        )
        cont_action_t = cont_t
        assignment_t = cat_logits_t.argmax(dim=-1)
        return decode_joint_action(
            leader_latent=leader_latent,
            vehicle_cont=cont_action_t.cpu().numpy(),
            assignment=assignment_t.cpu().numpy(),
            env=self.env,
            parameterization=self.parameterization,
        )


def train_from_config(
    config: dict[str, Any],
    *,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Train MAPPO using the same YAML schema as the TE-PPO entrypoint."""

    seed = int(config.get("seed", 123))
    _set_seed(seed)

    scenario_config = ScenarioConfig(**config.get("scenario", {}))
    algo_cfg = config.get("algorithm", {})
    parameterization = str(
        algo_cfg.get(
            "joint_action_parameterization",
            CONSTRAINT_AWARE_JOINT_ACTION_PARAMETERIZATION,
        )
    )
    if parameterization != CONSTRAINT_AWARE_JOINT_ACTION_PARAMETERIZATION:
        raise ValueError(
            f"unknown joint action parameterization: {parameterization!r}"
        )
    reward_cfg = config.get("reward", {})
    shaping_cfg = JointLearningShapingConfig(**reward_cfg)
    mappo_cfg = MAPPOConfig(
        total_timesteps=int(algo_cfg.get("total_timesteps", 2_000)),
        rollout_steps=int(algo_cfg.get("rollout_steps", 32)),
        collection_microbatch_size=int(
            algo_cfg.get("collection_microbatch_size", 32)
        ),
        n_epochs=int(algo_cfg.get("n_epochs", 4)),
        minibatch_size=int(algo_cfg.get("minibatch_size", 16)),
        learning_rate=float(algo_cfg.get("learning_rate", 3e-4)),
        clip_range=float(algo_cfg.get("clip_range", 0.2)),
        ent_coef=float(algo_cfg.get("ent_coef", 0.01)),
        vf_coef=float(algo_cfg.get("vf_coef", 0.5)),
        max_grad_norm=float(algo_cfg.get("max_grad_norm", 0.5)),
        hidden_sizes=tuple(algo_cfg.get("hidden_sizes", (128, 128))),
        device=str(algo_cfg.get("device", "cpu")),
        shaping=shaping_cfg,
        # `target_kl: null` in yaml (or an absent key) both disable KL early
        # stop — don't feed a None value into float().
        target_kl=(
            float(algo_cfg["target_kl"])
            if algo_cfg.get("target_kl") is not None
            else None
        ),
    )

    artifact_cfg = config.get("artifacts", {})
    run_name = str(artifact_cfg.get("run_name", "mappo_run"))
    if output_root is not None:
        run_dir = Path(output_root) / run_name
    else:
        run_dir = Path(str(artifact_cfg.get("root_dir", "artifacts/mappo"))) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config_snapshot.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    progress = TrainingProgress(total=mappo_cfg.total_timesteps, label=run_name)
    progress.phase("preparing model and environment")

    device = resolve_device(mappo_cfg.device)
    env = RSULeaderEnv(scenario_config=scenario_config)
    obs, _ = env.reset(seed=seed)

    leader_action_dim = scenario_config.num_tasks + 1 + scenario_config.num_vehicles
    global_dim = flatten_global_observation(obs).shape[0]
    vehicle_obs_dim = flatten_vehicle_observation(
        env.vehicle_observation(
            0,
            np.zeros(scenario_config.num_tasks),
            np.zeros(scenario_config.num_vehicles),
        )
    ).shape[0]

    leader_actor = GaussianActor(global_dim, leader_action_dim, mappo_cfg.hidden_sizes).to(device)
    initial_price_fraction = float(algo_cfg.get("initial_price_fraction", 0.5))
    final_mean_layer = leader_actor.mean_net[-1]
    if not isinstance(final_mean_layer, nn.Linear):
        raise RuntimeError("MAPPO leader price head is not a linear output layer")
    with torch.no_grad():
        final_mean_layer.bias[: scenario_config.num_tasks].fill_(
            price_logit_for_fraction(initial_price_fraction)
        )
    vehicle_actor = VehicleHybridActor(
        vehicle_obs_dim, scenario_config.num_tasks, mappo_cfg.hidden_sizes
    ).to(device)
    critic = Critic(global_dim, mappo_cfg.hidden_sizes).to(device)

    trainable_parameters = (
        list(leader_actor.parameters())
        + list(vehicle_actor.parameters())
        + list(critic.parameters())
    )
    optimizer = torch.optim.Adam(trainable_parameters, lr=mappo_cfg.learning_rate)
    env.reset(seed=seed)

    pricing_budget_max = scenario_config.pricing_budget_max
    K = scenario_config.num_tasks
    V = scenario_config.num_vehicles
    # Scenario is fixed for the run, so task metadata for reward shaping is
    # cached once instead of re-read on every step.
    _initial_state = env.require_state()
    task_weights_cache = tuple(float(task.beta) for task in _initial_state.tasks)
    task_deadlines_cache = tuple(float(task.tau_max_s) for task in _initial_state.tasks)
    if mappo_cfg.rollout_steps <= 0:
        raise ValueError("MAPPO rollout_steps must be positive")
    if mappo_cfg.collection_microbatch_size <= 0:
        raise ValueError("MAPPO collection_microbatch_size must be positive")
    # Every episode is a one-step contextual bandit.  A bounded pool is reused
    # across collection microbatches; tensors are concatenated before PPO so
    # the logical rollout/update batch remains unchanged.
    collection_env_count = min(
        mappo_cfg.rollout_steps, mappo_cfg.collection_microbatch_size
    )
    collection_envs = [env] + [
        RSULeaderEnv(scenario_config=scenario_config)
        for _ in range(collection_env_count - 1)
    ]

    # Periodic evaluation setup (for convergence curves)
    eval_cfg = config.get("evaluation", {})
    configured_reward_schema = eval_cfg.get("common_reward_schema")
    if (
        configured_reward_schema is not None
        and configured_reward_schema != COMMON_EVALUATION_REWARD_SCHEMA
    ):
        raise ValueError(
            f"unsupported common reward schema: {configured_reward_schema}"
        )
    eval_seed_list = eval_cfg.get("seed_list") or [seed + 10_000 + i for i in range(3)]
    eval_interval = resolve_evaluation_interval(
        eval_cfg, default=max(1, mappo_cfg.rollout_steps)
    )
    eval_schedule = EvaluationSchedule(
        interval_env_steps=eval_interval,
        force_terminal_evaluation=bool(
            eval_cfg.get("force_terminal_evaluation", True)
        ),
    )
    eval_log_dir = run_dir / "eval"
    eval_log_dir.mkdir(parents=True, exist_ok=True)
    validation_ranges = cpu_validation_ranges(eval_cfg)
    eval_envs = (
        [
            RSULeaderEnv(
                scenario_config=replace(
                    scenario_config,
                    cpu_cycles_per_s_range=band,
                    cpu_cycles_per_s_regimes=None,
                    cpu_regime_probabilities=None,
                )
            )
            for band in validation_ranges
        ]
        if validation_ranges else [env]
    )
    evaluations_timesteps: list[int] = []
    evaluations_results: list[list[float]] = []
    evaluations_raw_rsu_reward: list[list[float]] = []
    evaluations_common_reward: list[list[float]] = []
    evaluations_social_welfare: list[list[float]] = []
    evaluations_wall_time_s: list[float] = []

    # Selected-checkpoint snapshot holds CPU clones of the three networks
    # plus metadata; None until the first improving validation evaluation.
    best: dict[str, Any] | None = None

    history: list[dict[str, float]] = []
    timesteps_done = 0
    collection_wall_time_s = 0.0
    update_wall_time_s = 0.0
    evaluation_wall_time_s = 0.0
    # Same timing origin as the SB3 callback: immediately before the learning
    # loop, after environment and network construction.
    training_started = time.monotonic()

    def evaluate_checkpoint(timestep: int) -> None:
        nonlocal best, evaluation_wall_time_s
        evaluation_started = time.monotonic()
        eval_returns: list[float] = []
        eval_common_rewards: list[float] = []
        eval_social_welfare: list[float] = []
        eval_service_valid: list[float] = []
        for eval_env in eval_envs:
            eval_policy = MAPPOLearnedPolicy(
                leader_actor,
                vehicle_actor,
                eval_env,
                device,
                parameterization=parameterization,
            )
            for es in eval_seed_list:
                obs_e, _ = eval_env.reset(seed=int(es))
                state_e = eval_env.require_state()
                task_weights_e = tuple(float(task.beta) for task in state_e.tasks)
                task_deadlines_e = tuple(
                    float(task.tau_max_s) for task in state_e.tasks
                )
                joint_e = eval_policy.act(obs_e)
                _, reward_e, _, _, info_e = eval_env.step_joint(joint_e)
                eval_returns.append(float(reward_e))
                common_reward_e, _ = compute_common_evaluation_reward(
                    float(reward_e),
                    info_e,
                    config=shaping_cfg,
                    task_weights=task_weights_e,
                    task_deadlines=task_deadlines_e,
                )
                eval_common_rewards.append(common_reward_e)
                eval_social_welfare.append(float(info_e.get("social_welfare", 0.0)))
                eval_service_valid.append(
                    float(
                        episode_metrics_from_info(
                            dict(info_e), episode_reward=float(reward_e)
                        )["service_valid"]
                    )
                )
        evaluations_timesteps.append(int(timestep))
        evaluations_results.append(eval_returns)
        evaluations_raw_rsu_reward.append(eval_returns)
        evaluations_common_reward.append(eval_common_rewards)
        evaluations_social_welfare.append(eval_social_welfare)
        evaluations_wall_time_s.append(time.monotonic() - training_started)
        robust_score: RobustCheckpointScore | None = None
        if validation_ranges:
            robust_score = robust_checkpoint_score(
                eval_social_welfare,
                cell_count=len(validation_ranges),
                seeds_per_cell=len(eval_seed_list),
            )
            mean_sw = robust_score.minimum_cell_social_welfare
        else:
            mean_sw = float(np.mean(eval_social_welfare))
        mean_service_valid = float(np.mean(eval_service_valid))
        progress.set_social_welfare(mean_sw)
        replace_best = (
            should_replace_robust_checkpoint(
                None if best is None else best.get("robust_score"), robust_score
            )
            if robust_score is not None
            else should_replace_checkpoint(best, social_welfare=mean_sw)
        )
        if replace_best:
            best = {
                "leader_actor": {
                    key: value.detach().cpu().clone()
                    for key, value in leader_actor.state_dict().items()
                },
                "vehicle_actor": {
                    key: value.detach().cpu().clone()
                    for key, value in vehicle_actor.state_dict().items()
                },
                "critic": {
                    key: value.detach().cpu().clone()
                    for key, value in critic.state_dict().items()
                },
                "timestep": int(timestep),
                "social_welfare": mean_sw,
                "service_valid": mean_service_valid,
                "robust_score": robust_score,
            }
        env.reset(seed=seed + int(timestep))
        evaluation_wall_time_s += time.monotonic() - evaluation_started

    progress.restart_timing()
    progress.phase("formal training")
    while timesteps_done < mappo_cfg.total_timesteps:
        # ----- collect a rollout -----
        collection_started = time.monotonic()
        rollout_size = min(
            mappo_cfg.rollout_steps,
            mappo_cfg.total_timesteps - timesteps_done,
        )
        global_obs_chunks: list[torch.Tensor] = []
        leader_action_chunks: list[torch.Tensor] = []
        leader_logp_chunks: list[torch.Tensor] = []
        vehicle_obs_chunks: list[torch.Tensor] = []
        vehicle_cont_chunks: list[torch.Tensor] = []
        vehicle_cat_chunks: list[torch.Tensor] = []
        vehicle_logp_chunks: list[torch.Tensor] = []
        vehicle_assignment_mask_chunks: list[torch.Tensor] = []
        reward_buf: list[float] = []
        shaping_diag_buf: list[dict[str, float]] = []
        for batch_start in range(
            0, rollout_size, mappo_cfg.collection_microbatch_size
        ):
            batch_size = min(
                mappo_cfg.collection_microbatch_size,
                rollout_size - batch_start,
            )
            active_envs = collection_envs[:batch_size]
            global_obs_buf: list[np.ndarray] = []
            for batch_idx, rollout_env in enumerate(active_envs):
                rollout_obs, _ = rollout_env.reset(
                    seed=seed + timesteps_done + batch_start + batch_idx
                )
                global_obs_buf.append(flatten_global_observation(rollout_obs))

            global_obs_batch_t = torch.from_numpy(np.stack(global_obs_buf)).to(device)
            with torch.no_grad():
                leader_action_batch_t, leader_logp_batch_t = leader_actor.sample(
                    global_obs_batch_t
                )
            leader_action_np = leader_action_batch_t.cpu().numpy()
            leader_decisions = [
                decode_leader_latent(
                    raw_action,
                    num_tasks=K,
                    pricing_budget_max=pricing_budget_max,
                    eligible_vehicle_mask=rollout_env.candidate_vehicle_mask(),
                )
                for rollout_env, raw_action in zip(
                    active_envs, leader_action_np, strict=True
                )
            ]
            vehicle_obs_np = np.concatenate(
                [
                    vehicle_observation_matrix(rollout_env, prices, bandwidth)
                    for rollout_env, (prices, bandwidth) in zip(
                        active_envs, leader_decisions, strict=True
                    )
                ],
                axis=0,
            )
            vehicle_obs_batch_t = torch.from_numpy(vehicle_obs_np).to(device)
            vehicle_assignment_mask_batch_t = torch.from_numpy(
                np.concatenate(
                    [rollout_env.candidate_pair_mask() for rollout_env in active_envs],
                    axis=0,
                )
            ).to(device)
            with torch.no_grad():
                (
                    vehicle_cont_batch_t,
                    vehicle_cat_batch_t,
                    vehicle_logp_batch_t,
                ) = vehicle_actor.sample(
                    vehicle_obs_batch_t,
                    assignment_mask=vehicle_assignment_mask_batch_t,
                )
            vehicle_cont_np = vehicle_cont_batch_t.cpu().numpy()
            vehicle_cat_np = vehicle_cat_batch_t.cpu().numpy()

            for batch_idx, rollout_env in enumerate(active_envs):
                vehicle_start = batch_idx * V
                vehicle_stop = vehicle_start + V
                joint = decode_joint_action(
                    leader_latent=leader_action_np[batch_idx],
                    vehicle_cont=vehicle_cont_np[vehicle_start:vehicle_stop],
                    assignment=vehicle_cat_np[vehicle_start:vehicle_stop],
                    env=rollout_env,
                    parameterization=parameterization,
                )
                _, raw_reward, _, _, info = rollout_env.step_joint(
                    joint, info_mode="training"
                )
                shaped_reward, shaping_diag = compute_joint_learning_shaping(
                    float(raw_reward),
                    info,
                    cfg=shaping_cfg,
                    task_weights=task_weights_cache,
                    task_deadlines=task_deadlines_cache,
                )
                reward_buf.append(float(shaped_reward))
                shaping_diag_buf.append(shaping_diag)

            global_obs_chunks.append(global_obs_batch_t)
            leader_action_chunks.append(leader_action_batch_t)
            leader_logp_chunks.append(leader_logp_batch_t)
            vehicle_obs_chunks.append(vehicle_obs_batch_t)
            vehicle_cont_chunks.append(vehicle_cont_batch_t)
            vehicle_cat_chunks.append(vehicle_cat_batch_t)
            vehicle_logp_chunks.append(vehicle_logp_batch_t)
            vehicle_assignment_mask_chunks.append(vehicle_assignment_mask_batch_t)

        global_obs_t = torch.cat(global_obs_chunks, dim=0)
        leader_action_t = torch.cat(leader_action_chunks, dim=0)
        leader_logp_t = torch.cat(leader_logp_chunks, dim=0)
        vehicle_obs_t = torch.cat(vehicle_obs_chunks, dim=0)
        vehicle_cont_t = torch.cat(vehicle_cont_chunks, dim=0)
        vehicle_cat_t = torch.cat(vehicle_cat_chunks, dim=0)
        vehicle_old_logp_t = torch.cat(vehicle_logp_chunks, dim=0)
        vehicle_assignment_mask_t = torch.cat(
            vehicle_assignment_mask_chunks, dim=0
        )
        timesteps_done += rollout_size
        collection_wall_time_s += time.monotonic() - collection_started

        # ----- PPO update (one-step bandit: advantage = reward - V) -----
        update_started = time.monotonic()
        leader_action_t = leader_action_t.detach()
        leader_logp_t = leader_logp_t.detach()
        rewards_t = torch.tensor(reward_buf, dtype=torch.float32, device=device)

        vehicle_cont_t = vehicle_cont_t.detach()
        vehicle_cat_t = vehicle_cat_t.detach()
        vehicle_old_logp_t = vehicle_old_logp_t.detach()

        with torch.no_grad():
            values_old = critic(global_obs_t)
        # Normalize advantages for stability.
        advantages = rewards_t - values_old
        if advantages.numel() > 1:
            adv_std = advantages.std()
            if float(adv_std) > 1e-8:
                advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

        # Vehicle-level advantage broadcasts the leader-step advantage to each vehicle row.
        adv_per_step = advantages.detach()
        adv_vehicle = adv_per_step.repeat_interleave(V)

        n_steps = global_obs_t.shape[0]
        indices = np.arange(n_steps)
        vehicle_rows_by_step = _vehicle_rows_by_step(
            num_steps=n_steps,
            num_vehicles=V,
            device=device,
        )

        # SB3-style: trip target_kl on the *current epoch's* mean
        # approx_kl, not a cumulative mean across epochs (which would let
        # early low-KL samples dilute a later overshoot). Within one
        # minibatch we take max(leader_kl, vehicle_kl) so a drifting
        # vehicle head trips the stop even when leader KL dominates the mean.
        kl_triggered = False
        last_epoch_kl_mean = 0.0
        for _ in range(mappo_cfg.n_epochs):
            np.random.shuffle(indices)
            epoch_kl_samples: list[float] = []
            for start in range(0, n_steps, mappo_cfg.minibatch_size):
                batch_idx = indices[start : start + mappo_cfg.minibatch_size]
                batch_idx_t = torch.from_numpy(batch_idx).long().to(device)
                # Leader update
                leader_dist_b = leader_actor.distribution(global_obs_t[batch_idx_t])
                new_logp = leader_dist_b.log_prob(
                    leader_action_t[batch_idx_t]
                ).sum(dim=-1)
                leader_log_ratio = new_logp - leader_logp_t[batch_idx_t]
                ratio = torch.exp(leader_log_ratio)
                adv_b = adv_per_step[batch_idx_t]
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - mappo_cfg.clip_range, 1.0 + mappo_cfg.clip_range) * adv_b
                leader_loss = -torch.min(surr1, surr2).mean()

                values = critic(global_obs_t[batch_idx_t])
                value_loss = ((rewards_t[batch_idx_t] - values) ** 2).mean()

                # Vehicle update — slice the corresponding rows.
                row_indices = vehicle_rows_by_step[batch_idx_t].reshape(-1)
                (
                    veh_new_logp,
                    veh_mean_b,
                    vehicle_entropy_b,
                    assignment_entropy_b,
                ) = vehicle_actor.evaluate_actions_with_entropy(
                    vehicle_obs_t[row_indices],
                    vehicle_cont_t[row_indices],
                    vehicle_cat_t[row_indices],
                    assignment_mask=vehicle_assignment_mask_t[row_indices],
                )
                veh_old_logp = vehicle_old_logp_t[row_indices]
                veh_adv = adv_vehicle[row_indices]
                veh_log_ratio = veh_new_logp - veh_old_logp
                veh_ratio = torch.exp(veh_log_ratio)
                vsurr1 = veh_ratio * veh_adv
                vsurr2 = torch.clamp(
                    veh_ratio, 1.0 - mappo_cfg.clip_range, 1.0 + mappo_cfg.clip_range
                ) * veh_adv
                vehicle_loss = -torch.min(vsurr1, vsurr2).mean()

                leader_entropy = leader_dist_b.entropy().sum(dim=-1).mean()
                vehicle_entropy = vehicle_entropy_b.mean()
                assignment_entropy = assignment_entropy_b.mean()
                entropy_loss = -(leader_entropy + vehicle_entropy)
                loss = (
                    leader_loss
                    + vehicle_loss
                    + mappo_cfg.vf_coef * value_loss
                    + mappo_cfg.ent_coef * entropy_loss
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    mappo_cfg.max_grad_norm,
                )
                optimizer.step()

                with torch.no_grad():
                    leader_kl = float(
                        torch.mean(torch.exp(leader_log_ratio) - 1.0 - leader_log_ratio)
                    )
                    veh_kl = float(
                        torch.mean(torch.exp(veh_log_ratio) - 1.0 - veh_log_ratio)
                    )
                    epoch_kl_samples.append(max(leader_kl, veh_kl))
            if epoch_kl_samples:
                last_epoch_kl_mean = float(np.mean(epoch_kl_samples))
            if (
                mappo_cfg.target_kl is not None
                and last_epoch_kl_mean > 1.5 * mappo_cfg.target_kl
            ):
                kl_triggered = True
                break
        update_wall_time_s += time.monotonic() - update_started

        shaping_means = {
            f"mean_{key}": float(np.mean([d[key] for d in shaping_diag_buf]))
            for key in shaping_diag_buf[0].keys()
        } if shaping_diag_buf else {}
        history.append(
            {
                "timesteps": float(timesteps_done),
                "mean_reward": float(rewards_t.mean().item()),
                "leader_loss": float(leader_loss.item()),
                "vehicle_loss": float(vehicle_loss.item()),
                "value_loss": float(value_loss.item()),
                "entropy_loss": float(entropy_loss.item()),
                "leader_entropy": float(leader_entropy.item()),
                "vehicle_entropy": float(vehicle_entropy.item()),
                "vehicle_assignment_entropy": float(assignment_entropy.item()),
                "approx_kl": last_epoch_kl_mean,
                "kl_triggered": float(kl_triggered),
                **shaping_means,
            }
        )

        # Periodic evaluation on fixed seeds
        if eval_schedule.periodic_due(timesteps_done):
            evaluate_checkpoint(timesteps_done)

        progress.update(timesteps_done)

    if eval_schedule.terminal_due(timesteps_done):
        evaluate_checkpoint(timesteps_done)
    training_loop_wall_time_s = time.monotonic() - training_started
    progress.update(timesteps_done)
    progress.phase("saving best checkpoint and running final evaluation")
    progress.close()

    # Save periodic evaluations
    if evaluations_timesteps:
        np.savez(
            eval_log_dir / "evaluations.npz",
            timesteps=np.array(evaluations_timesteps, dtype=np.int64),
            results=np.array(evaluations_results, dtype=np.float64),
            raw_rsu_reward=np.array(
                evaluations_raw_rsu_reward, dtype=np.float64
            ),
            common_evaluation_reward=np.array(
                evaluations_common_reward, dtype=np.float64
            ),
            reward_schema=np.asarray(COMMON_EVALUATION_REWARD_SCHEMA),
            social_welfare=np.array(evaluations_social_welfare, dtype=np.float64),
            wall_time_s=np.array(evaluations_wall_time_s, dtype=np.float64),
            wall_time_schema=np.asarray("monotonic_training_loop_v2"),
            validation_cell_index=np.tile(
                np.repeat(np.arange(len(eval_envs)), len(eval_seed_list)),
                (len(evaluations_timesteps), 1),
            ),
            cpu_validation_ranges=np.asarray(validation_ranges, dtype=np.float64),
        )

    last_model_path = run_dir / CUSTOM_LAST_MODEL
    torch.save(
        {
            "leader_actor": leader_actor.state_dict(),
            "vehicle_actor": vehicle_actor.state_dict(),
            "critic": critic.state_dict(),
            "scenario": config.get("scenario", {}),
            "implementation_version": str(
                algo_cfg.get(
                    "implementation_version",
                    "mappo_constraint_aware_cold_start_v2",
                )
            ),
            "joint_action_parameterization": parameterization,
        },
        last_model_path,
    )
    if best is not None:
        leader_actor.load_state_dict(
            {k: v.to(device) for k, v in best["leader_actor"].items()}
        )
        vehicle_actor.load_state_dict(
            {k: v.to(device) for k, v in best["vehicle_actor"].items()}
        )
        critic.load_state_dict(
            {k: v.to(device) for k, v in best["critic"].items()}
        )
        print(
            f"[mappo] restored best checkpoint from ts={best['timestep']} "
            f"(social_welfare={best['social_welfare']:+.3f})"
        )
        save_leader_state = best["leader_actor"]
        save_vehicle_state = best["vehicle_actor"]
        save_critic_state = best["critic"]
        best_ts = best["timestep"]
        best_sw = best["social_welfare"]
        best_service_valid = float(best.get("service_valid", 0.0))
        best_robust_score = best.get("robust_score")
    else:  # pragma: no cover - positive budgets force a terminal evaluation
        raise RuntimeError("MAPPO training produced no selected checkpoint")

    selected_model_path = run_dir / CUSTOM_SELECTED_MODEL
    selected_tmp = run_dir / f".{CUSTOM_SELECTED_MODEL}.{os.getpid()}.tmp"
    torch.save(
        {
            "leader_actor": save_leader_state,
            "vehicle_actor": save_vehicle_state,
            "critic": save_critic_state,
            "scenario": config.get("scenario", {}),
            "implementation_version": str(
                algo_cfg.get(
                    "implementation_version",
                    "mappo_constraint_aware_cold_start_v2",
                )
            ),
            "joint_action_parameterization": parameterization,
            "best_timestep": best_ts,
            "best_social_welfare": best_sw,
            "best_service_valid": best_service_valid,
            "selection_metric": "social_welfare",
        },
        selected_tmp,
    )
    os.replace(selected_tmp, selected_model_path)
    fingerprint = config_digest(config)
    if fingerprint is None:
        raise RuntimeError("MAPPO configuration is not digestible")
    atomic_write_json(run_dir / SELECTION_MANIFEST, {
        "schema_version": 2 if validation_ranges else 1,
        "selection_metric": "social_welfare",
        "selection_rule": (
            "maximin_cpu_band_social_welfare"
            if validation_ranges else "mean_social_welfare"
        ),
        "selected_timestep": best_ts,
        "selected_metric_value": best_sw,
        "service_valid": best_service_valid,
        "validation_seeds": [int(value) for value in eval_seed_list],
        "cpu_validation_ranges": [list(value) for value in validation_ranges],
        "validation_grid_digest": (
            validation_grid_digest(validation_ranges) if validation_ranges else None
        ),
        "selected_macro_mean_social_welfare": (
            best_robust_score.macro_mean_social_welfare
            if best_robust_score is not None else best_sw
        ),
        "selected_cell_social_welfare_means": (
            list(best_robust_score.cell_social_welfare_means)
            if best_robust_score is not None else [best_sw]
        ),
        "config_fingerprint": fingerprint,
        "model_file": selected_model_path.name,
        "model_sha256": sha256_file(selected_model_path),
        "normalization_file": None,
        "normalization_sha256": None,
    })

    # Final deterministic evaluation
    final_evaluation_started = time.monotonic()
    eval_returns: list[float] = []
    for eval_env in eval_envs:
        policy = MAPPOLearnedPolicy(
            leader_actor,
            vehicle_actor,
            eval_env,
            device,
            parameterization=parameterization,
        )
        for eval_seed in eval_seed_list:
            obs_eval, _ = eval_env.reset(seed=int(eval_seed))
            joint = policy.act(obs_eval)
            _, reward_eval, _, _, _ = eval_env.step_joint(joint)
            eval_returns.append(float(reward_eval))
    final_evaluation_wall_time_s = time.monotonic() - final_evaluation_started

    accounted_training_time_s = (
        collection_wall_time_s + update_wall_time_s + evaluation_wall_time_s
    )
    performance = {
        "implementation_version": str(
            algo_cfg.get(
                "implementation_version",
                "mappo_constraint_aware_cold_start_v2",
            )
        ),
        "joint_action_parameterization": parameterization,
        "ent_coef": mappo_cfg.ent_coef,
        "rollout_samples_per_update": mappo_cfg.rollout_steps,
        "collection_microbatch_size": mappo_cfg.collection_microbatch_size,
        "timing_scope": "training_loop_excludes_final_evaluation",
        "training_loop_wall_time_s": training_loop_wall_time_s,
        "collection_wall_time_s": collection_wall_time_s,
        "update_wall_time_s": update_wall_time_s,
        "periodic_evaluation_wall_time_s": evaluation_wall_time_s,
        "other_training_wall_time_s": max(
            0.0, training_loop_wall_time_s - accounted_training_time_s
        ),
        "final_evaluation_wall_time_s": final_evaluation_wall_time_s,
        "steps_per_second": (
            float(timesteps_done) / training_loop_wall_time_s
            if training_loop_wall_time_s > 0.0
            else 0.0
        ),
    }

    summary = {
        "run_dir": str(run_dir),
        "selected_model_path": str(selected_model_path),
        "last_model_path": str(last_model_path),
        "seed": seed,
        "total_timesteps": timesteps_done,
        "history": history,
        "eval_seeds": [int(s) for s in eval_seed_list],
        "eval_mean_reward": float(np.mean(eval_returns)) if eval_returns else 0.0,
        "eval_episode_rewards": eval_returns,
        "best_service_valid": best_service_valid,
        "torch_num_threads": torch.get_num_threads(),
        "implementation_version": performance["implementation_version"],
        "joint_action_parameterization": parameterization,
        "performance": performance,
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    try:
        from src.rl.run_metadata import write_run_metadata

        write_run_metadata(run_dir)
    except Exception as exc:  # pragma: no cover
        print(f"[mappo] write_run_metadata failed: {exc}")
    return summary
