"""JDLS: joint-decision learning with one centralized hybrid actor.

Exactness label: paper-consistent approximation.

The actor emits Gaussian leader and follower logits and one categorical task
assignment per vehicle. The environment evaluates the assembled joint action
without the analytical best-response optimizer. Candidate masks and the C2/C3
feasible action map enforce the maintained latency and box constraints.
Task assignment contributes directly to the PPO log probability. Non-analytical
shaping is defined in src/baselines/joint_learning_reward.py.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, replace
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
)
from src.rl.mappo import (
    Critic,
    flatten_global_observation,
    mlp,
)
from src.rl.policy import price_logit_for_fraction
from src.rl.progress import TrainingProgress
from src.rl.train import resolve_device
from src.rl.train import seed_everything as _set_seed
from src.sim.env import JointAction, RSULeaderEnv
from src.sim.scenario import ScenarioConfig

__all__ = [
    "CONSTRAINT_AWARE_JOINT_ACTION_PARAMETERIZATION",
    "JDLSCentralizedActor",
    "JDLSConfig",
    "JDLSLearnedPolicy",
    "train_from_config",
]


def _masked_assignment_logits(
    logits: torch.Tensor,
    assignment_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Mask non-candidate task logits while keeping empty rows well-defined.

    Paper-exact candidate constraint: the categorical support is restricted by
    the maintained indicator ``c[i,k]``.  Task zero is a numerical placeholder
    only for vehicles with no admitted task; the physical decoder gives those
    vehicles zero bandwidth and sensing duration.
    """

    if assignment_mask is None:
        return logits
    if assignment_mask.shape != logits.shape:
        raise ValueError("assignment_mask must match assignment logits")
    mask = assignment_mask.to(device=logits.device, dtype=torch.bool)
    if mask.ndim != 3:
        raise ValueError("assignment_mask must have shape (batch, V, K)")
    effective = mask.clone()
    empty = ~effective.any(dim=-1)
    if torch.any(empty):
        rows = torch.nonzero(empty, as_tuple=False)
        effective[rows[:, 0], rows[:, 1], 0] = True
    return logits.masked_fill(~effective, -torch.inf)


class JDLSCentralizedActor(nn.Module):
    """Single centralized actor: global obs -> Gaussian(cont) + Categorical(assign)^V.

    Continuous output layout (``cont_dim = K + 1 + V + V*2``):

    - ``[0:K]`` — raw price logits (decoded via ``decode_prices``)
    - ``[K]`` — bandwidth utilization logit
    - ``[K+1:K+1+V]`` — per-vehicle bandwidth share logits
    - ``[K+1+V:K+1+V + V]`` — per-vehicle ``tau_logit``
    - ``[K+1+V + V:]`` — per-vehicle ``eta_logit``

    The assignment head outputs ``V*K`` logits reshaped to ``(V, K)`` for
    per-vehicle K-way Categorical sampling.
    """

    def __init__(
        self,
        obs_dim: int,
        num_tasks: int,
        num_vehicles: int,
        hidden_sizes: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.num_tasks = num_tasks
        self.num_vehicles = num_vehicles
        self.cont_dim = num_tasks + 1 + num_vehicles + num_vehicles * 2
        self.trunk = mlp(obs_dim, hidden_sizes, hidden_sizes[-1])
        self.cont_mean = nn.Linear(hidden_sizes[-1], self.cont_dim)
        self.cont_log_std = nn.Parameter(torch.full((self.cont_dim,), -1.0))
        self.assign_logits = nn.Linear(hidden_sizes[-1], num_vehicles * num_tasks)

    def heads(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = torch.tanh(self.trunk(obs))
        cont_mean = self.cont_mean(feats)
        assign_logits = self.assign_logits(feats).reshape(
            -1, self.num_vehicles, self.num_tasks
        )
        return cont_mean, assign_logits

    def sample(
        self,
        obs: torch.Tensor,
        assignment_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cont_mean, assign_logits = self.heads(obs)
        assign_logits = _masked_assignment_logits(assign_logits, assignment_mask)
        cont_std = torch.exp(self.cont_log_std).expand_as(cont_mean)
        cont_dist = Normal(cont_mean, cont_std)
        cont_action = cont_dist.rsample()
        cont_logp = cont_dist.log_prob(cont_action).sum(dim=-1)

        assign_dist = Categorical(logits=assign_logits)
        assign_action = assign_dist.sample()  # shape: (batch, V)
        assign_logp = assign_dist.log_prob(assign_action).sum(dim=-1)

        return cont_action, assign_action, cont_logp + assign_logp

    def log_prob_entropy(
        self,
        obs: torch.Tensor,
        cont_action: torch.Tensor,
        assign_action: torch.Tensor,
        assignment_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cont_mean, assign_logits = self.heads(obs)
        assign_logits = _masked_assignment_logits(assign_logits, assignment_mask)
        cont_std = torch.exp(self.cont_log_std).expand_as(cont_mean)
        cont_dist = Normal(cont_mean, cont_std)
        cont_logp = cont_dist.log_prob(cont_action).sum(dim=-1)
        cont_ent = cont_dist.entropy().sum(dim=-1)
        assign_dist = Categorical(logits=assign_logits)
        assign_logp = assign_dist.log_prob(assign_action).sum(dim=-1)
        assign_ent = assign_dist.entropy().sum(dim=-1)
        return cont_logp + assign_logp, cont_ent + assign_ent


@dataclass(frozen=True)
class JDLSConfig:
    total_timesteps: int = 2_000
    rollout_steps: int = 32
    n_epochs: int = 10
    minibatch_size: int = 64
    learning_rate: float = 3e-4
    clip_range: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    hidden_sizes: tuple[int, ...] = (256, 256)
    device: str = "cpu"
    # None disables KL early stopping; when set, abort remaining epochs of a
    # PPO update whenever the minibatch approx_kl exceeds 1.5×target_kl (SB3's
    # default rule).
    target_kl: float | None = None


def _build_joint_action(
    *,
    cont_np: np.ndarray,
    assign_np: np.ndarray,
    num_tasks: int,
    num_vehicles: int,
    env: RSULeaderEnv,
    parameterization: str = CONSTRAINT_AWARE_JOINT_ACTION_PARAMETERIZATION,
) -> JointAction:
    """Decode (cont, assign) into a :class:`JointAction` without touching argmax.

    Mirrors ``RSULeaderEnv.map_joint_action`` but takes the discrete
    ``assignment`` as already-sampled integers (from the policy's Categorical
    head) rather than as ``argmax`` of a Gaussian slice.

    ``candidate_masked_feasible_v1`` is paper-exact with respect to C2/C3 and
    otherwise a paper-consistent action parameterization.  It implements the
    equations recorded in ``docs/revision_experiment_protocol.md``:
    ``eta = exp(sigmoid(z_eta) * log(eta_max))`` (the maintained lower bound is
    one) and ``tau = tau_min + sigmoid(z_tau) * (deadline/Omega - tau_min)``.
    It never evaluates utility or calls the analytical best-response solver.
    """

    K = num_tasks
    V = num_vehicles
    leader_stop = K + 1 + V
    tau_logits = cont_np[leader_stop : leader_stop + V]
    eta_logits = cont_np[leader_stop + V : leader_stop + 2 * V]
    return decode_joint_action(
        leader_latent=cont_np[:leader_stop],
        vehicle_cont=np.stack((tau_logits, eta_logits), axis=1),
        assignment=assign_np,
        env=env,
        parameterization=parameterization,
    )


class JDLSLearnedPolicy:
    """Deterministic ``act`` interface for evaluation / final rollouts."""

    def __init__(
        self,
        actor: JDLSCentralizedActor,
        env: RSULeaderEnv,
        device: torch.device,
        parameterization: str = CONSTRAINT_AWARE_JOINT_ACTION_PARAMETERIZATION,
    ) -> None:
        self.actor = actor
        self.env = env
        self.device = device
        self.parameterization = parameterization

    @torch.no_grad()
    def act(self, observation: dict[str, object]) -> JointAction:
        scenario = self.env.scenario_config
        global_vec = flatten_global_observation(observation)
        obs_tensor = torch.from_numpy(global_vec).to(self.device).unsqueeze(0)
        cont_mean, assign_logits = self.actor.heads(obs_tensor)
        if self.parameterization == CONSTRAINT_AWARE_JOINT_ACTION_PARAMETERIZATION:
            assignment_mask = (
                torch.from_numpy(self.env.candidate_pair_mask())
                .to(self.device)
                .unsqueeze(0)
            )
            assign_logits = _masked_assignment_logits(assign_logits, assignment_mask)
        cont_np = cont_mean.squeeze(0).cpu().numpy()
        # argmax here is fine: it's eval-only and not part of any gradient path.
        assign_np = assign_logits.squeeze(0).argmax(dim=-1).cpu().numpy()
        return _build_joint_action(
            cont_np=cont_np,
            assign_np=assign_np,
            num_tasks=scenario.num_tasks,
            num_vehicles=scenario.num_vehicles,
            env=self.env,
            parameterization=self.parameterization,
        )


def train_from_config(
    config: dict[str, Any],
    *,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Train JDLS using the MAPPO-style YAML schema (+ a ``reward:`` section)."""

    normalization_cfg = config.get("normalization", {})
    if normalization_cfg.get("norm_obs", False) or normalization_cfg.get(
        "norm_reward", False
    ):
        raise ValueError(
            "JDLS uses a custom PPO loop and does not support observation or "
            "reward normalization; set normalization.norm_obs and "
            "normalization.norm_reward to false"
        )

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
    jdls_cfg = JDLSConfig(
        total_timesteps=int(algo_cfg.get("total_timesteps", 2_000)),
        rollout_steps=int(algo_cfg.get("rollout_steps", 32)),
        n_epochs=int(algo_cfg.get("n_epochs", 10)),
        minibatch_size=int(algo_cfg.get("minibatch_size", 64)),
        learning_rate=float(algo_cfg.get("learning_rate", 3e-4)),
        clip_range=float(algo_cfg.get("clip_range", 0.2)),
        vf_coef=float(algo_cfg.get("vf_coef", 0.5)),
        ent_coef=float(algo_cfg.get("ent_coef", 0.01)),
        max_grad_norm=float(algo_cfg.get("max_grad_norm", 0.5)),
        hidden_sizes=tuple(algo_cfg.get("hidden_sizes", (256, 256))),
        device=str(algo_cfg.get("device", "cpu")),
        # `target_kl: null` in yaml (or an absent key) both disable KL early
        # stop — don't feed a None value into float().
        target_kl=(
            float(algo_cfg["target_kl"])
            if algo_cfg.get("target_kl") is not None
            else None
        ),
    )
    shaping_cfg = JointLearningShapingConfig(**reward_cfg)

    artifact_cfg = config.get("artifacts", {})
    run_name = str(artifact_cfg.get("run_name", "jdls_run"))
    if output_root is not None:
        run_dir = Path(output_root) / run_name
    else:
        run_dir = Path(str(artifact_cfg.get("root_dir", "artifacts/jdls"))) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config_snapshot.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    progress = TrainingProgress(total=jdls_cfg.total_timesteps, label=run_name)
    progress.phase("preparing model and environment")

    device = resolve_device(jdls_cfg.device)

    env = RSULeaderEnv(scenario_config=scenario_config)
    obs, _ = env.reset(seed=seed)

    global_dim = flatten_global_observation(obs).shape[0]
    K = scenario_config.num_tasks
    V = scenario_config.num_vehicles

    actor = JDLSCentralizedActor(
        obs_dim=global_dim,
        num_tasks=K,
        num_vehicles=V,
        hidden_sizes=jdls_cfg.hidden_sizes,
    ).to(device)
    initial_price_fraction = float(algo_cfg.get("initial_price_fraction", 0.5))
    with torch.no_grad():
        actor.cont_mean.bias[:K].fill_(
            price_logit_for_fraction(initial_price_fraction)
        )
    critic = Critic(global_dim, jdls_cfg.hidden_sizes).to(device)

    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=jdls_cfg.learning_rate,
    )
    obs, _ = env.reset(seed=seed)

    _initial_state = env.require_state()
    task_weights_cache = tuple(float(task.beta) for task in _initial_state.tasks)
    task_deadlines_cache = tuple(float(task.tau_max_s) for task in _initial_state.tasks)

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
        eval_cfg, default=max(1, jdls_cfg.rollout_steps)
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

    # Best-checkpoint tracking on the shared social-welfare contract. The
    # snapshot holds CPU clones until the selected artifact is published.
    best: dict[str, Any] | None = None

    def evaluate_checkpoint(timestep: int) -> None:
        nonlocal best
        eval_returns: list[float] = []
        eval_common_rewards: list[float] = []
        eval_social_welfare: list[float] = []
        eval_service_valid: list[float] = []
        for eval_env in eval_envs:
            eval_policy = JDLSLearnedPolicy(
                actor, eval_env, device, parameterization=parameterization
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
                "actor": {
                    key: value.detach().cpu().clone()
                    for key, value in actor.state_dict().items()
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

    history: list[dict[str, float]] = []
    timesteps_done = 0
    # Same timing origin as the SB3 callback: immediately before the learning
    # loop, after environment and network construction.
    training_started = time.monotonic()
    progress.restart_timing()
    progress.phase("formal training")
    while timesteps_done < jdls_cfg.total_timesteps:
        obs_buf: list[np.ndarray] = []
        cont_buf: list[np.ndarray] = []
        assign_buf: list[np.ndarray] = []
        assignment_mask_buf: list[np.ndarray] = []
        logp_buf: list[float] = []
        reward_buf: list[float] = []
        shaping_diag_buf: list[dict[str, float]] = []

        for _ in range(jdls_cfg.rollout_steps):
            global_vec = flatten_global_observation(obs)
            obs_tensor = torch.from_numpy(global_vec).unsqueeze(0).to(device)
            pair_mask = (
                env.candidate_pair_mask()
                if parameterization == CONSTRAINT_AWARE_JOINT_ACTION_PARAMETERIZATION
                else None
            )
            pair_mask_tensor = (
                torch.from_numpy(pair_mask).to(device).unsqueeze(0)
                if pair_mask is not None
                else None
            )
            with torch.no_grad():
                cont_action, assign_action, logp = actor.sample(
                    obs_tensor, assignment_mask=pair_mask_tensor
                )
            cont_np = cont_action.squeeze(0).cpu().numpy()
            assign_np = assign_action.squeeze(0).cpu().numpy()
            joint = _build_joint_action(
                cont_np=cont_np,
                assign_np=assign_np,
                num_tasks=K,
                num_vehicles=V,
                env=env,
                parameterization=parameterization,
            )
            _, raw_reward, _, _, info = env.step_joint(
                joint, info_mode="training"
            )
            shaped, shaping_diag = compute_joint_learning_shaping(
                float(raw_reward),
                info,
                cfg=shaping_cfg,
                task_weights=task_weights_cache,
                task_deadlines=task_deadlines_cache,
            )
            if pair_mask is not None:
                valid_assignment = pair_mask[np.arange(V), assign_np]
                valid_indices = np.flatnonzero(valid_assignment)
                assigned_evaluations = {
                    int(v_idx): next(
                        (
                            evaluation
                            for evaluation in info["follower_responses"][
                                int(v_idx)
                            ].task_evaluations
                            if evaluation.task_id == int(assign_np[int(v_idx)])
                        ),
                        None,
                    )
                    for v_idx in valid_indices
                }
                shaping_diag.update(
                    {
                        "candidate_valid_assignment_count": float(
                            np.sum(valid_assignment)
                        ),
                        "projection_infeasible_assignment_count": float(
                            sum(
                                assigned_evaluations[int(v_idx)] is None
                                or not assigned_evaluations[int(v_idx)].feasible
                                for v_idx in valid_indices
                            )
                        ),
                        "ir_accepted_assignment_count": float(
                            sum(
                                response.selected_task_id is not None
                                for response in info["follower_responses"]
                            )
                        ),
                        "compression_ratio": float(
                            np.mean(np.asarray(joint.eta)[valid_indices])
                        )
                        if valid_indices.size
                        else 0.0,
                        "deadline_fraction": float(
                            np.mean(
                                [
                                    joint.tau_s[int(v_idx)]
                                    / task_deadlines_cache[int(assign_np[int(v_idx)])]
                                    for v_idx in valid_indices
                                ]
                            )
                        )
                        if valid_indices.size
                        else 0.0,
                        "bandwidth_total": float(np.sum(joint.bandwidth)),
                    }
                )
            obs_buf.append(global_vec)
            cont_buf.append(cont_np)
            assign_buf.append(assign_np)
            if pair_mask is not None:
                assignment_mask_buf.append(pair_mask)
            logp_buf.append(float(logp.item()))
            reward_buf.append(float(shaped))
            shaping_diag_buf.append(shaping_diag)

            obs, _ = env.reset(seed=seed + timesteps_done + 1)
            timesteps_done += 1
            if timesteps_done >= jdls_cfg.total_timesteps:
                break

        if not reward_buf:
            break

        obs_t = torch.from_numpy(np.stack(obs_buf)).to(device)
        cont_t = torch.from_numpy(np.stack(cont_buf)).to(device)
        assign_t = torch.from_numpy(np.stack(assign_buf)).long().to(device)
        assignment_mask_t = (
            torch.from_numpy(np.stack(assignment_mask_buf)).to(device)
            if assignment_mask_buf
            else None
        )
        old_logp_t = torch.tensor(logp_buf, dtype=torch.float32, device=device)
        rewards_t = torch.tensor(reward_buf, dtype=torch.float32, device=device)

        with torch.no_grad():
            values_old = critic(obs_t)
        advantages = rewards_t - values_old
        if advantages.numel() > 1:
            adv_std = advantages.std()
            if float(adv_std) > 1e-8:
                advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)
        advantages = advantages.detach()

        n_steps = obs_t.shape[0]
        indices = np.arange(n_steps)
        # SB3 PPO compares the *current epoch's* mean approx_kl against
        # 1.5×target — accumulating across epochs would let early low-KL
        # samples dilute a later overshoot and bypass the safeguard.
        kl_triggered = False
        last_epoch_kl_mean = 0.0
        for _ in range(jdls_cfg.n_epochs):
            np.random.shuffle(indices)
            epoch_kl_samples: list[float] = []
            for start in range(0, n_steps, jdls_cfg.minibatch_size):
                batch_idx = indices[start : start + jdls_cfg.minibatch_size]
                batch_idx_t = torch.from_numpy(batch_idx).long().to(device)
                new_logp, entropy = actor.log_prob_entropy(
                    obs_t[batch_idx_t],
                    cont_t[batch_idx_t],
                    assign_t[batch_idx_t],
                    assignment_mask=(
                        assignment_mask_t[batch_idx_t]
                        if assignment_mask_t is not None
                        else None
                    ),
                )
                log_ratio = new_logp - old_logp_t[batch_idx_t]
                ratio = torch.exp(log_ratio)
                adv_b = advantages[batch_idx_t]
                surr1 = ratio * adv_b
                surr2 = torch.clamp(
                    ratio, 1.0 - jdls_cfg.clip_range, 1.0 + jdls_cfg.clip_range
                ) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()
                ent_loss = -entropy.mean()

                values = critic(obs_t[batch_idx_t])
                value_loss = ((rewards_t[batch_idx_t] - values) ** 2).mean()

                loss = (
                    policy_loss
                    + jdls_cfg.ent_coef * ent_loss
                    + jdls_cfg.vf_coef * value_loss
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()),
                    jdls_cfg.max_grad_norm,
                )
                optimizer.step()

                # Schulman approx KL: E[exp(log_ratio) - 1 - log_ratio].
                with torch.no_grad():
                    epoch_kl_samples.append(
                        float(torch.mean(torch.exp(log_ratio) - 1.0 - log_ratio))
                    )
            if epoch_kl_samples:
                last_epoch_kl_mean = float(np.mean(epoch_kl_samples))
            if (
                jdls_cfg.target_kl is not None
                and last_epoch_kl_mean > 1.5 * jdls_cfg.target_kl
            ):
                kl_triggered = True
                break

        shaping_means = {
            f"mean_{key}": float(np.mean([d[key] for d in shaping_diag_buf]))
            for key in shaping_diag_buf[0].keys()
        } if shaping_diag_buf else {}
        prior_kl_triggers = sum(item["kl_triggered"] for item in history)
        history.append(
            {
                "timesteps": float(timesteps_done),
                "mean_reward": float(rewards_t.mean().item()),
                "policy_loss": float(policy_loss.item()),
                "value_loss": float(value_loss.item()),
                "entropy": float(entropy.mean().item()),
                "approx_kl": last_epoch_kl_mean,
                "kl_triggered": float(kl_triggered),
                "kl_trigger_rate": float(
                    (prior_kl_triggers + float(kl_triggered)) / (len(history) + 1)
                ),
                **shaping_means,
            }
        )

        if eval_schedule.periodic_due(timesteps_done):
            evaluate_checkpoint(timesteps_done)
            obs, _ = env.reset(seed=seed + timesteps_done)

        progress.update(timesteps_done)

    if eval_schedule.terminal_due(timesteps_done):
        evaluate_checkpoint(timesteps_done)
    progress.update(timesteps_done)
    progress.phase("saving best checkpoint and running final evaluation")
    progress.close()

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
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "scenario": config.get("scenario", {}),
            "implementation_version": str(
                algo_cfg.get(
                    "implementation_version",
                    "jdls_constraint_aware_cold_start_v1",
                )
            ),
            "joint_action_parameterization": parameterization,
        },
        last_model_path,
    )
    if best is not None:
        actor.load_state_dict({k: v.to(device) for k, v in best["actor"].items()})
        critic.load_state_dict({k: v.to(device) for k, v in best["critic"].items()})
        print(
            f"[jdls] restored best checkpoint from ts={best['timestep']} "
            f"(social_welfare={best['social_welfare']:+.3f})"
        )
        save_actor_state = best["actor"]
        save_critic_state = best["critic"]
        best_ts = best["timestep"]
        best_sw = best["social_welfare"]
        best_service_valid = float(best.get("service_valid", 0.0))
        best_robust_score = best.get("robust_score")
    else:  # pragma: no cover - positive budgets force a terminal evaluation
        raise RuntimeError("JDLS training produced no selected checkpoint")

    selected_model_path = run_dir / CUSTOM_SELECTED_MODEL
    selected_tmp = run_dir / f".{CUSTOM_SELECTED_MODEL}.{os.getpid()}.tmp"
    torch.save(
        {
            "actor": save_actor_state,
            "critic": save_critic_state,
            "scenario": config.get("scenario", {}),
            "implementation_version": str(
                algo_cfg.get(
                    "implementation_version",
                    "jdls_constraint_aware_cold_start_v1",
                )
            ),
            "best_timestep": best_ts,
            "best_social_welfare": best_sw,
            "best_service_valid": best_service_valid,
            "selection_metric": "social_welfare",
            "joint_action_parameterization": parameterization,
        },
        selected_tmp,
    )
    os.replace(selected_tmp, selected_model_path)
    fingerprint = config_digest(config)
    if fingerprint is None:
        raise RuntimeError("JDLS configuration is not digestible")
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

    eval_returns: list[float] = []
    for eval_env in eval_envs:
        policy = JDLSLearnedPolicy(
            actor, eval_env, device, parameterization=parameterization
        )
        for eval_seed in eval_seed_list:
            obs_eval, _ = eval_env.reset(seed=int(eval_seed))
            joint = policy.act(obs_eval)
            _, reward_eval, _, _, _ = eval_env.step_joint(joint)
            eval_returns.append(float(reward_eval))

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
        "implementation_version": str(
            algo_cfg.get(
                "implementation_version",
                "jdls_constraint_aware_cold_start_v1",
            )
        ),
        "joint_action_parameterization": parameterization,
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    try:
        from src.rl.run_metadata import write_run_metadata

        write_run_metadata(run_dir)
    except Exception as exc:  # pragma: no cover
        print(f"[jdls] write_run_metadata failed: {exc}")
    return summary
