"""Reward shaping for joint-learning baselines (JDLS, MAPPO).

Exactness label: paper-consistent approximation; auxiliary shaping terms are
training aids and are not reported as the common evaluation objective.

Joint-learning baselines jointly learn the leader's pricing/bandwidth and
every vehicle's sensing duration, compression ratio, and task assignment.
The environment accepts only feasible assignments with positive conditional
follower utility. Pre-filter signals evaluate assigned actions even when
the individual-rationality filter rejects them.

The shaped training reward starts from raw RSU utility, adds the auxiliary
terms below, and subtracts the common failure, latency, and participation
penalties. Raw RSU utility is the leader objective; social welfare is a
separate evaluation metric used for checkpoint selection and headline
convergence. Reward-convergence reports use the common penalty-adjusted
evaluation reward without these auxiliary terms. Algorithm YAMLs explicitly
specify their shaping coefficients.

Term taxonomy:

* ``omega_fail`` / ``omega_lat`` / ``omega_part`` — penalties mirroring
  :class:`~src.rl.ppo.TEPPORewardWrapper`, so task failure / latency
  overshoot / under-participation are not free. Also used by TE-PPO.
* ``omega_vehicle`` — small pull toward positive conditional follower
  utility, including finite utilities from infeasible assignments.
* ``omega_feasible`` — hard +1 per vehicle meeting its latency deadline.
* ``omega_pre_lat`` — pre-filter latency-overshoot penalty.
* ``omega_pre_quality`` — pre-filter ``ln(1+quality)`` reward for finite,
  defined quality. Default 0.
* ``omega_tau`` — reward on ``tau_s`` actions normalized by the maximum
  task deadline and clipped to [0,1] per vehicle. Default 0.
* ``omega_bandwidth_use`` — linear reward on the leader's bandwidth
  share action, clipped to [0,1] per vehicle. Default 0.
* ``penalize_missing_latency_as_cap`` — when enabled, an assigned task whose
  evaluator returns the no-rate/undefined-latency sentinel receives the
  configured capped pre-filter overshoot instead of being treated as zero.
  This is enabled by constraint-aware learned-response configurations,
  including the JDLS/MAPPO baselines and the no-BR ablation.

Shaping coefficients determine the relative scale of these training signals.
Use the separately recorded ``social_welfare`` and service metrics to assess
policy performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

__all__ = [
    "JointLearningShapingConfig",
    "compute_joint_learning_shaping",
]


@dataclass(frozen=True)
class JointLearningShapingConfig:
    # omega_vehicle is small because feasible per-vehicle utility can reach
    # ~O(100) in this scenario (payment×quality is the dominant term); summed
    # over 20 vehicles it's comparable to the penalty scale (~30) only with a
    # coefficient around 1e-3.
    omega_vehicle: float = 0.001
    # omega_feasible gives the policy a low-variance "did any vehicle finish
    # in time?" signal. +1 per feasible assignment is comparable in magnitude
    # to omega_part's per-task deficit penalty, so early training is driven by
    # feasibility rather than by utility noise.
    omega_feasible: float = 1.0
    # Pre-filter per-vehicle latency overshoot is computed for assigned tasks,
    # including those rejected by the environment's participation filter.
    omega_pre_lat: float = 0.01
    # Per-vehicle overshoot ratio is clipped at this cap so a badly-chosen
    # assignment (e.g., tiny deadline task with heavy compression) can't blow
    # up the signal and drown out the other shaping terms.
    pre_lat_overshoot_cap: float = 10.0
    penalize_missing_latency_as_cap: bool = False
    # Optional quality and action-direct terms default to zero.
    omega_pre_quality: float = 0.0
    omega_tau: float = 0.0
    omega_bandwidth_use: float = 0.0
    omega_fail: float = 2.0
    omega_lat: float = 3.0
    omega_part: float = 3.0
    min_participants: int = 1


def compute_joint_learning_shaping(
    raw_reward: float,
    info: dict[str, Any],
    *,
    cfg: JointLearningShapingConfig,
    task_weights: Sequence[float],
    task_deadlines: Sequence[float],
) -> tuple[float, dict[str, float]]:
    """Return ``(shaped_reward, diagnostics)``.

    Pure function: no env interaction. ``info`` **must** be the dict returned
    by :meth:`src.sim.env.RSULeaderEnv.step_joint` — it reads
    ``info["joint_action"]["assignment"]``, ``info["follower_responses"]``,
    and ``info["utility_components"]``, which are only populated on the
    ``step_joint`` path (not the default ``step`` path used by the
    leader-only baselines).
    """

    # Shaping signals are decomposed into three pieces:
    #
    # 1) ``num_feasible_assignments`` — how many vehicles picked an assignment
    #    whose (tau_s, eta, bandwidth) actually meets the latency deadline.
    #    Binary per-vehicle; rewarded by ``omega_feasible``.
    # 2) ``raw_assigned_utility_sum`` — sum of pre-filter utility over **all**
    #    assigned (vehicle, task) pairs, feasible or not. Covering infeasible
    #    assignments supplies a utility signal independently of participation.
    # 3) ``pre_filter_quality_log`` — Σ_v ln(1 + quality_v,assigned). Reads
    #    ``task_evaluations[k].quality`` directly when finite and defined.
    #
    # All three signals use fields evaluated before the participation filter.
    joint_assignment = np.asarray(info["joint_action"]["assignment"], dtype=np.int64)
    raw_assigned_utility_sum = 0.0
    num_feasible_assignments = 0
    pre_filter_latency_overshoot = 0.0
    pre_filter_quality_log = 0.0
    penalty_lat = 0.0
    latency_overshoot_total = 0.0
    # Single pass over vehicles: accumulate the pre-filter shaping signals
    # off the *assigned* task_evaluation AND the post-filter penalty_lat off
    # the *selected* task_evaluation. The two evaluations can differ because
    # the rationality filter in env.py may reject the assigned task and
    # leave ``selected_task_id`` as None or a different task.
    for v_idx, response in enumerate(info["follower_responses"]):
        evaluations = response.task_evaluations
        assigned_task_id = int(joint_assignment[v_idx])
        match = next(
            (e for e in evaluations if e.task_id == assigned_task_id), None
        )
        if match is not None:
            if match.total_latency_s is not None:
                deadline = float(task_deadlines[assigned_task_id])
                if deadline > 0.0:
                    overshoot = max(
                        (float(match.total_latency_s) - deadline) / deadline, 0.0
                    )
                    pre_filter_latency_overshoot += min(
                        overshoot, cfg.pre_lat_overshoot_cap
                    )
            elif cfg.penalize_missing_latency_as_cap:
                pre_filter_latency_overshoot += cfg.pre_lat_overshoot_cap
            if match.quality is not None and np.isfinite(match.quality):
                pre_filter_quality_log += float(
                    np.log1p(max(float(match.quality), 0.0))
                )
            utility = float(match.utility)
            if np.isfinite(utility):
                raw_assigned_utility_sum += utility
                if match.feasible:
                    num_feasible_assignments += 1

        if response.selected_task_id is None:
            continue
        selected_eval = next(
            e for e in evaluations if e.task_id == response.selected_task_id
        )
        if selected_eval.total_latency_s is None:
            continue
        deadline = float(task_deadlines[selected_eval.task_id])
        if deadline <= 0.0:
            continue
        overshoot = max(
            (float(selected_eval.total_latency_s) - deadline) / deadline, 0.0
        )
        latency_overshoot_total += overshoot
        penalty_lat += (
            cfg.omega_lat * float(task_weights[selected_eval.task_id]) * overshoot
        )

    effective_task_qualities = np.asarray(
        info["utility_components"]["effective_task_qualities"], dtype=np.float64
    )
    participant_counts = np.asarray(
        info["utility_components"]["participant_counts"], dtype=np.float64
    )

    penalty_fail = 0.0
    for beta_k, quality_k in zip(task_weights, effective_task_qualities, strict=True):
        if float(quality_k) <= 0.0:
            penalty_fail += cfg.omega_fail * float(beta_k)

    penalty_part = 0.0
    participant_deficit_total = 0.0
    for beta_k, participants_k in zip(task_weights, participant_counts, strict=True):
        deficit = max(float(cfg.min_participants) - float(participants_k), 0.0)
        participant_deficit_total += deficit
        penalty_part += cfg.omega_part * float(beta_k) * deficit

    # Normalize action-direct sensing duration by the maximum task deadline.
    # Each vehicle contributes in [0,1], independently of response feasibility.
    tau_nonzero_reward = 0.0
    if cfg.omega_tau != 0.0 and task_deadlines:
        max_deadline = max(float(d) for d in task_deadlines)
        if max_deadline > 0.0:
            tau_arr = np.asarray(
                info["joint_action"]["tau_s"], dtype=np.float64
            )
            tau_nonzero_reward = float(
                np.sum(np.clip(tau_arr / max_deadline, 0.0, 1.0))
            )

    # Sum the clipped action-direct bandwidth shares. Feasible decoded shares
    # sum to at most one.
    bandwidth_use_reward = 0.0
    if cfg.omega_bandwidth_use != 0.0:
        bw_data = info.get("joint_action", {}).get("bandwidth")
        if bw_data is not None:
            bw_arr = np.asarray(bw_data, dtype=np.float64)
            bandwidth_use_reward = float(np.sum(np.clip(bw_arr, 0.0, 1.0)))

    shaped = (
        float(raw_reward)
        + cfg.omega_vehicle * raw_assigned_utility_sum
        + cfg.omega_feasible * float(num_feasible_assignments)
        + cfg.omega_pre_quality * pre_filter_quality_log
        + cfg.omega_tau * tau_nonzero_reward
        + cfg.omega_bandwidth_use * bandwidth_use_reward
        - cfg.omega_pre_lat * pre_filter_latency_overshoot
        - penalty_fail
        - penalty_lat
        - penalty_part
    )

    diagnostics = {
        "raw_rsu_reward": float(raw_reward),
        "shaped_reward": shaped,
        "raw_assigned_utility_sum": raw_assigned_utility_sum,
        "num_feasible_assignments": float(num_feasible_assignments),
        "pre_filter_latency_overshoot": pre_filter_latency_overshoot,
        "pre_filter_quality_log": pre_filter_quality_log,
        "tau_nonzero_reward": tau_nonzero_reward,
        "bandwidth_use_reward": bandwidth_use_reward,
        "total_vehicle_utility": float(info.get("total_vehicle_utility", 0.0)),
        "penalty_fail": penalty_fail,
        "penalty_lat": penalty_lat,
        "penalty_part": penalty_part,
        "task_failure_count": float(np.sum(effective_task_qualities <= 0.0)),
        "participant_deficit_total": participant_deficit_total,
        "latency_overshoot_total": latency_overshoot_total,
    }
    return shaped, diagnostics
