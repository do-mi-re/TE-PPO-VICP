"""Common penalty-adjusted reward used for cross-algorithm evaluation.

Exactness label: paper-consistent approximation.  The implementation reuses
the maintained TE-PPO penalty definition without adding algorithm-specific
joint-learning shaping terms.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence

import numpy as np

COMMON_EVALUATION_REWARD_SCHEMA = "common_evaluation_reward_v1"


class CommonRewardConfig(Protocol):
    """Structural type for the four common penalty parameters."""

    omega_fail: float
    omega_lat: float
    omega_part: float
    min_participants: int


def compute_common_evaluation_reward(
    raw_rsu_reward: float,
    info: dict[str, Any],
    *,
    config: CommonRewardConfig,
    task_weights: Sequence[float],
    task_deadlines: Sequence[float],
) -> tuple[float, dict[str, float]]:
    """Return ``U_RSU - P_fail - P_lat - P_part`` and diagnostics.

    ``info`` is the full result of either ``RSULeaderEnv.step_joint`` or the
    leader-only environment step before its reward wrapper removes nested
    follower data.  The function is deterministic and does not inspect or
    modify policy state.
    """

    utility_components = dict(info["utility_components"])
    effective_task_qualities = np.asarray(
        utility_components["effective_task_qualities"], dtype=np.float64
    )
    participant_counts = np.asarray(
        utility_components["participant_counts"], dtype=np.float64
    )
    if len(task_weights) != len(effective_task_qualities):
        raise ValueError("task-weight and effective-quality counts differ")
    if len(task_weights) != len(participant_counts):
        raise ValueError("task-weight and participant counts differ")
    if len(task_weights) != len(task_deadlines):
        raise ValueError("task-weight and deadline counts differ")

    penalty_fail = sum(
        float(config.omega_fail) * float(beta_k)
        for beta_k, quality_k in zip(
            task_weights, effective_task_qualities, strict=True
        )
        if float(quality_k) <= 0.0
    )

    participant_deficit_total = 0.0
    penalty_part = 0.0
    for beta_k, participants_k in zip(
        task_weights, participant_counts, strict=True
    ):
        deficit = max(
            float(config.min_participants) - float(participants_k), 0.0
        )
        participant_deficit_total += deficit
        penalty_part += float(config.omega_part) * float(beta_k) * deficit

    penalty_lat = 0.0
    latency_overshoot_total = 0.0
    for response in tuple(info["follower_responses"]):
        if response.selected_task_id is None:
            continue
        selected_eval = next(
            evaluation
            for evaluation in response.task_evaluations
            if evaluation.task_id == response.selected_task_id
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
            float(config.omega_lat)
            * float(task_weights[selected_eval.task_id])
            * overshoot
        )

    diagnostics = {
        "penalty_fail": float(penalty_fail),
        "penalty_lat": float(penalty_lat),
        "penalty_part": float(penalty_part),
        "task_failure_count": float(np.sum(effective_task_qualities <= 0.0)),
        "participant_deficit_total": float(participant_deficit_total),
        "latency_overshoot_total": float(latency_overshoot_total),
    }
    common_reward = float(raw_rsu_reward) - sum(
        diagnostics[key]
        for key in ("penalty_fail", "penalty_lat", "penalty_part")
    )
    return common_reward, diagnostics


__all__ = [
    "COMMON_EVALUATION_REWARD_SCHEMA",
    "CommonRewardConfig",
    "compute_common_evaluation_reward",
]
