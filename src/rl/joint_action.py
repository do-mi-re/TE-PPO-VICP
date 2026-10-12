"""Physical action decoding shared by JDLS and MAPPO.

Exactness labels
----------------
* ``candidate_masked_feasible_v1`` is paper-exact for the maintained C2/C3
  latency and box constraints.
* Mapping unconstrained policy logits into that feasible interval is a
  paper-consistent approximation.  It never evaluates vehicle utility, searches
  a multiplier, or calls the analytical best-response solver.
"""

from __future__ import annotations

import numpy as np

from src.game.follower_solver import (
    feasible_sensing_duration_upper_bound,
    minimum_service_sensing_duration,
    uplink_rate_for_vehicle,
)
from src.rl.policy import decode_bandwidth, decode_prices, stable_sigmoid
from src.sim.env import JointAction, RSULeaderEnv

CONSTRAINT_AWARE_JOINT_ACTION_PARAMETERIZATION = "candidate_masked_feasible_v1"


def decode_leader_latent(
    raw: np.ndarray,
    *,
    num_tasks: int,
    pricing_budget_max: float,
    eligible_vehicle_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode price and bandwidth logits into a feasible leader action."""

    prices = decode_prices(raw[:num_tasks], pricing_budget_max=pricing_budget_max)
    bandwidth = decode_bandwidth(
        float(raw[num_tasks]),
        raw[num_tasks + 1 :],
        eligible_vehicle_mask=eligible_vehicle_mask,
    )
    return np.asarray(prices, dtype=np.float64), np.asarray(bandwidth, dtype=np.float64)


def decode_joint_action(
    *,
    leader_latent: np.ndarray,
    vehicle_cont: np.ndarray,
    assignment: np.ndarray,
    env: RSULeaderEnv,
    parameterization: str,
) -> JointAction:
    """Decode one sampled joint action using the declared physical mapping."""

    scenario = env.scenario_config
    state = env.require_state()
    task_count = scenario.num_tasks
    vehicle_count = scenario.num_vehicles
    vehicle_actions = np.asarray(vehicle_cont, dtype=np.float64)
    assignment_array = np.asarray(assignment, dtype=np.int64)
    if vehicle_actions.shape != (vehicle_count, 2):
        raise ValueError(
            f"vehicle_cont must have shape {(vehicle_count, 2)}, "
            f"got {vehicle_actions.shape}"
        )
    if assignment_array.shape != (vehicle_count,):
        raise ValueError(
            f"assignment must have shape {(vehicle_count,)}, "
            f"got {assignment_array.shape}"
        )

    prices, bandwidth = decode_leader_latent(
        np.asarray(leader_latent, dtype=np.float64),
        num_tasks=task_count,
        pricing_budget_max=scenario.pricing_budget_max,
        eligible_vehicle_mask=env.candidate_vehicle_mask(),
    )
    eta_max = float(env.solver_params.compression_ratio_max)
    if parameterization == CONSTRAINT_AWARE_JOINT_ACTION_PARAMETERIZATION:
        if eta_max < 1.0:
            raise ValueError("eta_max must be at least one")
        eta_fraction = np.asarray(stable_sigmoid(vehicle_actions[:, 1]), dtype=np.float64)
        eta = np.exp(eta_fraction * np.log(eta_max))
        tau = np.zeros(vehicle_count, dtype=np.float64)
        pair_mask = env.candidate_pair_mask()
        profiles = env.vehicle_channel_profiles()
        for vehicle_idx in range(vehicle_count):
            task_id = int(assignment_array[vehicle_idx])
            admitted = (
                0 <= task_id < task_count
                and bool(pair_mask[vehicle_idx, task_id])
                and float(bandwidth[vehicle_idx]) > 0.0
            )
            if not admitted:
                continue
            vehicle, channel = profiles[vehicle_idx]
            try:
                rate_bps = uplink_rate_for_vehicle(
                    vehicle=vehicle,
                    channel=channel,
                    bandwidth_fraction=float(bandwidth[vehicle_idx]),
                    params=env.solver_params,
                )
                upper = feasible_sensing_duration_upper_bound(
                    deadline_s=float(state.tasks[task_id].tau_max_s),
                    vehicle=vehicle,
                    params=env.solver_params,
                    uplink_rate_bps=rate_bps,
                    compression_ratio=float(eta[vehicle_idx]),
                )
                lower = minimum_service_sensing_duration(
                    vehicle=vehicle,
                    params=env.solver_params,
                )
            except ValueError:
                continue
            if upper < lower:
                # Preserve a finite physical action while leaving the assignment
                # detectably infeasible to the maintained evaluator/shaping path.
                tau[vehicle_idx] = lower
                continue
            tau_fraction = float(stable_sigmoid(float(vehicle_actions[vehicle_idx, 0])))
            tau[vehicle_idx] = lower + tau_fraction * (upper - lower)
    else:
        raise ValueError(f"unknown joint action parameterization: {parameterization!r}")

    return JointAction(
        prices=tuple(float(value) for value in prices),
        bandwidth=tuple(float(value) for value in bandwidth),
        tau_s=tuple(float(value) for value in tau),
        eta=tuple(float(value) for value in eta),
        assignment=tuple(int(value) for value in assignment_array),
    )


__all__ = [
    "CONSTRAINT_AWARE_JOINT_ACTION_PARAMETERIZATION",
    "decode_joint_action",
    "decode_leader_latent",
]
