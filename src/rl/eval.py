"""Evaluation helpers for trained leader PPO checkpoints.

Exactness label: implementation convenience only; metrics expose, but do not
redefine, the deterministic environment quantities.
"""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Any, Final

import gymnasium
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv

MIN_VALID_TASK_QUALITY_SUM: Final[float] = 1.0
MIN_VALID_RAW_DATA_BITS: Final[float] = 100_000.0
MIN_VALID_ENERGY_J: Final[float] = 1e-4
URGENCY_LEVELS: Final[tuple[str, ...]] = ("high", "mid", "low")

__all__ = [
    "EPISODE_METRIC_NAMES",
    "SWEEP_METRIC_NAMES",
    "episode_metrics_from_info",
    "evaluate_model",
    "write_episode_rows_csv",
    "synchronize_accelerator",
]


def synchronize_accelerator() -> None:
    """Make asynchronous CUDA/MPS kernels visible to wall-clock timing."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.synchronize()
    except (ImportError, RuntimeError):
        pass


# The ordered list of scalar metric keys emitted by ``episode_metrics_from_info``.
# Downstream code (sweep aggregation, paper-figure plotting) treats this as the
# authoritative schema; adding a new metric requires updating this tuple AND the
# returned dict below, and the ``_validate_metric_schema`` dev-check catches drift.
EPISODE_METRIC_NAMES: Final[tuple[str, ...]] = (
    "raw_rsu_reward",
    "te_ppo_reward",
    "penalty_fail",
    "penalty_lat",
    "penalty_part",
    "task_failure_count",
    "participant_deficit_total",
    "latency_overshoot_total",
    "task_value_sum",
    "payment_sum",
    "bandwidth_cost",
    "social_welfare",
    "weighted_energy_total",
    "effective_task_quality_sum",
    "perception_quality_total",
    "participant_count_sum",
    "energy_sensing_sum",
    "energy_computation_sum",
    "energy_transmission_sum",
    "energy_total_sum",
    "selected_latency_ratio_mean",
    "effective_mean_latency_s",
    "quality_per_joule",
    "computation_to_tx_energy_ratio",
    "mean_selected_latency_s",
    "mean_computation_latency_s",
    "mean_transmission_latency_s",
    "average_price",
    "task_success_rate",
    "task_count_high",
    "task_failure_count_high",
    "task_success_rate_high",
    "task_count_mid",
    "task_failure_count_mid",
    "task_success_rate_mid",
    "task_count_low",
    "task_failure_count_low",
    "task_success_rate_low",
    "total_raw_data_bits",
    "total_transmitted_data_bits",
    "task_selection_idle",
    "task_selection_high",
    "task_selection_mid",
    "task_selection_low",
    "num_vehicles_in_episode",
    "selected_vehicle_count",
    "service_valid",
    "zero_service_indicator",
    # Solver-level diagnostics surfaced from ``info['violated_constraints']``.
    # These make silent follower-solver downgrades observable in sweep rows.
    "follower_infeasible_count",
    "latency_violation_count",
    # Candidate-pruning diagnostics.  These are episode-static for a sampled
    # scenario and let candidate-pruning distinguish a genuinely active pruning setting from
    # a saturated grid cell that happens to admit the same pairs.
    "candidate_admitted_pair_count",
    "candidate_pruned_pair_count",
    "candidate_pruned_pair_ratio",
    "candidate_limit_pair_count",
    "candidate_cap_saturated",
    "candidate_vehicle_count",
    "bandwidth_total",
    "bandwidth_admitted_sum",
    "bandwidth_nonadmitted_sum",
    "bandwidth_selected_sum",
    "bandwidth_idle_sum",
    "bandwidth_nonadmitted_ratio",
    "bandwidth_selected_ratio",
    "bandwidth_eligible_entropy",
    "candidate_pruning_time_ms",
    "follower_solve_time_ms",
    "aggregation_time_ms",
)


# The subset of metrics that the sweep pipeline actually aggregates (mean/std
# over eval seeds and then over train seeds). The rest are still attached to
# per-episode rows but not rolled up.
SWEEP_METRIC_NAMES: Final[tuple[str, ...]] = (
    "raw_rsu_reward",
    "social_welfare",
    "effective_task_quality_sum",
    "perception_quality_total",
    "payment_sum",
    "mean_selected_latency_s",
    "effective_mean_latency_s",
    "mean_computation_latency_s",
    "mean_transmission_latency_s",
    "average_price",
    "task_success_rate",
    "task_count_high",
    "task_failure_count_high",
    "task_success_rate_high",
    "task_count_mid",
    "task_failure_count_mid",
    "task_success_rate_mid",
    "task_count_low",
    "task_failure_count_low",
    "task_success_rate_low",
    "total_raw_data_bits",
    "total_transmitted_data_bits",
    "task_selection_idle",
    "task_selection_high",
    "task_selection_mid",
    "task_selection_low",
    "num_vehicles_in_episode",
    "selected_vehicle_count",
    "service_valid",
    "zero_service_indicator",
    "energy_total_sum",
    "quality_per_joule",
    "follower_infeasible_count",
    "latency_violation_count",
    "candidate_admitted_pair_count",
    "candidate_pruned_pair_count",
    "candidate_pruned_pair_ratio",
    "candidate_limit_pair_count",
    "candidate_cap_saturated",
    "candidate_vehicle_count",
    "bandwidth_total",
    "bandwidth_admitted_sum",
    "bandwidth_nonadmitted_sum",
    "bandwidth_selected_sum",
    "bandwidth_idle_sum",
    "bandwidth_nonadmitted_ratio",
    "bandwidth_selected_ratio",
    "bandwidth_eligible_entropy",
    "candidate_pruning_time_ms",
    "follower_solve_time_ms",
    "aggregation_time_ms",
    "inference_wall_time_ms",
)


def _task_success_metrics_by_urgency(
    effective_task_qualities: np.ndarray,
    urgency_labels: tuple[str, ...],
) -> dict[str, float]:
    """Return auditable per-urgency task counts, failures, and success rates.

    Exactness label: implementation convenience only. Success uses the same
    maintained indicator as the aggregate metric: I_k = 1[Q_k > 0].
    """

    metrics: dict[str, float] = {}
    if effective_task_qualities.size != len(urgency_labels):
        if effective_task_qualities.size == 0 and not urgency_labels:
            for urgency in URGENCY_LEVELS:
                metrics[f"task_count_{urgency}"] = 0.0
                metrics[f"task_failure_count_{urgency}"] = 0.0
                metrics[f"task_success_rate_{urgency}"] = float("nan")
            return metrics
        raise ValueError(
            "task urgency labels and effective task qualities must have equal length"
        )
    unknown = sorted(set(urgency_labels) - set(URGENCY_LEVELS))
    if unknown:
        raise ValueError(f"unknown task urgency label(s): {unknown}")

    failures = effective_task_qualities <= 0.0
    labels = np.asarray(urgency_labels, dtype=object)
    for urgency in URGENCY_LEVELS:
        mask = labels == urgency
        count = int(np.sum(mask))
        failure_count = int(np.sum(failures[mask]))
        metrics[f"task_count_{urgency}"] = float(count)
        metrics[f"task_failure_count_{urgency}"] = float(failure_count)
        metrics[f"task_success_rate_{urgency}"] = (
            1.0 - failure_count / count if count > 0 else float("nan")
        )
    return metrics


def episode_metrics_from_info(
    info: dict[str, object],
    *,
    episode_reward: float,
) -> dict[str, float]:
    """Extract per-episode scalar metrics from the environment ``info`` payload.

    The returned dict's keys must match :data:`EPISODE_METRIC_NAMES` — this is
    checked once at module load via :func:`_validate_metric_schema`.
    """

    raw_reward = float(info.get("raw_rsu_reward", episode_reward))
    te_ppo_reward = float(info.get("te_ppo_reward", episode_reward))
    utility_components = dict(info.get("utility_components", {}))
    energy_components = dict(info.get("energy_components", {}))
    task_values = np.asarray(utility_components.get("task_values", []), dtype=np.float64)
    effective_task_qualities = np.asarray(
        utility_components.get("effective_task_qualities", []),
        dtype=np.float64,
    )
    participant_counts = np.asarray(utility_components.get("participant_counts", []), dtype=np.float64)
    bandwidth_cost = float(utility_components.get("bandwidth_cost", 0.0))
    task_value_sum = float(np.sum(task_values))
    payment_sum = task_value_sum - bandwidth_cost - raw_reward
    weighted_energy_total = float(utility_components.get("weighted_energy_total", 0.0))
    # Default to task_value_sum − weighted_energy − bandwidth_cost when env
    # didn't supply a precomputed value; matches utilities.social_welfare().
    social_welfare = float(
        utility_components.get(
            "social_welfare", task_value_sum - weighted_energy_total - bandwidth_cost
        )
    )

    sensing_energy_sum = float(np.sum(np.asarray(energy_components.get("sensing_j", []), dtype=np.float64)))
    computation_energy_sum = float(np.sum(np.asarray(energy_components.get("computation_j", []), dtype=np.float64)))
    transmission_energy_sum = float(np.sum(np.asarray(energy_components.get("transmission_j", []), dtype=np.float64)))
    total_energy_sum = float(np.sum(np.asarray(energy_components.get("total_j", []), dtype=np.float64)))
    effective_task_quality_sum = float(np.sum(effective_task_qualities))
    perception_quality_total = float(
        utility_components.get("beta_weighted_quality_total", 0.0)
    )
    if total_energy_sum > 1e-12:
        quality_per_joule = effective_task_quality_sum / total_energy_sum
    else:
        quality_per_joule = 0.0
    if transmission_energy_sum > 1e-12:
        computation_to_tx_energy_ratio = computation_energy_sum / transmission_energy_sum
    elif computation_energy_sum > 0.0:
        computation_to_tx_energy_ratio = math.inf
    else:
        computation_to_tx_energy_ratio = 0.0

    obs_snapshot = info.get("observation_snapshot", {})
    deadlines = np.asarray(
        obs_snapshot.get("tasks", {}).get("tau_max_s", []),
        dtype=np.float64,
    )
    latency_ratios: list[float] = []
    latency_absolutes: list[float] = []
    computation_latencies: list[float] = []
    transmission_latencies: list[float] = []
    selected_latency_by_vehicle: dict[int, float] = {}
    for response in info.get("follower_responses", ()):
        selected_task_id = getattr(response, "selected_task_id", None)
        if selected_task_id is None or int(selected_task_id) < 0:
            continue
        selected_eval = next(
            (
                evaluation
                for evaluation in getattr(response, "task_evaluations", ())
                if evaluation.task_id == selected_task_id
            ),
            None,
        )
        if selected_eval is None or selected_eval.total_latency_s is None:
            continue
        selected_latency = float(selected_eval.total_latency_s)
        latency_absolutes.append(selected_latency)
        vehicle_id = getattr(response, "vehicle_id", None)
        if vehicle_id is not None:
            selected_latency_by_vehicle[int(vehicle_id)] = selected_latency
        if selected_eval.computation_latency_s is not None:
            computation_latencies.append(float(selected_eval.computation_latency_s))
        if selected_eval.transmission_latency_s is not None:
            transmission_latencies.append(float(selected_eval.transmission_latency_s))
        deadline = float(deadlines[int(selected_task_id)]) if deadlines.size else 0.0
        if deadline > 0.0:
            latency_ratios.append(float(selected_eval.total_latency_s) / deadline)
    selected_latency_ratio_mean = float(np.mean(latency_ratios)) if latency_ratios else float("nan")
    mean_selected_latency_s = float(np.mean(latency_absolutes)) if latency_absolutes else float("nan")
    mean_computation_latency_s = float(np.mean(computation_latencies)) if computation_latencies else float("nan")
    mean_transmission_latency_s = float(np.mean(transmission_latencies)) if transmission_latencies else float("nan")

    leader_action = info.get("leader_action", {})
    prices = np.asarray(leader_action.get("prices", []), dtype=np.float64)
    average_price = float(np.mean(prices)) if prices.size > 0 else 0.0

    num_tasks = int(info.get("num_tasks", len(effective_task_qualities)))
    task_failure_count = float(info.get("task_failure_count", float(np.sum(effective_task_qualities <= 0.0))))
    task_success_rate = 1.0 - task_failure_count / num_tasks if num_tasks > 0 else 0.0
    task_urgency_labels = tuple(
        str(value) for value in info.get("task_urgency_labels", ())
    )
    urgency_success_metrics = _task_success_metrics_by_urgency(
        effective_task_qualities,
        task_urgency_labels,
    )

    raw_data_rate_bps = float(info.get("raw_data_rate_bps", 40_000_000.0))
    sensing_durations = np.asarray(info.get("follower_sensing_durations_s", []), dtype=np.float64)
    compression_ratios = np.asarray(info.get("follower_compression_ratios", []), dtype=np.float64)
    selected_tasks = np.asarray(info.get("selected_tasks", []), dtype=np.int64)
    selected_vehicle_count = float(np.sum(selected_tasks >= 0))
    timeout_latency = float(np.max(deadlines)) if deadlines.size else float("nan")
    effective_latencies: list[float] = []
    for idx, task_id in enumerate(selected_tasks):
        if int(task_id) >= 0 and idx in selected_latency_by_vehicle:
            effective_latencies.append(selected_latency_by_vehicle[idx])
        elif not math.isnan(timeout_latency):
            effective_latencies.append(timeout_latency)
    effective_mean_latency_s = (
        float(np.mean(effective_latencies))
        if effective_latencies
        else mean_selected_latency_s
    )
    total_raw_data_bits = 0.0
    total_transmitted_data_bits = 0.0
    for idx in range(len(selected_tasks)):
        if selected_tasks[idx] >= 0:
            raw_vol = raw_data_rate_bps * sensing_durations[idx]
            total_raw_data_bits += raw_vol
            eta = max(compression_ratios[idx], 1.0)
            total_transmitted_data_bits += raw_vol / eta
    service_valid = float(
        selected_vehicle_count > 0.0
        and effective_task_quality_sum >= MIN_VALID_TASK_QUALITY_SUM
        and total_raw_data_bits >= MIN_VALID_RAW_DATA_BITS
        and total_energy_sum >= MIN_VALID_ENERGY_J
    )
    zero_service_indicator = 1.0 - service_valid

    selection_idle = 0
    selection_high = 0
    selection_mid = 0
    selection_low = 0
    for idx in range(len(selected_tasks)):
        tid = int(selected_tasks[idx])
        if tid < 0:
            selection_idle += 1
        elif task_urgency_labels and tid < len(task_urgency_labels):
            label = task_urgency_labels[tid]
            if label == "high":
                selection_high += 1
            elif label == "mid":
                selection_mid += 1
            elif label == "low":
                selection_low += 1
    num_vehicles_in_episode = len(selected_tasks)

    # Surface solver-level downgrades (``follower_solver`` returns an
    # infeasible evaluation / deadline overruns) so aggregation can flag
    # silent numerical failures instead of letting them disappear into NaN.
    violated_constraints = info.get("violated_constraints") or {}
    follower_infeasible_count = float(
        len(violated_constraints.get("follower_infeasible_candidates", []) or [])
    )
    latency_violation_count = float(
        len(violated_constraints.get("selected_latency", []) or [])
    )
    pruning_diagnostics = info.get("candidate_pruning") or {}
    candidate_admitted_pair_count = float(
        pruning_diagnostics.get("admitted_pair_count", 0.0)
    )
    candidate_pruned_pair_count = float(
        pruning_diagnostics.get("pruned_pair_count", 0.0)
    )
    candidate_pair_count = (
        candidate_admitted_pair_count + candidate_pruned_pair_count
    )
    candidate_pruned_pair_ratio = (
        candidate_pruned_pair_count / candidate_pair_count
        if candidate_pair_count > 0.0
        else 0.0
    )
    candidate_limits = np.asarray(
        pruning_diagnostics.get("candidate_limits_by_task", []), dtype=np.float64
    )
    candidate_limit_pair_count = float(np.sum(candidate_limits))
    candidate_cap_saturated = float(
        bool(pruning_diagnostics.get("enabled", False))
        and candidate_limit_pair_count > 0.0
        and candidate_admitted_pair_count >= candidate_limit_pair_count
    )
    bandwidth_diagnostics = info.get("bandwidth_allocation") or {}

    return {
        "raw_rsu_reward": raw_reward,
        "te_ppo_reward": te_ppo_reward,
        "penalty_fail": float(info.get("penalty_fail", 0.0)),
        "penalty_lat": float(info.get("penalty_lat", 0.0)),
        "penalty_part": float(info.get("penalty_part", 0.0)),
        "task_failure_count": task_failure_count,
        "participant_deficit_total": float(info.get("participant_deficit_total", 0.0)),
        "latency_overshoot_total": float(info.get("latency_overshoot_total", 0.0)),
        "task_value_sum": task_value_sum,
        "payment_sum": payment_sum,
        "bandwidth_cost": bandwidth_cost,
        "social_welfare": social_welfare,
        "weighted_energy_total": weighted_energy_total,
        "effective_task_quality_sum": effective_task_quality_sum,
        "perception_quality_total": perception_quality_total,
        "participant_count_sum": float(np.sum(participant_counts)),
        "energy_sensing_sum": sensing_energy_sum,
        "energy_computation_sum": computation_energy_sum,
        "energy_transmission_sum": transmission_energy_sum,
        "energy_total_sum": total_energy_sum,
        "selected_latency_ratio_mean": selected_latency_ratio_mean,
        "effective_mean_latency_s": effective_mean_latency_s,
        "quality_per_joule": quality_per_joule,
        "computation_to_tx_energy_ratio": computation_to_tx_energy_ratio,
        "mean_selected_latency_s": mean_selected_latency_s,
        "mean_computation_latency_s": mean_computation_latency_s,
        "mean_transmission_latency_s": mean_transmission_latency_s,
        "average_price": average_price,
        "task_success_rate": task_success_rate,
        **urgency_success_metrics,
        "total_raw_data_bits": total_raw_data_bits,
        "total_transmitted_data_bits": total_transmitted_data_bits,
        "task_selection_idle": float(selection_idle),
        "task_selection_high": float(selection_high),
        "task_selection_mid": float(selection_mid),
        "task_selection_low": float(selection_low),
        "num_vehicles_in_episode": float(num_vehicles_in_episode),
        "selected_vehicle_count": selected_vehicle_count,
        "service_valid": service_valid,
        "zero_service_indicator": zero_service_indicator,
        "follower_infeasible_count": follower_infeasible_count,
        "latency_violation_count": latency_violation_count,
        "candidate_admitted_pair_count": candidate_admitted_pair_count,
        "candidate_pruned_pair_count": candidate_pruned_pair_count,
        "candidate_pruned_pair_ratio": candidate_pruned_pair_ratio,
        "candidate_limit_pair_count": candidate_limit_pair_count,
        "candidate_cap_saturated": candidate_cap_saturated,
        "candidate_vehicle_count": float(
            bandwidth_diagnostics.get("candidate_vehicle_count", 0.0)
        ),
        "bandwidth_total": float(bandwidth_diagnostics.get("total", 0.0)),
        "bandwidth_admitted_sum": float(
            bandwidth_diagnostics.get("admitted", 0.0)
        ),
        "bandwidth_nonadmitted_sum": float(
            bandwidth_diagnostics.get("nonadmitted", 0.0)
        ),
        "bandwidth_selected_sum": float(
            bandwidth_diagnostics.get("selected", 0.0)
        ),
        "bandwidth_idle_sum": float(bandwidth_diagnostics.get("idle", 0.0)),
        "bandwidth_nonadmitted_ratio": float(
            bandwidth_diagnostics.get("nonadmitted_ratio", 0.0)
        ),
        "bandwidth_selected_ratio": float(
            bandwidth_diagnostics.get("selected_ratio", 0.0)
        ),
        "bandwidth_eligible_entropy": float(
            bandwidth_diagnostics.get("eligible_normalized_entropy", 0.0)
        ),
        "candidate_pruning_time_ms": float(info.get("candidate_pruning_time_ms", 0.0)),
        "follower_solve_time_ms": float(info.get("follower_solve_time_ms", 0.0)),
        "aggregation_time_ms": float(info.get("aggregation_time_ms", 0.0)),
    }


def _validate_metric_schema() -> None:
    """Sanity check: ``episode_metrics_from_info`` keys must match the declared schema.

    Runs once at import time against a minimal synthetic info dict; if a
    developer adds a new metric without updating :data:`EPISODE_METRIC_NAMES`,
    this raises immediately instead of silently de-syncing downstream
    aggregation.
    """

    probe = episode_metrics_from_info(
        {
            "raw_rsu_reward": 0.0,
            "te_ppo_reward": 0.0,
            "utility_components": {
                "task_values": [],
                "effective_task_qualities": [],
                "participant_counts": [],
                "bandwidth_cost": 0.0,
            },
            "energy_components": {
                "sensing_j": [],
                "computation_j": [],
                "transmission_j": [],
                "total_j": [],
            },
            "follower_responses": (),
            "leader_action": {"prices": [], "bandwidth": []},
            "num_tasks": 0,
            "selected_tasks": [],
            "follower_sensing_durations_s": [],
            "follower_compression_ratios": [],
            "task_urgency_labels": [],
        },
        episode_reward=0.0,
    )
    declared = set(EPISODE_METRIC_NAMES)
    actual = set(probe.keys())
    if declared != actual:
        missing = declared - actual
        extra = actual - declared
        raise RuntimeError(
            "EPISODE_METRIC_NAMES is out of sync with episode_metrics_from_info. "
            f"Missing: {sorted(missing)}; Extra (undeclared): {sorted(extra)}."
        )


def _aggregate_optional_episode_metrics(infos: list[dict[str, object]]) -> dict[str, float]:
    metric_map: dict[str, list[float]] = {
        "raw_rsu_reward": [],
        "te_ppo_reward": [],
        "social_welfare": [],
        "penalty_fail": [],
        "penalty_lat": [],
        "penalty_part": [],
        "mean_selected_latency_s": [],
        "effective_mean_latency_s": [],
        "task_success_rate": [],
        "task_success_rate_high": [],
        "task_success_rate_mid": [],
        "task_success_rate_low": [],
        "total_raw_data_bits": [],
        "total_transmitted_data_bits": [],
        "task_selection_idle": [],
        "task_selection_high": [],
        "task_selection_mid": [],
        "task_selection_low": [],
        "num_vehicles_in_episode": [],
        "selected_vehicle_count": [],
        "service_valid": [],
        "zero_service_indicator": [],
    }
    for info in infos:
        for key in metric_map:
            value = info.get(key)
            if value is not None:
                metric_map[key].append(float(value))
    result: dict[str, float] = {}
    for key, values in metric_map.items():
        if values:
            result[f"mean_{key}"] = float(np.nanmean(values))
    return result


def evaluate_model(
    model: BaseAlgorithm,
    env: gymnasium.Env[dict[str, object], Any] | VecEnv,
    *,
    seeds: list[int] | tuple[int, ...],
    return_rows: bool = False,
    row_context: dict[str, object] | None = None,
) -> dict[str, float | list[float] | list[int] | list[dict[str, object]]]:
    """Run deterministic evaluation and return scalar summary metrics."""

    if len(seeds) == 0:
        raise ValueError("seeds must not be empty")
    eval_seeds = [int(item) for item in seeds]

    returns: list[float] = []
    lengths: list[int] = []
    infos_seen: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    if isinstance(env, VecEnv):
        if env.num_envs != 1:
            raise ValueError("VecEnv evaluation currently requires num_envs == 1")
        for episode_idx, episode_seed in enumerate(eval_seeds):
            env.seed(episode_seed)
            observation = env.reset()
            synchronize_accelerator()
            actor_started = time.perf_counter()
            action, _ = model.predict(observation, deterministic=True)
            synchronize_accelerator()
            actor_forward_ms = (time.perf_counter() - actor_started) * 1000.0
            env_started = time.perf_counter()
            _, reward, done, infos = env.step(action)
            synchronize_accelerator()
            env_step_ms = (time.perf_counter() - env_started) * 1000.0
            inference_wall_time_ms = actor_forward_ms + env_step_ms
            if not bool(done[0]):
                raise RuntimeError("leader environment evaluation expects one-step episodes")
            episode_reward = float(reward[0])
            returns.append(episode_reward)
            lengths.append(int(infos[0].get("episode", {}).get("l", 1)))
            info = dict(infos[0])
            infos_seen.append(info)
            if return_rows:
                row = {"episode_index": episode_idx, "eval_seed": episode_seed}
                if row_context is not None:
                    row.update(row_context)
                row.update(episode_metrics_from_info(info, episode_reward=episode_reward))
                pruning_ms = float(row.get("candidate_pruning_time_ms", 0.0))
                row["actor_forward_ms"] = actor_forward_ms
                row["env_step_ms"] = env_step_ms
                row["inference_wall_time_ms"] = inference_wall_time_ms
                row["total_cycle_ms"] = pruning_ms + inference_wall_time_ms
                rows.append(row)
    else:
        for episode_idx, episode_seed in enumerate(eval_seeds):
            observation, _ = env.reset(seed=episode_seed)
            synchronize_accelerator()
            actor_started = time.perf_counter()
            action, _ = model.predict(observation, deterministic=True)
            synchronize_accelerator()
            actor_forward_ms = (time.perf_counter() - actor_started) * 1000.0
            env_started = time.perf_counter()
            _, reward, terminated, truncated, info = env.step(action)
            synchronize_accelerator()
            env_step_ms = (time.perf_counter() - env_started) * 1000.0
            inference_wall_time_ms = actor_forward_ms + env_step_ms
            if not terminated and not truncated:
                raise RuntimeError("leader environment evaluation expects one-step episodes")
            episode_reward = float(reward)
            returns.append(episode_reward)
            lengths.append(1)
            scalar_info = dict(info)
            infos_seen.append(scalar_info)
            if return_rows:
                row = {"episode_index": episode_idx, "eval_seed": episode_seed}
                if row_context is not None:
                    row.update(row_context)
                row.update(episode_metrics_from_info(scalar_info, episode_reward=episode_reward))
                pruning_ms = float(row.get("candidate_pruning_time_ms", 0.0))
                row["actor_forward_ms"] = actor_forward_ms
                row["env_step_ms"] = env_step_ms
                row["inference_wall_time_ms"] = inference_wall_time_ms
                row["total_cycle_ms"] = pruning_ms + inference_wall_time_ms
                rows.append(row)

    rewards = np.asarray(returns, dtype=np.float64)
    payload: dict[str, float | list[float] | list[int]] = {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "episode_rewards": [float(value) for value in returns],
        "episode_lengths": [int(value) for value in lengths],
    }
    payload.update(_aggregate_optional_episode_metrics(infos_seen))
    if return_rows:
        payload["episode_rows"] = rows
    return payload


_validate_metric_schema()


def write_episode_rows_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    """Write episode-row records to CSV, using the union of row keys as the header."""

    if not rows:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header: list[str] = []
    for row in rows:
        for key in row:
            if key not in header:
                header.append(str(key))
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
