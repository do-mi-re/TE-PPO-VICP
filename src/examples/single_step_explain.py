"""Explain one paper-faithful Stackelberg interaction with a fixed scenario.

Exactness note:
- Paper-exact: the environment step, follower best
  responses, and all deterministic formulas are taken directly from the base
  environment and math modules.
- Paper-consistent approximation: geometry and leader actions are manually
  specified to illustrate the bounded model in one scheduling cycle.
- Implementation convenience only: the report and pretty-print layer reshape
  those values into a human-readable trace and JSON payload.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from src.models.communication import channel_gain
from src.models.sensing_quality import spatial_potential
from src.models.utilities import social_welfare
from src.sim.env import RSULeaderEnv
from src.sim.scenario import ScenarioConfig, candidate_limit, task_distance_for_vehicle
from src.sim.state import ChannelState, ScenarioState, TaskState, VehicleState

DEMO_SEED = 20260401


def make_demo_config() -> ScenarioConfig:
    """Return the fixed configuration used by the explainable single-step demo."""

    return ScenarioConfig(
        num_vehicles=3,
        pricing_budget_max=0.3,
        spatial_potential_max=50.0,
        diversity_factor=4.0,
        enable_candidate_pruning=True,
        candidate_pruning_radius_m=80.0,
        candidate_limit_range_m=80.0,
        urgency_profiles=(
            ("high", 3.0, 0.12),
            ("mid", 2.0, 0.30),
        ),
    )


def make_demo_scenario() -> ScenarioState:
    """Return the fixed manual scenario used for the explainable single step."""

    config = make_demo_config()

    def distance(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
        return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])

    task_rois = (
        (108.0, 96.0),
        (164.0, 124.0),
    )
    vehicle_positions = (
        (96.0, 96.0),
        (160.0, 116.0),
        (132.0, 174.0),
    )

    tasks = (
        TaskState(
            task_id=0,
            roi_center_m=task_rois[0],
            beta=3.0,
            tau_max_s=0.12,
            length_m=20.0,
            width_m=6.0,
            candidate_limit=candidate_limit(20.0, config.sensing_range_m, config.diversity_factor),
            urgency_label="high",
        ),
        TaskState(
            task_id=1,
            roi_center_m=task_rois[1],
            beta=2.0,
            tau_max_s=0.30,
            length_m=24.0,
            width_m=6.0,
            candidate_limit=candidate_limit(24.0, config.sensing_range_m, config.diversity_factor),
            urgency_label="mid",
        ),
    )
    vehicles = (
        VehicleState(
            vehicle_id=0,
            position_m=vehicle_positions[0],
            raw_data_rate_bps=40_000_000.0,
            resolution_coeff=14.0,
            cpu_cycles_per_s=18_000_000_000.0,
            sensor_power_w=5.5,
            tx_power_w=config.tx_power_w,
            energy_weight=2.5,
        ),
        VehicleState(
            vehicle_id=1,
            position_m=vehicle_positions[1],
            raw_data_rate_bps=40_000_000.0,
            resolution_coeff=13.0,
            cpu_cycles_per_s=16_000_000_000.0,
            sensor_power_w=6.0,
            tx_power_w=config.tx_power_w,
            energy_weight=3.0,
        ),
        VehicleState(
            vehicle_id=2,
            position_m=vehicle_positions[2],
            raw_data_rate_bps=40_000_000.0,
            resolution_coeff=11.0,
            cpu_cycles_per_s=11_000_000_000.0,
            sensor_power_w=7.5,
            tx_power_w=config.tx_power_w,
            energy_weight=8.0,
        ),
    )
    channels = tuple(
        ChannelState(
            vehicle_id=vehicle.vehicle_id,
            distance_to_rsu_m=distance(vehicle.position_m, config.rsu_position_m),
            fading_power=1.0,
            path_loss_exponent=config.path_loss_exponent,
        )
        for vehicle in vehicles
    )
    distance_to_roi_m = tuple(
        tuple(
            task_distance_for_vehicle(
                vehicle_position_m=vehicle.position_m,
                task_center_m=task.roi_center_m,
                task_length_m=task.length_m,
                task_width_m=task.width_m,
                roi_distance_model=config.roi_distance_model,
                distance_floor_m=config.distance_floor_m,
            )
            for task in tasks
        )
        for vehicle in vehicles
    )
    occlusion_factors = (
        (0.10, 0.70),
        (0.65, 0.10),
        (0.55, 0.45),
    )
    return ScenarioState(
        tasks=tasks,
        vehicles=vehicles,
        channels=channels,
        distance_to_roi_m=distance_to_roi_m,
        occlusion_factors=occlusion_factors,
    )


def make_demo_action() -> dict[str, np.ndarray]:
    """Return the fixed direct leader action used in the demo."""

    return {
        "prices": np.array([0.30, 0.25], dtype=np.float64),
        "bandwidth": np.array([0.45, 0.35, 0.20], dtype=np.float64),
    }


def _serialize_matrix(matrix: tuple[tuple[float, ...], ...]) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix]


def _selected_eval(response: Any) -> Any | None:
    if response.selected_task_id is None:
        return None
    return next(
        evaluation
        for evaluation in response.task_evaluations
        if evaluation.task_id == response.selected_task_id
    )


def _build_social_welfare_inputs(
    state: ScenarioState,
    follower_responses: tuple[Any, ...],
) -> tuple[list[list[float]], list[list[int]], list[list[float]], list[list[float]], list[list[float]]]:
    qualities_by_task: list[list[float]] = [[] for _ in state.tasks]
    selections_by_task: list[list[int]] = [[] for _ in state.tasks]
    sensing_energies_by_task: list[list[float]] = [[] for _ in state.tasks]
    computation_energies_by_task: list[list[float]] = [[] for _ in state.tasks]
    transmission_energies_by_task: list[list[float]] = [[] for _ in state.tasks]

    for task in state.tasks:
        for response in follower_responses:
            evaluation = next(
                candidate for candidate in response.task_evaluations if candidate.task_id == task.task_id
            )
            selected = int(response.selected_task_id == task.task_id and evaluation.feasible)
            qualities_by_task[task.task_id].append(0.0 if evaluation.quality is None else float(evaluation.quality))
            selections_by_task[task.task_id].append(selected)
            sensing_energies_by_task[task.task_id].append(
                0.0 if evaluation.sensing_energy_j is None else float(evaluation.sensing_energy_j)
            )
            computation_energies_by_task[task.task_id].append(
                0.0 if evaluation.computation_energy_j is None else float(evaluation.computation_energy_j)
            )
            transmission_energies_by_task[task.task_id].append(
                0.0 if evaluation.transmission_energy_j is None else float(evaluation.transmission_energy_j)
            )

    return (
        qualities_by_task,
        selections_by_task,
        sensing_energies_by_task,
        computation_energies_by_task,
        transmission_energies_by_task,
    )


def run_single_step_report(
    *,
    config: ScenarioConfig,
    state: ScenarioState,
    action: dict[str, np.ndarray],
    seed: int,
) -> dict[str, object]:
    """Run one base-environment step and return a complete explainable trace."""

    env = RSULeaderEnv(
        scenario_config=config,
        scenario_factory=lambda rng, scenario_config, fixed_state=state: fixed_state,
    )
    env.reset(seed=seed)
    _, reward, terminated, truncated, info = env.step(action)
    follower_responses = tuple(info["follower_responses"])

    candidate_rows: list[dict[str, object]] = []
    for vehicle_idx, vehicle in enumerate(state.vehicles):
        for task in state.tasks:
            evaluation = next(
                candidate
                for candidate in follower_responses[vehicle_idx].task_evaluations
                if candidate.task_id == task.task_id
            )
            candidate_rows.append(
                {
                    "vehicle_id": vehicle_idx,
                    "task_id": task.task_id,
                    "A_i_k": float(
                        spatial_potential(
                            resolution_coeff=vehicle.resolution_coeff,
                            occlusion_factor=state.occlusion_factors[vehicle_idx][task.task_id],
                            distance_to_roi_m=state.distance_to_roi_m[vehicle_idx][task.task_id],
                            sensing_range_m=config.sensing_range_m,
                            sensing_decay_exponent=config.sensing_decay_exponent,
                            max_value=config.spatial_potential_max,
                        )
                    ),
                    "q_i_k": 0.0 if evaluation.quality is None else float(evaluation.quality),
                    "tau_s_s": 0.0 if evaluation.tau_s_s is None else float(evaluation.tau_s_s),
                    "tau_c_s": (
                        0.0 if evaluation.computation_latency_s is None else float(evaluation.computation_latency_s)
                    ),
                    "tau_tr_s": (
                        0.0 if evaluation.transmission_latency_s is None else float(evaluation.transmission_latency_s)
                    ),
                    "E_s_j": 0.0 if evaluation.sensing_energy_j is None else float(evaluation.sensing_energy_j),
                    "E_c_j": (
                        0.0 if evaluation.computation_energy_j is None else float(evaluation.computation_energy_j)
                    ),
                    "E_tr_j": (
                        0.0 if evaluation.transmission_energy_j is None else float(evaluation.transmission_energy_j)
                    ),
                    "tilde_U_i_k": float(evaluation.utility),
                    "x_i_k": int(follower_responses[vehicle_idx].selected_task_id == task.task_id),
                    "eta_star": 0.0 if evaluation.eta is None else float(evaluation.eta),
                    "latency_total_s": 0.0 if evaluation.total_latency_s is None else float(evaluation.total_latency_s),
                    "deadline_s": float(task.tau_max_s),
                    "feasible": bool(evaluation.feasible),
                }
            )

    vehicle_decisions: list[dict[str, object]] = []
    for response in follower_responses:
        x_vector = [
            int(response.selected_task_id == task.task_id)
            for task in state.tasks
        ]
        vehicle_decisions.append(
            {
                "vehicle_id": response.vehicle_id,
                "selected_task_id": -1 if response.selected_task_id is None else int(response.selected_task_id),
                "x_i_k": x_vector,
                "tau_s_star_s": 0.0 if response.selected_tau_s_s is None else float(response.selected_tau_s_s),
                "eta_star": 0.0 if response.selected_eta is None else float(response.selected_eta),
                "selected_utility": float(response.selected_utility),
            }
        )

    (
        qualities_by_task,
        selections_by_task,
        sensing_energies_by_task,
        computation_energies_by_task,
        transmission_energies_by_task,
    ) = _build_social_welfare_inputs(state, follower_responses)
    welfare = social_welfare(
        task_weights_beta=[task.beta for task in state.tasks],
        qualities_by_task=qualities_by_task,
        selections_by_task=selections_by_task,
        energy_weights_by_vehicle=[vehicle.energy_weight for vehicle in state.vehicles],
        sensing_energies_by_task=sensing_energies_by_task,
        computation_energies_by_task=computation_energies_by_task,
        transmission_energies_by_task=transmission_energies_by_task,
        total_bandwidth_hz=config.system_bandwidth_hz,
        bandwidth_cost_coeff=config.bandwidth_cost_coeff_per_hz,
        allocated_bandwidth_fractions=list(action["bandwidth"]),
    )

    selected_latency = []
    for response in follower_responses:
        evaluation = _selected_eval(response)
        if evaluation is None or evaluation.total_latency_s is None:
            continue
        deadline = state.tasks[evaluation.task_id].tau_max_s
        selected_latency.append(
            {
                "vehicle_id": response.vehicle_id,
                "task_id": evaluation.task_id,
                "total_latency_s": float(evaluation.total_latency_s),
                "deadline_s": float(deadline),
                "satisfied": bool(evaluation.total_latency_s <= deadline + config.bisection_tol),
            }
        )

    report: dict[str, object] = {
        "seed": seed,
        "exactness": {
            "base_step_path": "paper-exact",
            "report_layer": "implementation convenience only",
        },
        "leader_action": {
            "prices": [float(value) for value in action["prices"]],
            "bandwidth": [float(value) for value in action["bandwidth"]],
        },
        "scenario": {
            "rsu_position_m": [float(value) for value in config.rsu_position_m],
            "task_roi_centers_m": [[float(value) for value in task.roi_center_m] for task in state.tasks],
            "vehicle_positions_m": [[float(value) for value in vehicle.position_m] for vehicle in state.vehicles],
            "distance_to_rsu_m": [float(channel.distance_to_rsu_m) for channel in state.channels],
            "channel_gain": [
                float(
                    channel_gain(
                        fading_power=channel.fading_power,
                        distance_to_rsu_m=channel.distance_to_rsu_m,
                        path_loss_exponent=channel.path_loss_exponent,
                    )
                )
                for channel in state.channels
            ],
            "distance_to_roi_m": _serialize_matrix(state.distance_to_roi_m),
            "occlusion_factor": _serialize_matrix(state.occlusion_factors),
        },
        "per_candidate": candidate_rows,
        "per_vehicle_decisions": vehicle_decisions,
        "selected_tasks": [int(value) for value in info["selected_tasks"]],
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "rsu_utility": float(reward),
        "social_welfare": float(welfare),
        "utility_components": {
            "task_values": [float(value) for value in info["utility_components"]["task_values"]],
            "vehicle_selected_utilities": [
                float(value) for value in info["utility_components"]["vehicle_selected_utilities"]
            ],
            "bandwidth_cost": float(info["utility_components"]["bandwidth_cost"]),
            "effective_task_qualities": [
                float(value) for value in info["utility_components"]["effective_task_qualities"]
            ],
            "participant_counts": [int(value) for value in info["utility_components"]["participant_counts"]],
        },
        "constraints": {
            "leader_action": {
                "prices_in_bounds": bool(
                    np.all(action["prices"] >= 0.0) and np.all(action["prices"] <= config.pricing_budget_max)
                ),
                "bandwidth_nonnegative": bool(np.all(action["bandwidth"] >= 0.0)),
                "bandwidth_sum_leq_1": bool(float(np.sum(action["bandwidth"])) <= 1.0 + 1e-12),
            },
            "selected_latency": selected_latency,
            "at_most_one_task_per_vehicle": [
                {
                    "vehicle_id": decision["vehicle_id"],
                    "selected_count": int(sum(int(value) for value in decision["x_i_k"])),
                    "satisfied": bool(sum(int(value) for value in decision["x_i_k"]) <= 1),
                }
                for decision in vehicle_decisions
            ],
            "violated_constraints": {
                "follower_infeasible_candidates": list(info["violated_constraints"]["follower_infeasible_candidates"]),
                "selected_latency": list(info["violated_constraints"]["selected_latency"]),
            },
        },
    }

    return report


def run_demo() -> dict[str, object]:
    """Run the fixed demo scenario and return the explainable trace."""

    return run_single_step_report(
        config=make_demo_config(),
        state=make_demo_scenario(),
        action=make_demo_action(),
        seed=DEMO_SEED,
    )


def _format_float(value: float) -> str:
    if math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1e-12):
        return "0"
    return f"{value:.6f}"


def _print_table(title: str, columns: list[str], rows: list[list[str]]) -> None:
    print(f"\n{title}")
    widths = [len(column) for column in columns]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    print(" | ".join(column.ljust(widths[idx]) for idx, column in enumerate(columns)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)))


def main() -> None:
    """Run the demo and print both readable tables and the raw JSON trace."""

    report = run_demo()
    print("Single-Step Stackelberg Example")
    print(f"seed: {report['seed']}")
    print(f"leader prices p_k: {report['leader_action']['prices']}")
    print(f"leader bandwidth w_i: {report['leader_action']['bandwidth']}")
    print(f"selected_tasks: {report['selected_tasks']}")
    print(f"rsu_utility: {_format_float(float(report['rsu_utility']))}")
    print(f"social_welfare: {_format_float(float(report['social_welfare']))}")

    scenario = report["scenario"]
    geometry_rows = []
    for vehicle_idx, position in enumerate(scenario["vehicle_positions_m"]):
        row = [
            str(vehicle_idx),
            str(position),
            _format_float(float(scenario["distance_to_rsu_m"][vehicle_idx])),
            _format_float(float(scenario["channel_gain"][vehicle_idx])),
            str(scenario["distance_to_roi_m"][vehicle_idx]),
            str(scenario["occlusion_factor"][vehicle_idx]),
        ]
        geometry_rows.append(row)
    _print_table(
        "Geometry and Channels",
        ["vehicle", "position_m", "d_i", "g_i", "d_i_k", "O_i_k"],
        geometry_rows,
    )

    candidate_rows = []
    for candidate in report["per_candidate"]:
        candidate_rows.append(
            [
                str(candidate["vehicle_id"]),
                str(candidate["task_id"]),
                _format_float(float(candidate["A_i_k"])),
                _format_float(float(candidate["q_i_k"])),
                _format_float(float(candidate["tau_s_s"])),
                _format_float(float(candidate["tau_c_s"])),
                _format_float(float(candidate["tau_tr_s"])),
                _format_float(float(candidate["E_s_j"])),
                _format_float(float(candidate["E_c_j"])),
                _format_float(float(candidate["E_tr_j"])),
                _format_float(float(candidate["tilde_U_i_k"])),
                str(bool(candidate["feasible"])),
            ]
        )
    _print_table(
        "Per-Candidate Metrics",
        [
            "vehicle",
            "task",
            "A_i_k",
            "q_i_k",
            "tau_s",
            "tau_c",
            "tau_tr",
            "E_s",
            "E_c",
            "E_tr",
            "tilde_U",
            "feasible",
        ],
        candidate_rows,
    )

    decision_rows = []
    for decision in report["per_vehicle_decisions"]:
        decision_rows.append(
            [
                str(decision["vehicle_id"]),
                str(decision["selected_task_id"]),
                str(decision["x_i_k"]),
                _format_float(float(decision["tau_s_star_s"])),
                _format_float(float(decision["eta_star"])),
                _format_float(float(decision["selected_utility"])),
            ]
        )
    _print_table(
        "Per-Vehicle Decisions",
        ["vehicle", "selected_task", "x_i_k", "tau_s*", "eta*", "selected_utility"],
        decision_rows,
    )

    constraint_rows = []
    for item in report["constraints"]["selected_latency"]:
        constraint_rows.append(
            [
                str(item["vehicle_id"]),
                str(item["task_id"]),
                _format_float(float(item["total_latency_s"])),
                _format_float(float(item["deadline_s"])),
                str(bool(item["satisfied"])),
            ]
        )
    _print_table(
        "Selected Latency Checks",
        ["vehicle", "task", "tau_total", "deadline", "satisfied"],
        constraint_rows,
    )

    print("\nRaw JSON")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
