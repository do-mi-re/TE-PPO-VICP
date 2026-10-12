"""Run small fixed-scenario sanity experiments before PPO training.

Exactness note:
- Paper-exact: every sweep point is evaluated through the
  base ``RSULeaderEnv`` with direct leader actions and analytic follower best
  responses.
- Implementation convenience only: the trend summaries, tables, and optional
  JSON export.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.examples.single_step_explain import (
    DEMO_SEED,
    make_demo_action,
    make_demo_config,
    make_demo_scenario,
    run_single_step_report,
)

PRICE_SWEEP = (0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30)
BANDWIDTH_SWEEP = (0.1, 0.2, 0.3, 0.45, 0.6)
OCCLUSION_SWEEP = (0.1, 0.3, 0.5, 0.7, 0.9)
LAMBDA_SCALE_SWEEP = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
_TREND_TOL = 1e-9


def _non_decreasing(values: list[float]) -> bool:
    return all(
        right >= left - _TREND_TOL
        for left, right in zip(values, values[1:], strict=False)
    )


def _non_increasing(values: list[float]) -> bool:
    return all(
        right <= left + _TREND_TOL
        for left, right in zip(values, values[1:], strict=False)
    )


def _strictly_decreasing(values: list[float]) -> bool:
    return all(
        right < left - _TREND_TOL
        for left, right in zip(values, values[1:], strict=False)
    )


def _classify_shape(values: list[float]) -> str:
    if _non_decreasing(values):
        return "monotone_increasing"
    if _non_increasing(values):
        return "monotone_decreasing"
    for peak_idx in range(1, len(values) - 1):
        prefix = values[: peak_idx + 1]
        suffix = values[peak_idx:]
        if _non_decreasing(prefix) and _non_increasing(suffix):
            has_rise = any(
                right > left + _TREND_TOL
                for left, right in zip(prefix, prefix[1:], strict=False)
            )
            has_drop = any(
                right < left - _TREND_TOL
                for left, right in zip(suffix, suffix[1:], strict=False)
            )
            if has_rise and has_drop:
                return "interior_peak"
    return "mixed"


def _selected_rows_for_task(report: dict[str, object], task_id: int) -> list[dict[str, object]]:
    return [
        row
        for row in report["per_candidate"]
        if row["task_id"] == task_id and row["x_i_k"] == 1
    ]


def _decision_for_vehicle(report: dict[str, object], vehicle_id: int) -> dict[str, object]:
    return next(decision for decision in report["per_vehicle_decisions"] if decision["vehicle_id"] == vehicle_id)


def _candidate_row(report: dict[str, object], *, vehicle_id: int, task_id: int) -> dict[str, object]:
    return next(
        row
        for row in report["per_candidate"]
        if row["vehicle_id"] == vehicle_id and row["task_id"] == task_id
    )


def _bandwidth_action_for_vehicle_zero(weight_zero: float) -> dict[str, np.ndarray]:
    base_action = make_demo_action()
    remaining = 1.0 - weight_zero
    base_v1 = float(base_action["bandwidth"][1])
    base_v2 = float(base_action["bandwidth"][2])
    scale = remaining / (base_v1 + base_v2)
    return {
        "prices": np.array(base_action["prices"], dtype=np.float64),
        "bandwidth": np.array([weight_zero, base_v1 * scale, base_v2 * scale], dtype=np.float64),
    }


def run_price_monotonicity_experiment() -> dict[str, object]:
    """Sweep one task's price and summarize follower/leader responses."""

    config = make_demo_config()
    state = make_demo_scenario()
    base_action = make_demo_action()
    rows: list[dict[str, object]] = []
    for price in PRICE_SWEEP:
        action = {
            "prices": np.array([float(base_action["prices"][0]), price], dtype=np.float64),
            "bandwidth": np.array(base_action["bandwidth"], dtype=np.float64),
        }
        report = run_single_step_report(config=config, state=state, action=action, seed=DEMO_SEED)
        selected_rows = _selected_rows_for_task(report, task_id=1)
        rows.append(
            {
                "price": price,
                "participant_count": len(selected_rows),
                "task_total_tau_s": float(sum(float(row["tau_s_s"]) for row in selected_rows)),
                "task_payment": float(sum(price * float(row["q_i_k"]) for row in selected_rows)),
                "rsu_utility": float(report["rsu_utility"]),
                "social_welfare": float(report["social_welfare"]),
                "selected_tasks": list(report["selected_tasks"]),
            }
        )

    social_welfare_shape = _classify_shape([row["social_welfare"] for row in rows])
    notes = []
    if social_welfare_shape != "interior_peak":
        notes.append(f"social welfare on this grid is {social_welfare_shape}")

    return {
        "experiment_id": "A_price_monotonicity",
        "description": "Sweep task 1 price on the fixed demo scenario.",
        "sweep_parameter": {
            "name": "task_1_price",
            "values": list(PRICE_SWEEP),
        },
        "rows": rows,
        "trend_checks": {
            "participants_non_decreasing": _non_decreasing([row["participant_count"] for row in rows]),
            "task_tau_non_decreasing": _non_decreasing([row["task_total_tau_s"] for row in rows]),
            "payment_non_decreasing": _non_decreasing([row["task_payment"] for row in rows]),
            "social_welfare_shape": social_welfare_shape,
        },
        "notes": notes,
    }


def run_bandwidth_monotonicity_experiment() -> dict[str, object]:
    """Increase one vehicle's bandwidth share while redistributing the remainder."""

    config = make_demo_config()
    state = make_demo_scenario()
    rows: list[dict[str, object]] = []
    for weight_zero in BANDWIDTH_SWEEP:
        action = _bandwidth_action_for_vehicle_zero(weight_zero)
        report = run_single_step_report(config=config, state=state, action=action, seed=DEMO_SEED)
        decision = _decision_for_vehicle(report, vehicle_id=0)
        selected_task_id = int(decision["selected_task_id"])
        selected_row = None if selected_task_id < 0 else _candidate_row(
            report,
            vehicle_id=0,
            task_id=selected_task_id,
        )
        rows.append(
            {
                "bandwidth_vehicle_0": weight_zero,
                "selected_task_id": selected_task_id,
                "tau_tr_s": None if selected_row is None else float(selected_row["tau_tr_s"]),
                "E_tr_j": None if selected_row is None else float(selected_row["E_tr_j"]),
                "selected_utility": float(decision["selected_utility"]),
                "bandwidth_allocation": [float(value) for value in action["bandwidth"]],
            }
        )

    valid_rows = [row for row in rows if row["selected_task_id"] >= 0]
    selected_tasks = [row["selected_task_id"] for row in rows]
    selection_regime_changes = len(set(selected_tasks)) > 1
    utility_non_decreasing = False
    if not selection_regime_changes and valid_rows:
        utility_non_decreasing = _non_decreasing([float(row["selected_utility"]) for row in valid_rows])

    notes = []
    if selection_regime_changes:
        notes.append("vehicle 0 changed selected task across the sweep")

    return {
        "experiment_id": "B_bandwidth_monotonicity",
        "description": "Increase vehicle 0 bandwidth and redistribute the remaining budget in the baseline ratio.",
        "sweep_parameter": {
            "name": "vehicle_0_bandwidth",
            "values": list(BANDWIDTH_SWEEP),
        },
        "rows": rows,
        "trend_checks": {
            "tau_tr_strictly_decreasing": _strictly_decreasing([float(row["tau_tr_s"]) for row in valid_rows]),
            "E_tr_strictly_decreasing": _strictly_decreasing([float(row["E_tr_j"]) for row in valid_rows]),
            "selected_utility_non_decreasing_if_task_unchanged": utility_non_decreasing,
            "selection_regime_changes": selection_regime_changes,
        },
        "notes": notes,
    }


def run_occlusion_sensitivity_experiment() -> dict[str, object]:
    """Sweep one occlusion factor while keeping prices and bandwidth fixed."""

    config = make_demo_config()
    base_state = make_demo_scenario()
    base_action = make_demo_action()
    rows: list[dict[str, object]] = []
    for occlusion in OCCLUSION_SWEEP:
        occlusion_rows = [list(row) for row in base_state.occlusion_factors]
        occlusion_rows[0][0] = occlusion
        state = replace(
            base_state,
            occlusion_factors=tuple(tuple(row) for row in occlusion_rows),
        )
        report = run_single_step_report(config=config, state=state, action=base_action, seed=DEMO_SEED)
        candidate = _candidate_row(report, vehicle_id=0, task_id=0)
        decision = _decision_for_vehicle(report, vehicle_id=0)
        rows.append(
            {
                "occlusion": occlusion,
                "A_i_k": float(candidate["A_i_k"]),
                "q_i_k": float(candidate["q_i_k"]),
                "tilde_U_i_k": float(candidate["tilde_U_i_k"]),
                "selected_task_id": int(decision["selected_task_id"]),
            }
        )

    notes = []
    if all(row["selected_task_id"] == rows[0]["selected_task_id"] for row in rows):
        notes.append("selection threshold was not crossed on this occlusion grid")
    if all(row["A_i_k"] == config.spatial_potential_max for row in rows):
        notes.append("spatial potential remains at the saturation bound on this grid")

    return {
        "experiment_id": "C_occlusion_sensitivity",
        "description": "Increase O_{0,0} on the fixed demo scenario while holding prices and bandwidth constant.",
        "sweep_parameter": {
            "name": "occlusion_vehicle_0_task_0",
            "values": list(OCCLUSION_SWEEP),
        },
        "rows": rows,
        "trend_checks": {
            "A_non_increasing": _non_increasing([row["A_i_k"] for row in rows]),
            "q_non_increasing": _non_increasing([row["q_i_k"] for row in rows]),
            "tilde_utility_non_increasing": _non_increasing([row["tilde_U_i_k"] for row in rows]),
            "selection_becomes_less_favorable": (
                any(row["selected_task_id"] != rows[0]["selected_task_id"] for row in rows)
                or rows[-1]["tilde_U_i_k"] < rows[0]["tilde_U_i_k"] - _TREND_TOL
            ),
        },
        "notes": notes,
    }


def run_lambda_sensitivity_experiment() -> dict[str, object]:
    """Scale the fleet-wide energy weights and measure conservative behavior."""

    config = make_demo_config()
    base_state = make_demo_scenario()
    base_action = make_demo_action()
    rows: list[dict[str, object]] = []
    for scale in LAMBDA_SCALE_SWEEP:
        vehicles = tuple(
            replace(vehicle, energy_weight=vehicle.energy_weight * scale)
            for vehicle in base_state.vehicles
        )
        state = replace(base_state, vehicles=vehicles)
        report = run_single_step_report(config=config, state=state, action=base_action, seed=DEMO_SEED)
        decision_vehicle_2 = _decision_for_vehicle(report, vehicle_id=2)
        rows.append(
            {
                "lambda_scale": scale,
                "selected_tasks": list(report["selected_tasks"]),
                "tau_s_star_all": [
                    float(decision["tau_s_star_s"]) for decision in report["per_vehicle_decisions"]
                ],
                "non_participation_ratio": float(
                    np.mean(np.asarray(report["selected_tasks"], dtype=np.int64) == -1)
                ),
                "vehicle_2_selected_task_id": int(decision_vehicle_2["selected_task_id"]),
                "vehicle_2_tau_s_star": float(decision_vehicle_2["tau_s_star_s"]),
                "vehicle_2_selected_utility": float(decision_vehicle_2["selected_utility"]),
            }
        )

    tau_before_dropout = [
        float(row["vehicle_2_tau_s_star"])
        for row in rows
        if row["vehicle_2_selected_task_id"] >= 0
    ]

    return {
        "experiment_id": "D_lambda_sensitivity",
        "description": "Scale all vehicle energy weights and track conservativeness plus drop-out behavior.",
        "sweep_parameter": {
            "name": "lambda_scale",
            "values": list(LAMBDA_SCALE_SWEEP),
        },
        "rows": rows,
        "trend_checks": {
            "non_participation_ratio_non_decreasing": _non_decreasing(
                [row["non_participation_ratio"] for row in rows]
            ),
            "vehicle_2_tau_non_increasing_before_dropout": _non_increasing(tau_before_dropout),
            "vehicle_2_eventually_drops_out": any(row["vehicle_2_selected_task_id"] < 0 for row in rows[1:]),
        },
        "notes": [],
    }


def run_all_sanity_experiments() -> dict[str, object]:
    """Run the four fixed sanity experiments and return one combined payload."""

    experiments = [
        run_price_monotonicity_experiment(),
        run_bandwidth_monotonicity_experiment(),
        run_occlusion_sensitivity_experiment(),
        run_lambda_sensitivity_experiment(),
    ]
    return {
        "seed": DEMO_SEED,
        "base_scenario": "fixed_demo_scenario",
        "experiments": experiments,
    }


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _print_table(title: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        print(f"\n{title}\n(no rows)")
        return
    columns = list(rows[0].keys())
    str_rows = [[_format_value(row[column]) for column in columns] for row in rows]
    widths = [len(column) for column in columns]
    for row in str_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    print(f"\n{title}")
    print(" | ".join(column.ljust(widths[idx]) for idx, column in enumerate(columns)))
    print("-+-".join("-" * width for width in widths))
    for row in str_rows:
        print(" | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)))


def main() -> None:
    """Run the four sanity experiments and print the combined summary."""

    parser = argparse.ArgumentParser(description="Run fixed-scenario sanity experiments.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    payload = run_all_sanity_experiments()
    for experiment in payload["experiments"]:
        print(f"\n{experiment['experiment_id']}: {experiment['description']}")
        print(f"sweep_parameter: {experiment['sweep_parameter']}")
        print(f"trend_checks: {experiment['trend_checks']}")
        if experiment["notes"]:
            print(f"notes: {experiment['notes']}")
        _print_table("rows", experiment["rows"])

    print("\nRaw JSON")
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
