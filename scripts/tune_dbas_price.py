"""Tune DBAS's fixed price on local calibration scenarios.

Exactness label: paper-consistent approximation. DBAS remains the same
distance-based bandwidth heuristic; this script only calibrates its fixed
price so it is not an arbitrary high-payment baseline.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.revision import dbas_calibration_contract_digest, load_revision_config
from src.evaluation.sweep import (
    build_sweep_config,
    evaluate_run_dir,
    load_algorithm_config,
)
from src.sim.scenario import ScenarioConfig

VEHICLE_GRID = (10, 20, 30, 40, 50, 60)
BANDWIDTH_GRID_HZ = (
    500_000.0,
    1_000_000.0,
    2_000_000.0,
    5_000_000.0,
    10_000_000.0,
    20_000_000.0,
    30_000_000.0,
    40_000_000.0,
)
MIN_TASK_SUCCESS_RATE = 0.75
MIN_SELECTED_VEHICLE_COUNT = 0.0


def _mean(rows: list[dict[str, object]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row and not math.isnan(float(row[key]))]
    return float(np.mean(values)) if values else float("nan")


def _to_hashable(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple((key, _to_hashable(value[key])) for key in sorted(value))
    if isinstance(value, list | tuple):
        return tuple(_to_hashable(item) for item in value)
    return value


def _scenario_key(
    *,
    base_scenario: dict[str, Any],
    overrides: dict[str, Any],
) -> Any:
    effective = dict(base_scenario)
    effective.update(overrides)
    return _to_hashable(asdict(ScenarioConfig(**effective)))


def _calibration_scenarios(base_config: dict[str, Any]) -> tuple[tuple[str, dict[str, Any]], ...]:
    candidates: list[tuple[str, dict[str, Any]]] = [("default", {})]
    candidates.extend(
        (f"vehicles_{value}", {"num_vehicles": value})
        for value in VEHICLE_GRID
    )
    candidates.extend(
        (f"bandwidth_{int(value)}", {"system_bandwidth_hz": value})
        for value in BANDWIDTH_GRID_HZ
    )

    base_scenario = dict(base_config.get("scenario", {}))
    scenarios: list[tuple[str, dict[str, Any]]] = []
    seen: set[Any] = set()
    for name, overrides in candidates:
        key = _scenario_key(base_scenario=base_scenario, overrides=overrides)
        if key in seen:
            continue
        seen.add(key)
        scenarios.append((name, overrides))
    return tuple(scenarios)


def summarize_candidate(
    *,
    price: float,
    pricing_budget_max: float,
    episode_rows: list[dict[str, object]],
) -> dict[str, float]:
    fixed_price_fraction = price / pricing_budget_max
    return {
        "price": float(price),
        "fixed_price_fraction": float(fixed_price_fraction),
        "episode_count": float(len(episode_rows)),
        "social_welfare_mean": _mean(episode_rows, "social_welfare"),
        "raw_rsu_reward_mean": _mean(episode_rows, "raw_rsu_reward"),
        "task_success_rate_mean": _mean(episode_rows, "task_success_rate"),
        "selected_vehicle_count_mean": _mean(episode_rows, "selected_vehicle_count"),
        "payment_sum_mean": _mean(episode_rows, "payment_sum"),
        "effective_task_quality_sum_mean": _mean(episode_rows, "effective_task_quality_sum"),
    }


def choose_best_candidate(rows: list[dict[str, float]]) -> dict[str, float]:
    """Choose DBAS price by welfare, then RSU utility, then lower price."""

    valid = [
        row for row in rows
        if float(row["task_success_rate_mean"]) >= MIN_TASK_SUCCESS_RATE
        and float(row["selected_vehicle_count_mean"]) > MIN_SELECTED_VEHICLE_COUNT
    ]
    if not valid:
        raise ValueError(
            "no DBAS price candidate satisfied "
            f"task_success_rate_mean >= {MIN_TASK_SUCCESS_RATE} and "
            f"selected_vehicle_count_mean > {MIN_SELECTED_VEHICLE_COUNT}"
        )
    return max(
        valid,
        key=lambda row: (
            float(row["social_welfare_mean"]),
            float(row["raw_rsu_reward_mean"]),
            -float(row["price"]),
        ),
    )


def evaluate_candidate(
    *,
    price: float,
    base_config: dict[str, Any],
    eval_seeds: tuple[int, ...],
) -> dict[str, float]:
    pricing_budget_max = float(base_config.get("scenario", {}).get("pricing_budget_max", 0.3))
    if pricing_budget_max <= 0.0:
        raise ValueError("scenario.pricing_budget_max must be positive")
    if price < 0.0 or price > pricing_budget_max:
        raise ValueError("candidate price must be within [0, pricing_budget_max]")

    fixed_price_fraction = price / pricing_budget_max
    candidate_config = dict(base_config)
    candidate_config["algorithm"] = dict(base_config.get("algorithm", {}))
    candidate_config["algorithm"]["fixed_price_fraction"] = fixed_price_fraction

    episode_rows: list[dict[str, object]] = []
    for scenario_name, overrides in _calibration_scenarios(base_config):
        config = build_sweep_config(
            "dbas",
            base_config=candidate_config,
            scenario_overrides=overrides,
            train_seed=123,
            eval_seeds=eval_seeds,
            run_name_suffix=scenario_name,
        )
        rows = evaluate_run_dir(
            "dbas",
            Path("."),
            eval_seeds,
            dbas_config=config,
        )
        episode_rows.extend(rows)

    return summarize_candidate(
        price=price,
        pricing_budget_max=pricing_budget_max,
        episode_rows=episode_rows,
    )


def _parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write empty DBAS calibration CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune DBAS fixed price locally")
    parser.add_argument(
        "--output-csv",
        default="outputs/revision_2026_v5/calibration/dbas_price_calibration.csv",
        help="Where to write candidate summaries",
    )
    parser.add_argument(
        "--fast-local",
        action="store_true",
        help="Use two eval seeds for a quick local calibration pass",
    )
    parser.add_argument(
        "--price-grid",
        default=None,
        help="Comma-separated actual prices (default: configs/revision.yaml)",
    )
    args = parser.parse_args()

    revision = load_revision_config()
    base_config = load_algorithm_config("dbas")
    validation_seeds = tuple(revision["seeds"]["validation"])
    eval_seeds = validation_seeds[:2] if args.fast_local else validation_seeds
    prices = (
        _parse_float_list(args.price_grid)
        if args.price_grid
        else tuple(revision["baseline_calibration"]["dbas_price_grid"])
    )
    summaries = [
        evaluate_candidate(price=price, base_config=base_config, eval_seeds=eval_seeds)
        for price in prices
    ]
    best = choose_best_candidate(summaries)
    _write_csv(Path(args.output_csv), summaries)
    selection_path = Path(
        revision["baseline_calibration"]["dbas_selection_file"]
    )
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    import yaml
    selection_path.write_text(yaml.safe_dump({
        "schema_version": 1,
        "selection_basis": "validation social welfare with service constraint",
        "algorithm.fixed_price_fraction": best["fixed_price_fraction"],
        "price": best["price"],
        "validation_seed_count": len(eval_seeds),
        "calibration_contract_digest": dbas_calibration_contract_digest(
            revision, price_grid=prices, validation_seeds=eval_seeds
        ),
    }, sort_keys=False), encoding="utf-8")

    print(f"wrote {args.output_csv}")
    print(f"wrote {selection_path}")
    print(
        "recommended DBAS fixed_price_fraction="
        f"{best['fixed_price_fraction']:.12g} "
        f"(price={best['price']:.12g}, "
        f"social_welfare_mean={best['social_welfare_mean']:.6g}, "
        f"raw_rsu_reward_mean={best['raw_rsu_reward_mean']:.6g})"
    )


if __name__ == "__main__":
    main()
