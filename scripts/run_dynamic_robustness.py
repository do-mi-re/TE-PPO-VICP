"""Replay canonical checkpoints under the robustness dynamic-condition stress cells.

Each evaluation seed is an independent scheduling cycle with freshly sampled
vehicle positions, fading, and occlusion. This is not a temporal mobility
trace; the output records that limitation explicitly. Shape-changing ``V/K``
experiments live in ``run_scalability_experiment.py`` and retrain policies.

Exactness label: implementation convenience only (experiment orchestration).
"""

from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import dataclass
from pathlib import Path

from src.evaluation import orchestration as orch
from src.evaluation.provenance import (
    write_evaluation_contract,
)
from src.evaluation.revision import (
    formal_algorithm_config,
    headline_run_dir,
    headline_specs,
    load_revision_config,
    revision_roots,
    train_seeds_for,
)
from src.evaluation.sweep import (
    aggregate_sweep_rows,
    evaluate_run_dir,
    tag_episode_rows,
    write_sweep_csv,
)


@dataclass(frozen=True)
class StressCell:
    experiment: str
    level: str
    display_value: float
    overrides: tuple[tuple[str, object], ...]


def build_cells(config: dict | None = None) -> tuple[StressCell, ...]:
    cfg = config or load_revision_config()
    scale = cfg["scalability"]
    cells = [
        StressCell(
            "occlusion", str(label), float(sum(band) / 2),
            (("occlusion_range", tuple(float(v) for v in band)),),
        )
        for label, band in scale["occlusion_bands"].items()
    ]
    cells.extend(
        StressCell(
            "path_loss", f"alpha_{float(value):g}", float(value),
            (("path_loss_exponent", float(value)),),
        )
        for value in scale["path_loss_grid"]
    )
    cells.extend(
        StressCell(
            "rsu_bandwidth", f"{float(value) / 1e6:g}MHz", float(value) / 1e6,
            (("system_bandwidth_hz", float(value)),),
        )
        for value in scale["rsu_bandwidth_grid_hz"]
    )
    return tuple(cells)


def _seeds(algorithm: str, cfg: dict) -> tuple[int, ...]:
    if algorithm == "dbas":
        return (int(cfg["seeds"]["sweep_train"][0]),)
    available = train_seeds_for(algorithm, config=cfg)
    return tuple(seed for seed in cfg["seeds"]["sweep_train"] if seed in available)


def run() -> int:
    cfg = load_revision_config()
    algorithms = tuple(cfg["scalability"]["algorithms"])
    missing = [
        headline_run_dir(algorithm, seed, cfg)
        for algorithm in algorithms if algorithm != "dbas"
        for seed in _seeds(algorithm, cfg)
        if not (headline_run_dir(algorithm, seed, cfg) / "summary.json").exists()
    ]
    if missing:
        print("error: canonical headline checkpoints are missing", file=sys.stderr)
        return 1
    seed_bank = str(cfg["scalability"]["dynamic_evaluation_seed_bank"])
    eval_seeds = tuple(cfg["seeds"][seed_bank])
    training_eval_seeds = tuple(cfg["seeds"]["validation"])
    headline_expected = {
        (spec.algorithm, spec.train_seed): orch.materialize_config(
            spec.to_worker_dict(), training_eval_seeds
        )
        for spec in headline_specs(cfg)
        if spec.algorithm != "dbas"
    }
    rows: list[dict[str, object]] = []
    for cell in build_cells(cfg):
        override = dict(cell.overrides)
        print(f"evaluating {cell.experiment}/{cell.level}: {override}")
        for algorithm in algorithms:
            for seed in _seeds(algorithm, cfg):
                if algorithm == "dbas":
                    dbas_config = formal_algorithm_config("dbas", cfg)
                    effective_config = copy.deepcopy(dbas_config)
                    effective_config.setdefault("scenario", {}).update(override)
                    trial = evaluate_run_dir(
                        "dbas", Path("."), eval_seeds,
                        dbas_config=dbas_config,
                        scenario_override=override,
                        expected_config=effective_config,
                    )
                else:
                    trial = evaluate_run_dir(
                        algorithm, headline_run_dir(algorithm, seed, cfg), eval_seeds,
                        scenario_override=override,
                        expected_config=headline_expected[(algorithm, seed)],
                    )
                tag_episode_rows(
                    trial, algorithm_name=algorithm, train_seed=seed,
                    scenario_overrides=override,
                    sweep_param=cell.experiment,
                    sweep_value=cell.display_value,
                )
                for row in trial:
                    row["stress_experiment"] = cell.experiment
                    row["stress_level"] = cell.level
                    row["dynamics_semantics"] = "independent_one_step_scenario_draws"
                    row["runtime_evidence_status"] = (
                        "diagnostic_unwarmed_non_final"
                    )
                rows.extend(trial)
    summary = aggregate_sweep_rows(
        rows,
        group_fields=("algorithm", "stress_experiment", "stress_level", "sweep_value"),
        uncertainty="seed",
    )
    target = revision_roots(cfg)[1] / "dynamic_robustness"
    target.mkdir(parents=True, exist_ok=True)
    contract = {
        "revision_id": cfg["revision_id"],
        "revision_schema_version": int(cfg["schema_version"]),
        "experiment": "dynamic_robustness",
        "evaluation_seed_bank": seed_bank,
        "evaluation_seeds": list(eval_seeds),
        "training_seeds": list(cfg["seeds"]["sweep_train"]),
        "algorithms": list(algorithms),
        "cells": [
            {
                "experiment": cell.experiment,
                "level": cell.level,
                "display_value": cell.display_value,
                "overrides": dict(cell.overrides),
            }
            for cell in build_cells(cfg)
        ],
        "aggregation_unit": "training_seed_mean_over_evaluation_scenarios",
    }
    write_sweep_csv(target / "episodes.csv", rows)
    write_sweep_csv(target / "summary.csv", summary)
    write_evaluation_contract(target, contract)
    print(f"wrote scalability dynamic-condition outputs to {target}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_revision_config()
    if args.dry_run:
        cells = build_cells(cfg)
        seed_bank = str(cfg["scalability"]["dynamic_evaluation_seed_bank"])
        print(f"stress cells={len(cells)}; evaluation seeds={list(cfg['seeds'][seed_bank])}")
        for cell in cells:
            print(f"  {cell.experiment}/{cell.level}: {dict(cell.overrides)}")
        return 0
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
