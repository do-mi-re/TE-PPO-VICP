"""Train and evaluate the full-range CPU robustness experiment.

Exactness label: implementation convenience only.  The optional CPU regime
sampler is documented as a paper-consistent approximation in ScenarioConfig;
this script manages artifact identities, checkpoint validation, and aggregation.
"""

from __future__ import annotations

import argparse
import copy
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.evaluation import orchestration as orch
from src.evaluation.provenance import (
    config_digest,
    write_evaluation_contract,
)
from src.evaluation.revision import (
    cpu_robustness_contract_digest,
    cpu_robustness_run_dir,
    cpu_robustness_specs,
    load_cpu_robustness_dbas_calibration,
    load_revision_config,
    revision_roots,
)
from src.evaluation.seed_aggregation import (
    summarize_across_training_seeds,
    summarize_by_seed,
)
from src.evaluation.sweep import (
    evaluate_run_dir,
    tag_episode_rows,
    write_sweep_csv,
)
from src.parallel import TaskSpec

CPU_ROBUSTNESS_METRICS = (
    "social_welfare",
    "raw_rsu_reward",
    "task_success_rate",
    "perception_quality_total",
    "task_success_rate_high",
    "bandwidth_nonadmitted_ratio",
)


def _set_dotted(config: dict[str, Any], dotted: str, value: Any) -> None:
    node = config
    parts = dotted.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _materialized_configs(
    cfg: dict[str, Any]
) -> dict[tuple[str, int], dict[str, Any]]:
    eval_seeds = tuple(cfg["seeds"][cfg["cpu_robustness"]["validation_seed_bank"]])
    return {
        (spec.algorithm, int(spec.train_seed)): orch.materialize_config(
            spec.to_worker_dict(), eval_seeds
        )
        for spec in cpu_robustness_specs(cfg)
    }


def _dbas_config(cfg: dict[str, Any], *, required: bool) -> dict[str, Any]:
    spec = next(
        spec for spec in cpu_robustness_specs(cfg) if spec.algorithm == "dbas"
    )
    validation_seeds = tuple(
        cfg["seeds"][cfg["cpu_robustness"]["validation_seed_bank"]]
    )
    config = orch.materialize_config(spec.to_worker_dict(), validation_seeds)
    for dotted, value in load_cpu_robustness_dbas_calibration(
        cfg, required=required
    ).items():
        _set_dotted(config, dotted, value)
    return config


def _fixed_cpu_override(band: tuple[float, float]) -> dict[str, Any]:
    return {
        "cpu_cycles_per_s_range": tuple(float(value) for value in band),
        "cpu_cycles_per_s_regimes": None,
        "cpu_regime_probabilities": None,
    }


def _mean(rows: list[dict[str, object]], key: str) -> float:
    values = [
        float(row[key]) for row in rows
        if key in row and math.isfinite(float(row[key]))
    ]
    return float(np.mean(values)) if values else float("nan")


def calibrate_dbas(cfg: dict[str, Any]) -> None:
    contract = cfg["cpu_robustness"]
    ranges = tuple(contract["cpu_capacity_ranges_cycles_per_s"])
    seeds = tuple(cfg["seeds"][contract["validation_seed_bank"]])
    base = _dbas_config(cfg, required=False)
    price_grid = tuple(float(value) for value in cfg["baseline_calibration"]["dbas_price_grid"])
    p_max = float(base["scenario"]["pricing_budget_max"])
    candidate_rows: list[dict[str, object]] = []
    candidate_scores: list[dict[str, float]] = []
    for price in price_grid:
        config = copy.deepcopy(base)
        config["algorithm"]["fixed_price_fraction"] = price / p_max
        band_rows: list[dict[str, object]] = []
        for band_index, band in enumerate(ranges):
            rows = evaluate_run_dir(
                "dbas",
                Path("."),
                seeds,
                dbas_config=config,
                scenario_override=_fixed_cpu_override(band),
            )
            row = {
                "price": price,
                "fixed_price_fraction": price / p_max,
                "cpu_band_index": band_index,
                "cpu_range_low_cycles_per_s": band[0],
                "cpu_range_high_cycles_per_s": band[1],
                "social_welfare": _mean(rows, "social_welfare"),
                "raw_rsu_reward": _mean(rows, "raw_rsu_reward"),
                "task_success_rate": _mean(rows, "task_success_rate"),
                "selected_vehicle_count": _mean(rows, "selected_vehicle_count"),
            }
            candidate_rows.append(row)
            band_rows.append(row)
        valid = all(
            float(row["task_success_rate"]) >= 0.75
            and float(row["selected_vehicle_count"]) > 0.0
            for row in band_rows
        )
        candidate_scores.append({
            "price": price,
            "fixed_price_fraction": price / p_max,
            "valid": float(valid),
            "minimum_social_welfare": min(float(row["social_welfare"]) for row in band_rows),
            "macro_social_welfare": _mean(band_rows, "social_welfare"),
            "macro_rsu_utility": _mean(band_rows, "raw_rsu_reward"),
        })
    valid_scores = [row for row in candidate_scores if bool(row["valid"])]
    if not valid_scores:
        raise ValueError("no CPU-robust DBAS price satisfies the per-band service gate")
    selected = max(
        valid_scores,
        key=lambda row: (
            row["minimum_social_welfare"],
            row["macro_social_welfare"],
            row["macro_rsu_utility"],
            -row["price"],
        ),
    )
    _, outputs = revision_roots(cfg)
    target = outputs / "cpu_robustness" / "calibration"
    target.mkdir(parents=True, exist_ok=True)
    write_sweep_csv(target / "dbas_price_cells.csv", candidate_rows)
    write_sweep_csv(target / "dbas_price_summary.csv", candidate_scores)
    selection_path = Path(contract["dbas_selection_file"])
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(yaml.safe_dump({
        "schema_version": 1,
        "selection_basis": "maximin social welfare over five validation CPU bands",
        "algorithm.fixed_price_fraction": selected["fixed_price_fraction"],
        "price": selected["price"],
        "validation_seed_count": len(seeds),
        "validation_grid_digest": config_digest(ranges),
        "calibration_contract_digest": cpu_robustness_contract_digest(cfg),
        "minimum_social_welfare": selected["minimum_social_welfare"],
        "macro_social_welfare": selected["macro_social_welfare"],
        "macro_rsu_utility": selected["macro_rsu_utility"],
    }, sort_keys=False), encoding="utf-8")
    print(f"wrote CPU-robust DBAS calibration to {selection_path}")


def train(
    cfg: dict[str, Any], *, dry_run: bool,
    force: bool, retrain_stale: bool, max_workers: int | None,
) -> int:
    contract = cfg["cpu_robustness"]
    eval_seeds = tuple(cfg["seeds"][contract["validation_seed_bank"]])
    specs = [
        spec for spec in cpu_robustness_specs(cfg)
        if spec.algorithm != "dbas"
    ]
    artifacts, outputs = revision_roots(cfg)
    output_root = artifacts / "cpu_robustness"
    plan = orch.build_train_plan(
        specs, output_root, eval_seeds,
        force=force, retrain_stale=retrain_stale,
    )
    if dry_run:
        orch.report_train_plan(plan)
        return 0
    blocked = [entry for entry in plan if entry.action == "blocked"]
    if blocked:
        orch.report_train_plan(plan)
        print("error: stale CPU robustness artifacts require --retrain-stale", file=sys.stderr)
        return 1
    pending = [entry for entry in plan if entry.action in ("train", "retrain")]
    run_id = orch.new_run_id()
    for entry in pending:
        if entry.run_dir.exists() and entry.action == "retrain":
            shutil.rmtree(entry.run_dir)
    sb3_slots = int(cfg["execution"]["sb3_cpu_slots_per_task"])
    tasks = []
    for entry in pending:
        materialized = orch.materialize_config(entry.spec.to_worker_dict(), eval_seeds)
        tasks.append(TaskSpec(
            func=orch.train_worker,
            args=(
                entry.spec.to_worker_dict(), eval_seeds, str(output_root),
                "scheduled",
            ),
            label=f"cpu-robustness/{entry.spec.run_name}",
            cpus_needed=orch.training_task_cpu_slots(
                materialized, sb3_cpu_slots=sb3_slots
            ),
            vram_needed_mb=orch.MODEL_TASK_VRAM_MB,
            vram_reserve_mb=orch.MODEL_TASK_VRAM_MB,
        ))
    manifest = outputs / "manifests" / f"cpu_robustness_{run_id}.jsonl"
    if tasks:
        orch.run_tasks_with_manifest(
            tasks,
            parallel=True,
            max_workers=(max_workers or int(cfg["execution"]["max_workers"])),
            fail_fast=True,
            manifest=manifest,
            on_success=lambda result: orch.write_training_record(result, run_id=run_id),
        )
    print(f"CPU robustness training complete; manifest: {manifest}")
    return 0


def _evaluate_rows(
    cfg: dict[str, Any], *, eval_seed_bank: str
) -> list[dict[str, object]]:
    contract = cfg["cpu_robustness"]
    eval_seeds = tuple(cfg["seeds"][eval_seed_bank])
    specs = cpu_robustness_specs(cfg)
    expected = _materialized_configs(cfg)
    dbas_config = _dbas_config(cfg, required=True)
    rows: list[dict[str, object]] = []
    for band_index, band in enumerate(contract["cpu_capacity_ranges_cycles_per_s"]):
        override = _fixed_cpu_override(band)
        display = (float(band[0]) + float(band[1])) / 2e9
        for spec in specs:
            if spec.algorithm == "dbas":
                effective = copy.deepcopy(dbas_config)
                effective["scenario"].update(override)
                trial = evaluate_run_dir(
                    "dbas", Path("."), eval_seeds,
                    dbas_config=dbas_config,
                    scenario_override=override,
                    expected_config=effective,
                )
            else:
                expected_config = expected[(spec.algorithm, int(spec.train_seed))]
                trial = evaluate_run_dir(
                    spec.algorithm,
                    cpu_robustness_run_dir(spec.algorithm, spec.train_seed, cfg),
                    eval_seeds,
                    scenario_override=override,
                    expected_config=expected_config,
                )
            tag_episode_rows(
                trial,
                algorithm_name=spec.algorithm,
                train_seed=spec.train_seed,
                scenario_overrides=override,
                sweep_param="cpu_capacity_gcycles_per_s",
                sweep_value=display,
            )
            for row in trial:
                row["cpu_band_index"] = band_index
                row["cpu_range_low_cycles_per_s"] = float(band[0])
                row["cpu_range_high_cycles_per_s"] = float(band[1])
                row["analysis_kind"] = "cpu_domain_randomized_policy_evaluation"
                row["cpu_robustness_protocol_version"] = int(contract["protocol_version"])
            rows.extend(trial)
    return rows


def _write_evaluation(
    cfg: dict[str, Any]
) -> tuple[Path, list[dict[str, object]], list[dict[str, object]]]:
    contract = cfg["cpu_robustness"]
    seed_bank = contract["evaluation_seed_bank"]
    rows = _evaluate_rows(cfg, eval_seed_bank=seed_bank)
    seed_summary = summarize_by_seed(
        rows,
        group_fields=("algorithm", "sweep_value"),
        metrics=CPU_ROBUSTNESS_METRICS,
    )
    summary = summarize_across_training_seeds(
        seed_summary,
        group_fields=("algorithm", "sweep_value"),
        metrics=CPU_ROBUSTNESS_METRICS,
    )
    _, outputs = revision_roots(cfg)
    target = outputs / "cpu_robustness" / "formal"
    target.mkdir(parents=True, exist_ok=True)
    evaluation_contract = {
        "revision_id": cfg["revision_id"],
        "revision_schema_version": int(cfg["schema_version"]),
        "cpu_robustness_protocol_version": int(contract["protocol_version"]),
        "experiment": "cpu_robustness_formal",
        "training_seed_bank": contract["training_seed_bank"],
        "training_seeds": list(
            cfg["seeds"][contract["training_seed_bank"]]
        ),
        "evaluation_seed_bank": seed_bank,
        "evaluation_seeds": list(cfg["seeds"][seed_bank]),
        "cpu_capacity_ranges_cycles_per_s": [list(value) for value in contract["cpu_capacity_ranges_cycles_per_s"]],
        "training_distribution": "uniform_episode_level_cpu_regime",
        "checkpoint_selection": contract["checkpoint_selection"],
        "aggregation_unit": "training_seed_mean_over_evaluation_scenarios",
    }
    write_sweep_csv(target / "episodes.csv", rows)
    write_sweep_csv(target / "seed_summary.csv", seed_summary)
    write_sweep_csv(target / "summary.csv", summary)
    write_evaluation_contract(target, evaluation_contract)
    return target, rows, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("phase", choices=("plan", "calibrate", "train", "evaluate"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retrain-stale", action="store_true")
    parser.add_argument("--max-workers", type=int)
    args = parser.parse_args(argv)
    cfg = load_revision_config()
    if args.phase in ("plan", "train"):
        return train(cfg, dry_run=args.phase == "plan" or args.dry_run,
                     force=args.force, retrain_stale=args.retrain_stale,
                     max_workers=args.max_workers)
    if args.phase == "calibrate":
        if not args.dry_run:
            calibrate_dbas(cfg)
        return 0
    if args.dry_run:
        print("CPU robustness evaluation plan: selected checkpoints required")
        return 0
    _write_evaluation(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
