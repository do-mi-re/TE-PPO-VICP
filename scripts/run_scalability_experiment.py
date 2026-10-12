"""Train/evaluate the robustness shape-changing vehicle and task scalability grids.

Every non-default ``V`` or ``K`` cell gets its own policy; checkpoint replay
across incompatible observation/action shapes is forbidden. The default
``V=20,K=3`` cell reuses the first three canonical headline replicates.

Exactness label: implementation convenience only (experiment orchestration).
"""

from __future__ import annotations

import argparse
import copy
import shutil
import sys
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
    scalability_specs,
    timing_scalability_specs,
    train_seeds_for,
)
from src.evaluation.sweep import (
    aggregate_sweep_rows,
    evaluate_run_dir,
    tag_episode_rows,
    write_sweep_csv,
)
from src.parallel import TaskSpec


def _root(axis: str, cfg: dict) -> Path:
    return revision_roots(cfg)[0] / "scalability" / axis


def train(
    axis: str,
    *,
    dry_run: bool,
    retrain_stale: bool,
    max_workers: int | None = None,
) -> int:
    cfg = load_revision_config()
    sb3_cpu_slots = int(cfg["execution"]["sb3_cpu_slots_per_task"])
    try:
        source_specs = (
            timing_scalability_specs(cfg)
            if axis == "timing"
            else scalability_specs(axis, cfg)
        )
        specs = [spec for spec in source_specs if spec.algorithm != "dbas"]
    except FileNotFoundError as exc:
        print(f"scalability dependency: {exc}")
        return 0 if dry_run else 1
    root = _root(axis, cfg)
    eval_seeds = tuple(cfg["seeds"]["validation"])
    plan = orch.build_train_plan(
        specs, root, eval_seeds,
        force=False, retrain_stale=retrain_stale,
    )
    if dry_run:
        orch.report_train_plan(plan)
        print(f"SB3 CPU admission slots/task: {sb3_cpu_slots}")
        return 0
    if any(entry.action == "blocked" for entry in plan):
        orch.report_train_plan(plan)
        print("error: pass --retrain-stale for stale/partial scalability runs", file=sys.stderr)
        return 1
    run_id = orch.new_run_id()
    manifest = revision_roots(cfg)[1] / "manifests" / f"scale_{axis}_{run_id}.jsonl"
    pending = [entry for entry in plan if entry.action in ("train", "retrain")]
    worker_limit = int(
        max_workers
        if max_workers is not None
        else cfg.get("execution", {}).get("max_workers", 1)
    )
    tasks: list[TaskSpec] = []
    for entry in pending:
        if entry.action == "retrain":
            shutil.rmtree(entry.run_dir)
        materialized = orch.materialize_config(
            entry.spec.to_worker_dict(), eval_seeds
        )
        tasks.append(TaskSpec(
            func=orch.train_worker,
            args=(
                entry.spec.to_worker_dict(),
                eval_seeds,
                str(root),
                "scheduled",
            ),
            label=f"scale-{axis}/{entry.spec.run_name}",
            cpus_needed=orch.training_task_cpu_slots(
                materialized, sb3_cpu_slots=sb3_cpu_slots
            ),
            vram_needed_mb=orch.MODEL_TASK_VRAM_MB,
            vram_reserve_mb=orch.MODEL_TASK_VRAM_MB,
        ))
    if tasks:
        try:
            orch.run_tasks_with_manifest(
                tasks,
                parallel=worker_limit > 1,
                max_workers=worker_limit,
                fail_fast=True,
                manifest=manifest,
                on_success=lambda result: orch.write_training_record(
                    result, run_id=run_id
                ),
            )
        except (Exception, SystemExit) as exc:
            print(f"error: scalability training failed: {exc}", file=sys.stderr)
            return 1
    print(f"scalability training complete; workers={worker_limit}; manifest: {manifest}")
    return 0


def _default_specs(axis: str, cfg: dict) -> list[tuple[str, int, Path, float]]:
    default_value = 20.0 if axis == "vehicles" else 3.0
    specs = []
    for algorithm in tuple(cfg["scalability"]["algorithms"]):
        seeds = train_seeds_for(algorithm, scope="sweep", config=cfg)
        if algorithm == "dbas":
            seeds = seeds[:1]
        for seed in seeds:
            specs.append((
                algorithm, int(seed), headline_run_dir(algorithm, seed, cfg), default_value
            ))
    return specs


def evaluate(axis: str) -> int:
    if axis == "timing":
        print("timing checkpoints are evaluated by scripts/measure_inference_latency.py")
        return 0
    cfg = load_revision_config()
    root = _root(axis, cfg)
    trained_specs = scalability_specs(axis, cfg)
    missing = [
        spec.run_dir(root) for spec in trained_specs
        if spec.algorithm != "dbas" and not (spec.run_dir(root) / "summary.json").exists()
    ]
    default_specs = _default_specs(axis, cfg)
    missing.extend(
        path for algorithm, _seed, path, _value in default_specs
        if algorithm != "dbas" and not (path / "summary.json").exists()
    )
    if missing:
        print(f"error: {len(missing)} scalability checkpoints are missing", file=sys.stderr)
        return 1
    seed_bank = str(cfg["scalability"]["evaluation_seed_bank"])
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

    def collect(algorithm: str, seed: int, run_dir: Path, value: float,
                scenario_override: dict | None = None,
                expected_config: dict | None = None) -> None:
        if algorithm == "dbas":
            dbas_config = formal_algorithm_config("dbas", cfg)
            effective_config = copy.deepcopy(dbas_config)
            if scenario_override:
                effective_config.setdefault("scenario", {}).update(scenario_override)
            trial = evaluate_run_dir(
                "dbas", root, eval_seeds,
                dbas_config=dbas_config,
                scenario_override=scenario_override,
                expected_config=effective_config,
            )
        else:
            trial = evaluate_run_dir(
                algorithm, run_dir, eval_seeds, expected_config=expected_config
            )
        tag_episode_rows(
            trial, algorithm_name=algorithm, train_seed=seed,
            scenario_overrides=scenario_override,
            sweep_param="num_vehicles" if axis == "vehicles" else "num_tasks",
            sweep_value=value,
        )
        for row in trial:
            row["runtime_evidence_status"] = "diagnostic_unwarmed_non_final"
        rows.extend(trial)

    for algorithm, seed, run_dir, value in default_specs:
        collect(
            algorithm,
            seed,
            run_dir,
            value,
            expected_config=headline_expected.get((algorithm, seed)),
        )
    for spec in trained_specs:
        collect(
            spec.algorithm, spec.train_seed, spec.run_dir(root),
            float(spec.display_value),
            spec.scenario_overrides if spec.algorithm == "dbas" else None,
            None if spec.algorithm == "dbas" else orch.materialize_config(
                spec.to_worker_dict(), training_eval_seeds
            ),
        )

    summary = aggregate_sweep_rows(rows, uncertainty="seed")
    target = revision_roots(cfg)[1] / "scalability" / axis
    target.mkdir(parents=True, exist_ok=True)
    contract = {
        "revision_id": cfg["revision_id"],
        "revision_schema_version": int(cfg["schema_version"]),
        "experiment": f"scalability_{axis}",
        "evaluation_seed_bank": seed_bank,
        "evaluation_seeds": list(eval_seeds),
        "training_seeds": list(cfg["seeds"]["sweep_train"]),
        "algorithms": list(cfg["scalability"]["algorithms"]),
        "grid": list(
            cfg["scalability"][
                "vehicle_grid" if axis == "vehicles" else "task_grid"
            ]
        ),
        "aggregation_unit": "training_seed_mean_over_evaluation_scenarios",
    }
    write_sweep_csv(target / "episodes.csv", rows)
    write_sweep_csv(target / "summary.csv", summary)
    write_evaluation_contract(target, contract)
    print(f"wrote scalability {axis} outputs to {target}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("phase", choices=("plan", "train", "evaluate"))
    parser.add_argument("--axis", choices=("vehicles", "tasks", "timing"), required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retrain-stale", action="store_true")
    parser.add_argument("--max-workers", type=int)
    args = parser.parse_args(argv)
    if args.phase == "plan":
        return train(
            args.axis, dry_run=True, retrain_stale=False,
            max_workers=args.max_workers,
        )
    if args.phase == "train":
        return train(
            args.axis, dry_run=args.dry_run, retrain_stale=args.retrain_stale,
            max_workers=args.max_workers,
        )
    return evaluate(args.axis)


if __name__ == "__main__":
    raise SystemExit(main())
