"""Plan, train, and evaluate the configured headline experiments.

Exactness label: implementation convenience only.

Artifacts are reused only after configuration and selected-checkpoint identity
checks. Stale or partial runs require an explicit replacement option.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys

from src.evaluation import orchestration as orch
from src.evaluation.provenance import (
    write_evaluation_contract,
)
from src.evaluation.revision import (
    CORE_ALGORITHMS,
    formal_algorithm_config,
    headline_specs,
    load_revision_config,
    revision_roots,
)
from src.evaluation.sweep import (
    aggregate_sweep_rows,
    evaluate_run_dir,
    tag_episode_rows,
    write_sweep_csv,
)
from src.parallel import TaskSpec


def _train_specs(
    *,
    dry_run: bool,
    force: bool,
    retrain_stale: bool,
    refresh_isolated_timing: bool,
    max_workers: int | None = None,
) -> int:
    cfg = load_revision_config()
    worker_limit = int(
        max_workers
        if max_workers is not None
        else cfg.get("execution", {}).get("max_workers", 8)
    )
    sb3_cpu_slots = int(cfg["execution"]["sb3_cpu_slots_per_task"])
    artifacts_root, outputs_root = revision_roots(cfg)
    output_root = artifacts_root / "headline"
    eval_seeds = tuple(cfg["seeds"]["validation"])
    try:
        specs = [spec for spec in headline_specs(cfg) if spec.algorithm != "dbas"]
    except ValueError as exc:
        print(f"headline configuration error: {exc}")
        return 1
    plan = orch.build_train_plan(
        specs, output_root, eval_seeds,
        force=force, retrain_stale=retrain_stale,
    )
    isolated_seeds = set(cfg["seeds"]["isolated_timing_train"])

    def is_isolated(entry: orch.TrainPlanEntry) -> bool:
        return (
            entry.spec.algorithm in CORE_ALGORITHMS
            and entry.spec.train_seed in isolated_seeds
        )

    def has_isolated_record(entry: orch.TrainPlanEntry) -> bool:
        try:
            record = json.loads(
                (entry.run_dir / "training_record.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError, TypeError):
            return False
        return record.get("execution_mode") == "isolated"

    mistimed = [
        entry for entry in plan
        if is_isolated(entry) and entry.freshness == "fresh"
        and not has_isolated_record(entry)
    ]
    if dry_run:
        orch.report_train_plan(plan)
        print(f"revision training_max_workers: {worker_limit}")
        print(f"SB3 CPU admission slots/task: {sb3_cpu_slots}")
        print(
            "isolated timing runs: "
            f"{sum(is_isolated(entry) for entry in plan)} "
            f"(non-isolated fresh records: {len(mistimed)})"
        )
        if mistimed and not (refresh_isolated_timing or force):
            print("  use --refresh-isolated-timing to replace only those records")
        return 0
    if mistimed and not (refresh_isolated_timing or force):
        print(
            "error: the common timing seed has fresh checkpoints but was not "
            "trained in isolation; pass --refresh-isolated-timing to explicitly "
            "replace only the three timing runs",
            file=sys.stderr,
        )
        for entry in mistimed:
            print(f"  {entry.run_dir}", file=sys.stderr)
        return 1
    blocked = [
        entry for entry in plan
        if entry.action == "blocked"
        and not (refresh_isolated_timing and is_isolated(entry))
    ]
    if blocked:
        orch.report_train_plan(plan)
        print(
            "error: stale/partial revision artifacts require --retrain-stale; "
            "nothing was changed",
            file=sys.stderr,
        )
        return 1

    pending = [
        entry for entry in plan
        if entry.action in ("train", "retrain")
        or (refresh_isolated_timing and is_isolated(entry))
    ]
    run_id = orch.new_run_id()
    manifest = outputs_root / "manifests" / f"headline_{run_id}.jsonl"
    for entry in pending:
        if entry.run_dir.exists() and (
            entry.action == "retrain"
            or (refresh_isolated_timing and is_isolated(entry))
        ):
            shutil.rmtree(entry.run_dir)
    isolated_tasks: list[TaskSpec] = []
    scheduled_tasks: list[TaskSpec] = []
    for entry in pending:
        materialized = orch.materialize_config(
            entry.spec.to_worker_dict(), eval_seeds
        )
        isolated = is_isolated(entry)
        task = (
            TaskSpec(
                func=orch.train_worker,
                args=(
                    entry.spec.to_worker_dict(),
                    eval_seeds,
                    str(output_root),
                    "isolated" if isolated else "scheduled",
                ),
                label=f"headline/{entry.spec.run_name}",
                cpus_needed=orch.training_task_cpu_slots(
                    materialized, sb3_cpu_slots=sb3_cpu_slots
                ),
                vram_needed_mb=orch.MODEL_TASK_VRAM_MB,
                vram_reserve_mb=orch.MODEL_TASK_VRAM_MB,
            )
        )
        (isolated_tasks if isolated else scheduled_tasks).append(task)

    def run_phase(tasks: list[TaskSpec], *, isolated: bool) -> None:
        if not tasks:
            return
        # Even with one worker, use the scheduler process for isolated timing:
        # this keeps CUDA assignment and worker-local hardware capture identical
        # to normal scheduled jobs while preventing all sibling load.
        phase_workers = 1 if isolated else worker_limit
        orch.run_tasks_with_manifest(
            tasks,
            parallel=isolated or phase_workers > 1,
            max_workers=phase_workers,
            fail_fast=True,
            manifest=manifest,
            on_success=lambda result: orch.write_training_record(
                result, run_id=run_id
            ),
        )

    if isolated_tasks or scheduled_tasks:
        try:
            run_phase(isolated_tasks, isolated=True)
            run_phase(scheduled_tasks, isolated=False)
        except (Exception, SystemExit) as exc:
            print(f"error: headline training failed: {exc}", file=sys.stderr)
            return 1
    print(f"headline training complete; manifest: {manifest}")
    return 0


def _evaluate_headline() -> int:
    cfg = load_revision_config()
    artifacts_root, outputs_root = revision_roots(cfg)
    output_root = artifacts_root / "headline"
    eval_seeds = tuple(cfg["seeds"]["test"])
    training_eval_seeds = tuple(cfg["seeds"]["validation"])
    specs = headline_specs(cfg)
    missing = [
        spec.run_dir(output_root)
        for spec in specs
        if spec.algorithm != "dbas" and not (spec.run_dir(output_root) / "summary.json").exists()
    ]
    if missing:
        print("error: headline checkpoints are incomplete:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 1

    rows: list[dict[str, object]] = []
    for spec in specs:
        if spec.algorithm == "dbas":
            dbas_config = formal_algorithm_config("dbas", cfg)
            algorithm_rows = evaluate_run_dir(
                "dbas", output_root, eval_seeds,
                dbas_config=dbas_config,
                scenario_override=spec.scenario_overrides,
                expected_config=dbas_config,
            )
        else:
            expected_config = orch.materialize_config(
                spec.to_worker_dict(), training_eval_seeds
            )
            algorithm_rows = evaluate_run_dir(
                spec.algorithm, spec.run_dir(output_root), eval_seeds,
                expected_config=expected_config,
            )
        tag_episode_rows(
            algorithm_rows,
            algorithm_name=spec.algorithm,
            train_seed=spec.train_seed,
            scenario_overrides=spec.scenario_overrides,
        )
        rows.extend(algorithm_rows)

    aggregates = aggregate_sweep_rows(
        rows, group_fields=("algorithm",), uncertainty="seed"
    )
    target = outputs_root / "headline"
    target.mkdir(parents=True, exist_ok=True)
    contract = {
        "revision_id": cfg["revision_id"],
        "revision_schema_version": int(cfg["schema_version"]),
        "experiment": "headline",
        "training_seeds": list(cfg["seeds"]["headline_train"]),
        "evaluation_seed_bank": "test",
        "evaluation_seeds": list(eval_seeds),
        "algorithms": list(cfg["scalability"]["algorithms"]),
        "joint_learning_implementation_versions": dict(
            cfg["joint_learning_baselines"]["implementation_versions"]
        ),
        "aggregation_unit": "training_seed_mean_over_evaluation_scenarios",
    }
    write_sweep_csv(target / "episodes.csv", rows)
    write_sweep_csv(target / "summary.csv", aggregates)
    write_evaluation_contract(target, contract)
    print(f"wrote headline data to {target}")
    return 0


def _print_plan() -> int:
    cfg = load_revision_config()
    artifacts_root, outputs_root = revision_roots(cfg)
    specs = headline_specs(cfg)
    counts: dict[str, int] = {}
    for spec in specs:
        counts[spec.algorithm] = counts.get(spec.algorithm, 0) + 1
    print(f"revision_id={cfg['revision_id']}")
    print(f"artifacts_root={artifacts_root}")
    print(f"outputs_root={outputs_root}")
    print(f"validation_seeds={list(cfg['seeds']['validation'])}")
    print(f"test_seeds={list(cfg['seeds']['test'])}")
    print(f"isolated_timing_train={list(cfg['seeds']['isolated_timing_train'])}")
    print("headline independent runs:")
    for algorithm, count in counts.items():
        suffix = " (stateless evaluation)" if algorithm == "dbas" else ""
        print(f"  {algorithm}: {count}{suffix}")
    print("next: train-headline -> evaluate-headline -> pruning-sensitivity")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "phase",
        choices=("plan", "train-headline", "evaluate-headline", "pruning-sensitivity"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retrain-stale", action="store_true")
    parser.add_argument(
        "--refresh-isolated-timing",
        action="store_true",
        help=("Explicitly replace the common proposed/JDLS/MAPPO timing-seed "
              "runs and execute them one at a time"),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Parallel training tasks (default: revision execution.max_workers)",
    )
    args = parser.parse_args(argv)
    if args.max_workers is not None and args.max_workers < 1:
        parser.error("--max-workers must be positive")
    if args.phase == "plan":
        return _print_plan()
    if args.phase == "train-headline":
        return _train_specs(
            dry_run=args.dry_run,
            force=args.force,
            retrain_stale=args.retrain_stale,
            refresh_isolated_timing=args.refresh_isolated_timing,
            max_workers=args.max_workers,
        )
    if args.phase == "evaluate-headline":
        if args.dry_run:
            cfg = load_revision_config()
            missing = [path for path in (
                spec.run_dir(revision_roots(cfg)[0] / "headline")
                for spec in headline_specs(cfg)
                if spec.algorithm != "dbas"
            ) if not path.exists()]
            print(f"evaluate-headline: missing_checkpoints={len(missing)}")
            return 0
        return _evaluate_headline()
    from scripts.run_pruning_sensitivity import main as pruning_main
    return pruning_main(["plan"] if args.dry_run else ["evaluate"])


if __name__ == "__main__":
    raise SystemExit(main())
