"""Train and evaluate the controlled five-arm component ablation.

All five arms, including ``full``, use dedicated matched training seeds and are
materialized from the same unified-core TE-PPO configuration.  The
fixed-price arm is trained only after its matched full ablation run has
produced validation rows from which the static price is calibrated.

Exactness label: implementation convenience only (experiment orchestration).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from src.evaluation import orchestration as orch
from src.evaluation.ablation import (
    ABLATION_ARMS,
    ABLATION_CONDITIONAL_RATIO_METRIC,
    ABLATION_ENGINES,
    ABLATION_UNCONDITIONAL_METRICS,
    ablation_run_dir,
    build_ablation_config,
    calibrated_price_from_run,
    conditional_ratio_summary,
    paired_ablation_delta_summary,
    paired_ablation_ratio_delta_summary,
    validate_controlled_ablation,
)
from src.evaluation.revision import (
    formal_algorithm_config,
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
from src.rl.train import train_from_config


def _train_ablation_job(
    *,
    config: dict,
    output_root: str,
    arm: str,
    seed: int,
) -> dict[str, object]:
    """Train one materialized ablation arm in a scheduler worker."""
    from src.rl.run_metadata import collect_hardware_record

    summary = train_from_config(
        config,
        output_root=Path(output_root),
    )
    digest = orch.config_digest(config)
    if digest is not None:
        (Path(summary["run_dir"]) / orch.TRAIN_FINGERPRINT_FILE).write_text(
            digest, encoding="utf-8"
        )
    return {
        "run_dir": summary["run_dir"],
        "run_name": config["artifacts"]["run_name"],
        "algorithm": ABLATION_ENGINES[arm],
        "train_seed": seed,
        "config_digest": digest,
        "purpose": "ablation",
        "total_timesteps": config["algorithm"]["total_timesteps"],
        "execution_mode": "scheduled",
        "hardware": collect_hardware_record(),
    }


def _materialize(
    seed: int,
    cfg: dict,
    *,
    base_config: dict,
    fixed_price: float,
) -> dict[str, dict]:
    configs = {
        arm: build_ablation_config(
            arm,
            seed,
            fixed_price=fixed_price if arm == "no_dynamic_pricing" else None,
            revision_config=cfg,
            base_config=base_config,
        )
        for arm in ABLATION_ARMS
    }
    validate_controlled_ablation(configs)
    return configs


def _freshness(run_dir: Path, config: dict) -> str:
    if not run_dir.exists():
        return "missing"
    if not (run_dir / "summary.json").exists():
        return "partial"
    expected = orch.config_digest(config)
    return orch.train_freshness(
        run_dir, expected, expected_config=config
    )


def summarize_ablation_rows(
    rows: list[dict[str, object]],
    *,
    train_seeds: tuple[int, ...],
    eval_seeds: tuple[int, ...],
) -> list[dict[str, object]]:
    """Build paper-facing summaries while preserving conditional N/A values."""

    summary = aggregate_sweep_rows(
        rows,
        group_fields=("ablation_arm", "ablation_engine"),
        metrics=ABLATION_UNCONDITIONAL_METRICS,
        uncertainty="seed",
    )
    for row in summary:
        arm = str(row["ablation_arm"])
        ratio = conditional_ratio_summary(
            rows,
            arm=arm,
            numerator="energy_total_sum",
            denominator="selected_vehicle_count",
            train_seeds=train_seeds,
            eval_seeds=eval_seeds,
        )
        for field, value in ratio.items():
            row[f"{ABLATION_CONDITIONAL_RATIO_METRIC}_{field}"] = value

        for metric in ABLATION_UNCONDITIONAL_METRICS:
            delta = paired_ablation_delta_summary(
                rows,
                arm=arm,
                metric=metric,
                train_seeds=train_seeds,
                eval_seeds=eval_seeds,
            )
            prefix = f"{metric}_change_vs_full_pct"
            row[prefix] = delta["mean"]
            row[f"{prefix}_std"] = delta["std"]
            row[f"{prefix}_ci95"] = delta["ci95"]
            row[f"{prefix}_paired_permutation_p"] = delta[
                "paired_permutation_p"
            ]
            row[f"{prefix}_pairs"] = delta["pairs"]

        ratio_delta = paired_ablation_ratio_delta_summary(
            rows,
            arm=arm,
            numerator="energy_total_sum",
            denominator="selected_vehicle_count",
            train_seeds=train_seeds,
            eval_seeds=eval_seeds,
        )
        prefix = f"{ABLATION_CONDITIONAL_RATIO_METRIC}_change_vs_full_pct"
        row[prefix] = ratio_delta["mean"]
        row[f"{prefix}_std"] = ratio_delta["std"]
        row[f"{prefix}_ci95"] = ratio_delta["ci95"]
        row[f"{prefix}_paired_permutation_p"] = ratio_delta[
            "paired_permutation_p"
        ]
        row[f"{prefix}_pairs"] = ratio_delta["pairs"]
        row[f"{prefix}_status"] = ratio_delta["status"]
    return summary


def train(
    *,
    dry_run: bool,
    retrain_stale: bool,
    force: bool,
    max_workers: int | None = None,
) -> int:
    cfg = load_revision_config()
    sb3_cpu_slots = int(cfg["execution"]["sb3_cpu_slots_per_task"])
    artifacts_root, outputs_root = revision_roots(cfg)
    output_root = artifacts_root / "ablation"
    seeds = tuple(cfg["seeds"]["ablation_train"])
    try:
        base_config = formal_algorithm_config("proposed", cfg)
    except ValueError as exc:
        print(f"ablation configuration error: {exc}")
        return 1

    # Phase 1: train a dedicated full arm for every matched seed.  Using a
    # headline checkpoint here would silently compare differently tuned and
    # differently purposed runs.
    full_jobs: list[tuple[str, int, dict, Path, str]] = []
    for seed in seeds:
        config = build_ablation_config(
            "full", seed, revision_config=cfg, base_config=base_config
        )
        run_dir = ablation_run_dir("full", seed, cfg)
        full_jobs.append(("full", seed, config, run_dir, _freshness(run_dir, config)))

    if dry_run:
        print(f"ablation train jobs={len(seeds) * len(ABLATION_ARMS)}")
        print(f"SB3 CPU admission slots/task: {sb3_cpu_slots}")
        print("  full arms run first; fixed prices then use their validation rows")
        return 0

    worker_limit = int(
        max_workers
        if max_workers is not None
        else cfg.get("execution", {}).get("max_workers", 1)
    )

    def run_jobs(jobs: list[tuple[str, int, dict, Path, str]], *, run_id: str, manifest: Path) -> int:
        blocked = [job for job in jobs if job[4] in ("partial", "stale")]
        if blocked and not (retrain_stale or force):
            print("error: pass --retrain-stale to replace partial/stale ablation runs", file=sys.stderr)
            return 1
        pending = [job for job in jobs if force or job[4] != "fresh"]
        tasks: list[TaskSpec] = []
        for arm, seed, config, run_dir, state in pending:
            if run_dir.exists() and (force or state in ("partial", "stale")):
                shutil.rmtree(run_dir)
            tasks.append(TaskSpec(
                func=_train_ablation_job,
                kwargs={
                    "config": config,
                    "output_root": str(output_root),
                    "arm": arm,
                    "seed": seed,
                },
                label=f"ablation/{arm}/s{seed}",
                cpus_needed=orch.training_task_cpu_slots(
                    config, sb3_cpu_slots=sb3_cpu_slots
                ),
                vram_needed_mb=orch.MODEL_TASK_VRAM_MB,
                vram_reserve_mb=orch.MODEL_TASK_VRAM_MB,
            ))
        if not tasks:
            return 0
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
            print(f"error: ablation training failed: {exc}", file=sys.stderr)
            return 1
        return 0

    run_id = orch.new_run_id()
    manifest = outputs_root / "manifests" / f"ablation_{run_id}.jsonl"
    if run_jobs(full_jobs, run_id=run_id, manifest=manifest):
        return 1

    jobs: list[tuple[str, int, dict, Path, str]] = []
    for seed in seeds:
        full_dir = ablation_run_dir("full", seed, cfg)
        fixed_price = calibrated_price_from_run(full_dir)
        configs = _materialize(
            seed, cfg, base_config=base_config, fixed_price=fixed_price
        )
        for arm in ABLATION_ARMS[1:]:
            config = configs[arm]
            run_dir = ablation_run_dir(arm, seed, cfg)
            jobs.append((arm, seed, config, run_dir, _freshness(run_dir, config)))
    if run_jobs(jobs, run_id=run_id, manifest=manifest):
        return 1
    print(f"ablation training complete; workers={worker_limit}; manifest: {manifest}")
    return 0


def evaluate() -> int:
    cfg = load_revision_config()
    _, outputs_root = revision_roots(cfg)
    seeds = tuple(cfg["seeds"]["ablation_train"])
    eval_seeds = tuple(cfg["seeds"]["test"])
    missing: list[Path] = []
    for arm in ABLATION_ARMS:
        for seed in seeds:
            run_dir = ablation_run_dir(arm, seed, cfg)
            if not (run_dir / "summary.json").exists():
                missing.append(run_dir)
    if missing:
        print("error: missing ablation checkpoints:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 1

    try:
        base_config = formal_algorithm_config("proposed", cfg)
        expected_configs = {
            (arm, seed): configs[arm]
            for seed in seeds
            for configs in (_materialize(
                seed,
                cfg,
                base_config=base_config,
                fixed_price=calibrated_price_from_run(
                    ablation_run_dir("full", seed, cfg)
                ),
            ),)
            for arm in ABLATION_ARMS
        }
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: cannot validate ablation artifacts: {exc}", file=sys.stderr)
        return 1

    rows: list[dict[str, object]] = []
    for arm in ABLATION_ARMS:
        engine = ABLATION_ENGINES[arm]
        for seed in seeds:
            run_dir = ablation_run_dir(arm, seed, cfg)
            arm_rows = evaluate_run_dir(
                engine,
                run_dir,
                eval_seeds,
                expected_config=expected_configs[(arm, seed)],
            )
            tag_episode_rows(
                arm_rows, algorithm_name=engine, train_seed=seed
            )
            for row in arm_rows:
                row["ablation_arm"] = arm
                row["ablation_engine"] = engine
                row["runtime_evidence_status"] = "diagnostic_unwarmed_non_final"
                participants = float(row.get("selected_vehicle_count", 0.0))
                row["participation_rate"] = float(participants > 0.0)
                row["energy_per_participating_vehicle_j"] = (
                    float(row.get("energy_total_sum", 0.0)) / participants
                    if participants > 0.0 else float("nan")
                )
                if participants <= 0.0:
                    for metric in (
                        "energy_total_sum",
                        "total_raw_data_bits",
                        "total_transmitted_data_bits",
                        "effective_task_quality_sum",
                    ):
                        if abs(float(row.get(metric, 0.0))) > 1e-9:
                            raise ValueError(
                                f"{arm}/{metric}: zero selected vehicles but "
                                "non-zero service output"
                            )
            rows.extend(arm_rows)

    target = outputs_root / "ablation"
    target.mkdir(parents=True, exist_ok=True)
    # Preserve the expensive evaluation rows even if a downstream derived
    # statistic fails. The final validator still requires the matching summary.
    write_sweep_csv(target / "episodes.csv", rows)

    summary = summarize_ablation_rows(
        rows, train_seeds=seeds, eval_seeds=eval_seeds
    )
    write_sweep_csv(target / "summary.csv", summary)
    print(f"wrote ablation ablation outputs to {target}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "phase",
        choices=("plan", "train", "evaluate"),
        default="plan",
        nargs="?",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retrain-stale", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-workers", type=int)
    args = parser.parse_args(argv)
    if args.phase == "plan":
        return train(
            dry_run=True, retrain_stale=False, force=False,
            max_workers=args.max_workers,
        )
    if args.phase == "train":
        return train(
            dry_run=args.dry_run,
            retrain_stale=args.retrain_stale,
            force=args.force,
            max_workers=args.max_workers,
        )
    return evaluate()


if __name__ == "__main__":
    raise SystemExit(main())
