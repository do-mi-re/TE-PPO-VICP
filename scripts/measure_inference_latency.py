"""Measure complete scheduling-cycle latency using selected checkpoints.

Every ``(V, K)`` cell is evaluated with a policy trained for that exact
observation/action shape. The script fails closed when a checkpoint is
missing; randomly initialized shape-only networks are not paper evidence.
Candidate pruning occurs during ``reset`` but is included explicitly in the
scheduling-cycle total.

Exactness label: implementation convenience only (isolated benchmarking and
report generation; no model or training semantics are changed).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np

from src.evaluation import orchestration as orch
from src.evaluation.ablation import (
    ABLATION_ARMS,
    ABLATION_ENGINES,
    ablation_run_dir,
    build_ablation_config,
    calibrated_price_from_run,
)
from src.evaluation.provenance import validate_run_artifact
from src.evaluation.revision import (
    formal_algorithm_config,
    headline_run_dir,
    headline_specs,
    load_revision_config,
    revision_roots,
    timing_scalability_specs,
)
from src.evaluation.sweep import evaluate_run_dir
from src.rl.checkpoint_selection import selected_model_name
from src.rl.run_metadata import collect_hardware_record

SCENARIO_SEEDS = (40_000, 40_001, 40_002, 40_003)


def checkpoint_grid(cfg: dict) -> dict[tuple[int, int], Path]:
    """Return the exact trained checkpoint expected for every timing cell."""

    artifacts, _ = revision_roots(cfg)
    seed = int(cfg["seeds"]["sweep_train"][0])
    paths = {(20, 3): headline_run_dir("proposed", seed, cfg)}
    root = artifacts / "scalability" / "timing"
    for spec in timing_scalability_specs(cfg):
        overrides = spec.scenario_overrides
        paths[(int(overrides["num_vehicles"]), len(overrides["urgency_profiles"]))] = (
            spec.run_dir(root)
        )
    return paths


def checkpoint_expected_configs(cfg: dict) -> dict[tuple[int, int], dict]:
    """Materialize the current training config expected for every timing cell."""

    validation_seeds = tuple(cfg["seeds"]["validation"])
    seed = int(cfg["seeds"]["sweep_train"][0])
    headline = next(
        spec
        for spec in headline_specs(cfg)
        if spec.algorithm == "proposed" and spec.train_seed == seed
    )
    configs = {
        (20, 3): orch.materialize_config(
            headline.to_worker_dict(), validation_seeds
        )
    }
    for spec in timing_scalability_specs(cfg):
        overrides = spec.scenario_overrides
        key = (int(overrides["num_vehicles"]), len(overrides["urgency_profiles"]))
        configs[key] = orch.materialize_config(
            spec.to_worker_dict(), validation_seeds
        )
    return configs


def _median(rows: list[dict[str, object]], key: str) -> float:
    return float(statistics.median(float(row[key]) for row in rows))


def _optional_median(rows: list[dict[str, object]], key: str) -> float | str:
    values = [
        float(row[key])
        for row in rows
        if row.get(key) not in (None, "") and math.isfinite(float(row[key]))
    ]
    return float(statistics.median(values)) if values else ""


def _timing_rows(
    algorithm: str,
    run_dir: Path,
    *,
    reps: int,
    warmup: int,
    expected_config: dict,
) -> list[dict[str, object]]:
    """Load a checkpoint once, warm it in-process, and retain measured rows."""

    count = warmup + reps
    seeds = tuple(
        SCENARIO_SEEDS[index % len(SCENARIO_SEEDS)] for index in range(count)
    )
    rows = evaluate_run_dir(
        algorithm, run_dir, seeds, expected_config=expected_config
    )
    return rows[warmup:]


def measure_checkpoint(
    run_dir: Path,
    *,
    num_vehicles: int,
    num_tasks: int,
    reps: int,
    warmup: int,
    expected_config: dict,
) -> dict[str, object]:
    """Evaluate one exact-shape checkpoint and aggregate component timings."""

    required = (
        run_dir / "summary.json",
        run_dir / "config_snapshot.yaml",
        run_dir / selected_model_name("proposed"),
        run_dir / "selection.json",
        run_dir / orch.TRAIN_FINGERPRINT_FILE,
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"missing checkpoint evidence for V={num_vehicles}, K={num_tasks}: "
            + ", ".join(map(str, missing))
        )

    rows = _timing_rows(
        "proposed", run_dir, reps=reps, warmup=warmup,
        expected_config=expected_config,
    )
    actor_ms = _median(rows, "actor_forward_ms")
    pruning_ms = _median(rows, "candidate_pruning_time_ms")
    follower_ms = _median(rows, "follower_solve_time_ms")
    aggregation_ms = _median(rows, "aggregation_time_ms")
    env_step_ms = _median(rows, "env_step_ms")
    total_samples = [float(row["total_cycle_ms"]) for row in rows]
    total_ms = float(statistics.median(total_samples))
    other_ms = max(0.0, env_step_ms - follower_ms - aggregation_ms)
    provenance = validate_run_artifact(
        run_dir, expected_config=expected_config,
        checkpoint_name=selected_model_name("proposed"),
    )

    return {
        "num_vehicles": num_vehicles,
        "num_tasks": num_tasks,
        "benchmark_kind": "checkpoint",
        "warmup_repetitions": warmup,
        "repetitions": reps,
        "actor_forward_ms": round(actor_ms, 4),
        "candidate_pruning_ms": round(pruning_ms, 4),
        "follower_solve_ms": round(follower_ms, 4),
        "aggregation_ms": round(aggregation_ms, 4),
        "other_env_ms": round(other_ms, 4),
        "total_cycle_ms": round(total_ms, 4),
        "cycle_p95_ms": round(float(np.percentile(total_samples, 95)), 4),
        "solve_share_pct": round(100.0 * follower_ms / total_ms, 2),
        "admitted_pair_count": round(_median(rows, "candidate_admitted_pair_count")),
        "pruned_pair_count": round(_median(rows, "candidate_pruned_pair_count")),
        "run_dir": str(run_dir),
        "config_fingerprint": provenance.config_fingerprint,
        "checkpoint_sha256": provenance.checkpoint_sha256,
        "normalization_sha256": provenance.normalization_sha256,
        "selection_sha256": provenance.selection_sha256,
        "artifact_kind": "trained_checkpoint",
    }


def ablation_checkpoint_specs(
    cfg: dict,
) -> list[tuple[str, str, int, Path, dict]]:
    """Return exact engine/path/config identities for all formal ablation runs."""

    base = formal_algorithm_config("proposed", cfg)
    specs: list[tuple[str, str, int, Path, dict]] = []
    for seed in cfg["seeds"]["ablation_train"]:
        seed = int(seed)
        fixed_price = calibrated_price_from_run(ablation_run_dir("full", seed, cfg))
        for arm in ABLATION_ARMS:
            config = build_ablation_config(
                arm,
                seed,
                fixed_price=fixed_price if arm == "no_dynamic_pricing" else None,
                revision_config=cfg,
                base_config=base,
            )
            specs.append((
                arm,
                ABLATION_ENGINES[arm],
                seed,
                ablation_run_dir(arm, seed, cfg),
                config,
            ))
    return specs


def measure_ablation_checkpoint(
    arm: str,
    algorithm: str,
    train_seed: int,
    run_dir: Path,
    expected_config: dict,
    *,
    reps: int,
    warmup: int,
) -> dict[str, object]:
    """Measure one selected ablation checkpoint after an in-process warm-up."""

    provenance = validate_run_artifact(
        run_dir,
        expected_config=expected_config,
        checkpoint_name=selected_model_name(algorithm),
    )
    rows = _timing_rows(
        algorithm, run_dir, reps=reps, warmup=warmup,
        expected_config=expected_config,
    )
    wall_samples = [float(row["inference_wall_time_ms"]) for row in rows]
    pruning_samples = [
        float(row.get("candidate_pruning_time_ms", 0.0)) for row in rows
    ]
    total_samples = [
        wall + pruning
        for wall, pruning in zip(wall_samples, pruning_samples, strict=True)
    ]
    return {
        "ablation_arm": arm,
        "algorithm": algorithm,
        "train_seed": train_seed,
        "benchmark_kind": "checkpoint_warmed",
        "warmup_repetitions": warmup,
        "repetitions": reps,
        "actor_forward_ms": _optional_median(rows, "actor_forward_ms"),
        "candidate_pruning_ms": round(statistics.median(pruning_samples), 4),
        "follower_solve_ms": _optional_median(rows, "follower_solve_time_ms"),
        "aggregation_ms": _optional_median(rows, "aggregation_time_ms"),
        "inference_wall_ms": round(statistics.median(wall_samples), 4),
        "total_cycle_ms": round(statistics.median(total_samples), 4),
        "cycle_p95_ms": round(float(np.percentile(total_samples, 95)), 4),
        "run_dir": str(run_dir),
        "config_fingerprint": provenance.config_fingerprint,
        "checkpoint_sha256": provenance.checkpoint_sha256,
        "normalization_sha256": provenance.normalization_sha256,
        "selection_sha256": provenance.selection_sha256,
        "artifact_kind": "trained_checkpoint",
    }


def summarize_ablation_timing(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Aggregate per-checkpoint medians across the matched training seeds."""

    output: list[dict[str, object]] = []
    for arm in ABLATION_ARMS:
        group = [row for row in rows if row["ablation_arm"] == arm]
        if not group:
            raise ValueError(f"missing ablation timing arm: {arm}")
        record: dict[str, object] = {
            "ablation_arm": arm,
            "algorithm": group[0]["algorithm"],
            "seed_count": len(group),
            "benchmark_kind": "checkpoint_warmed_seed_summary",
        }
        for metric in ("total_cycle_ms", "cycle_p95_ms", "inference_wall_ms"):
            values = [float(row[metric]) for row in group]
            mean = statistics.mean(values)
            std = statistics.stdev(values) if len(values) > 1 else float("nan")
            record[f"{metric}_mean"] = mean
            record[f"{metric}_std"] = std
            record[f"{metric}_ci95"] = (
                1.96 * std / math.sqrt(len(values))
                if len(values) > 1 else float("nan")
            )
        output.append(record)
    return output


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def hardware_record() -> dict[str, object]:
    return collect_hardware_record()


def render_figure(rows: list[dict[str, object]], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.plotting.revision_style import (
        FIGSIZE_SINGLE_COLUMN,
        LEGEND_STYLE,
        apply_ieee_style,
        errorbar_style,
        save_figure,
        setup_axes,
    )

    apply_ieee_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_COLUMN, layout="constrained")
    setup_axes(ax)
    palette = ("#C2410C", "#1D4ED8", "#047857", "#7C3AED")
    markers = ("o", "s", "^", "D")
    linestyles = ("-", "--", "-.", ":")
    max_p95_row = max(rows, key=lambda row: float(row["cycle_p95_ms"]))
    for index, tasks in enumerate(sorted({int(row["num_tasks"]) for row in rows})):
        series = sorted(
            (row for row in rows if int(row["num_tasks"]) == tasks),
            key=lambda row: int(row["num_vehicles"]),
        )
        medians = np.asarray([float(row["total_cycle_ms"]) for row in series])
        p95 = np.asarray([float(row["cycle_p95_ms"]) for row in series])
        ax.errorbar(
            [int(row["num_vehicles"]) for row in series],
            medians,
            yerr=np.vstack((np.zeros_like(medians), np.maximum(0.0, p95 - medians))),
            marker=markers[index % len(markers)],
            linestyle=linestyles[index % len(linestyles)],
            linewidth=1.3,
            markersize=3.6,
            color=palette[index % len(palette)],
            label=f"K = {tasks}",
            **errorbar_style(palette[index % len(palette)]),
        )
    max_x = int(max_p95_row["num_vehicles"])
    max_y = float(max_p95_row["cycle_p95_ms"])
    ax.annotate(
        f"Maximum p95: {max_y:.2f} ms",
        xy=(max_x, max_y),
        xytext=(-6, 8),
        textcoords="offset points",
        ha="right",
        arrowprops={"arrowstyle": "-", "color": "#4B5563", "linewidth": 0.7},
        color="#374151",
    )
    ax.set_xticks(sorted({int(row["num_vehicles"]) for row in rows}))
    ax.set_ylim(bottom=0.0, top=max_y * 1.16)
    ax.set_xlabel("Number of Vehicles")
    ax.set_ylabel("Per-Cycle Inference Time (ms)")
    ax.legend(loc="upper left", ncol=2, **LEGEND_STYLE)
    save_figure(fig, output)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    cfg = load_revision_config()
    _, outputs = revision_roots(cfg)
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--scope", choices=("all", "scalability", "ablation"), default="all"
    )
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.reps <= 0 or args.warmup < 0:
        parser.error("--reps must be positive and --warmup non-negative")

    if args.scope in ("all", "scalability"):
        try:
            paths = checkpoint_grid(cfg)
            expected_configs = checkpoint_expected_configs(cfg)
        except FileNotFoundError as exc:
            print(f"timing dependency: {exc}")
            return 0 if args.dry_run else 1
        expected = {
            (int(v), int(k))
            for v in cfg["scalability"]["timing_vehicle_grid"]
            for k in cfg["scalability"]["timing_task_grid"]
        }
        if set(paths) != expected or set(expected_configs) != expected:
            raise RuntimeError("timing checkpoint/config map does not cover the full grid")
        if args.dry_run:
            missing = [path for path in paths.values() if not (path / "summary.json").exists()]
            print(f"timing grid: {len(paths)} checkpoint-backed cells; missing={len(missing)}")
        else:
            rows = [
                measure_checkpoint(
                    path, num_vehicles=v, num_tasks=k, reps=args.reps,
                    warmup=args.warmup, expected_config=expected_configs[(v, k)],
                )
                for (v, k), path in sorted(paths.items())
            ]
            table = outputs / "tables" / "inference_latency.csv"
            write_rows(table, rows)
            if not args.no_figure:
                render_figure(rows, outputs / "figures" / "figR_latency")
            print(f"wrote {len(rows)} checkpoint-backed timing cells to {table}")

    if args.scope in ("all", "ablation"):
        try:
            specs = ablation_checkpoint_specs(cfg)
        except FileNotFoundError as exc:
            print(f"ablation timing dependency: {exc}")
            return 0 if args.dry_run else 1
        if args.dry_run:
            missing = [path for _, _, _, path, _ in specs if not (path / "summary.json").exists()]
            print(f"ablation timing: {len(specs)} checkpoints; missing={len(missing)}")
        else:
            ablation_rows = [
                measure_ablation_checkpoint(
                    arm, algorithm, seed, path, config,
                    reps=args.reps, warmup=args.warmup,
                )
                for arm, algorithm, seed, path, config in specs
            ]
            write_rows(
                outputs / "tables" / "ablation_inference_latency.csv",
                ablation_rows,
            )
            write_rows(
                outputs / "tables" / "ablation_inference_latency_summary.csv",
                summarize_ablation_timing(ablation_rows),
            )

    if args.dry_run:
        return 0
    tables = outputs / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "hardware.json").write_text(
        json.dumps(hardware_record(), indent=2, sort_keys=True), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
