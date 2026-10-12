"""Report learning efficiency convergence, common reward, and equal-budget performance.

Social welfare remains the checkpoint-selection and headline convergence
metric.  A separate curve reports the common penalty-adjusted evaluation
reward; algorithm-specific rollout shaping rewards remain excluded.  Each
statistic is first computed per independent training seed and then summarized
across seeds.

Exactness label: implementation convenience only (statistical reporting).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np
import yaml

from src.evaluation import orchestration as orch
from src.evaluation.provenance import (
    provenance_bundle_digest,
    sha256_file,
    validate_run_artifact,
)
from src.evaluation.revision import (
    headline_specs,
    load_revision_config,
    train_seeds_for,
)
from src.evaluation.sweep import ALGORITHM_LABELS
from src.rl.checkpoint_selection import selected_model_name

OUTPUT_DIR = Path("outputs/revision_2026_v5/tables")
FIGURE_STEM = Path("outputs/revision_2026_v5/figures/figR_learning_efficiency")
REWARD_FIGURE_STEM = Path(
    "outputs/revision_2026_v5/figures/figR_reward_convergence"
)
DEFAULT_ARTIFACTS_ROOT = Path("artifacts/revision_2026_v5/headline")
DEFAULT_PATTERN = "{algo}_s{seed}_headline"
DEFAULT_ALGORITHMS = ("proposed", "jdls", "mappo", "ebas", "spis", "rdus")
THRESHOLDS = (0.80, 0.90, 0.95)
BUDGETS = (20_000, 40_000, 60_000)
FINAL_FRACTION = 0.10
REWARD_FIGURE_XLABEL = "Training Timesteps"
REWARD_FIGURE_YLABEL = "Training Reward"


def load_curve(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Load timesteps, per-evaluation welfare, and cumulative wall time."""
    with np.load(path) as data:
        if "social_welfare" not in data:
            raise KeyError(
                f"{path} has no social_welfare; shaped results are not comparable"
            )
        timesteps = np.asarray(data["timesteps"], dtype=np.int64)
        welfare = np.asarray(data["social_welfare"], dtype=np.float64).mean(axis=1)
        schema = str(data["wall_time_schema"].item()) if "wall_time_schema" in data else ""
        wall_time = (
            np.asarray(data["wall_time_s"], dtype=np.float64)
            if schema == "monotonic_training_loop_v2" and "wall_time_s" in data
            else None
        )
    return timesteps, welfare, wall_time


def load_common_reward_curve(
    path: Path,
    *,
    algorithm: str,
    expected_schema: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Load common reward under the required evaluation schema."""

    with np.load(path, allow_pickle=False) as data:
        timesteps = np.asarray(data["timesteps"], dtype=np.int64)
        welfare_shape = (
            np.asarray(data["social_welfare"]).shape
            if "social_welfare" in data
            else None
        )
        if "common_evaluation_reward" in data:
            schema = str(data["reward_schema"].item()) if "reward_schema" in data else ""
            if schema != expected_schema:
                raise ValueError(
                    f"{path}: common reward schema {schema!r} != {expected_schema!r}"
                )
            values = np.asarray(
                data["common_evaluation_reward"], dtype=np.float64
            )
            source_key = "common_evaluation_reward"
        else:
            raise KeyError(
                f"{path}: {algorithm} lacks {expected_schema}; refresh its "
                "headline training artifact"
            )
    if values.ndim != 2:
        raise ValueError(f"{path}: common reward samples must be a matrix")
    if values.shape[0] != len(timesteps):
        raise ValueError(
            f"{path}: common reward rows do not match recorded timesteps"
        )
    if welfare_shape is not None and values.shape != welfare_shape:
        raise ValueError(
            f"{path}: common reward and social-welfare evaluation grids differ"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path}: common reward samples must be finite")
    return timesteps, values.mean(axis=1), source_key


def threshold_index(curve: np.ndarray, target: float) -> int | None:
    """First index of the final contiguous run at or above ``target``."""
    reached = curve >= target
    if not reached.any() or not reached[-1]:
        return None
    index = len(reached) - 1
    while index > 0 and reached[index - 1]:
        index -= 1
    return index


def steps_to_threshold(
    timesteps: np.ndarray, curve: np.ndarray, target: float
) -> int | None:
    index = threshold_index(curve, target)
    return int(timesteps[index]) if index is not None else None


def progress_target(reference: float, final: float, fraction: float) -> float | None:
    """Return an affine progress target, or ``None`` without positive progress."""

    if not 0.0 < fraction <= 1.0:
        raise ValueError("progress fraction must lie in (0, 1]")
    if final <= reference:
        return None
    return reference + fraction * (final - reference)


def value_at_budget(
    timesteps: np.ndarray, curve: np.ndarray, budget: int
) -> float | None:
    eligible = np.nonzero(timesteps <= budget)[0]
    return float(curve[eligible[-1]]) if eligible.size else None


def normalized_aulc(timesteps: np.ndarray, curve: np.ndarray) -> float:
    """Trapezoidal learning-curve area normalized by the observed step span."""
    if len(curve) == 0:
        return float("nan")
    if len(curve) == 1 or timesteps[-1] == timesteps[0]:
        return float(curve[0])
    return float(np.trapezoid(curve, timesteps) / (timesteps[-1] - timesteps[0]))


def validate_curve_coverage(
    timesteps: np.ndarray,
    curve: np.ndarray,
    wall_time: np.ndarray | None,
    run_dir: Path,
) -> None:
    """Reject malformed or truncated curves before paper-facing aggregation."""

    if timesteps.ndim != 1 or curve.ndim != 1 or not len(timesteps):
        raise ValueError(f"{run_dir}: convergence arrays must be non-empty vectors")
    if len(timesteps) != len(curve):
        raise ValueError(f"{run_dir}: timestep/welfare length mismatch")
    if len(timesteps) > 1 and np.any(np.diff(timesteps) <= 0):
        raise ValueError(f"{run_dir}: timesteps must be strictly increasing")
    if not np.isfinite(curve).all():
        raise ValueError(f"{run_dir}: welfare curve contains non-finite values")
    if wall_time is not None:
        if wall_time.ndim != 1 or len(wall_time) != len(timesteps):
            raise ValueError(f"{run_dir}: wall-time/timestep length mismatch")
        if not np.isfinite(wall_time).all() or np.any(np.diff(wall_time) < 0):
            raise ValueError(f"{run_dir}: invalid cumulative wall-time samples")

    snapshot_path = run_dir / "config_snapshot.yaml"
    if not snapshot_path.exists():
        raise ValueError(f"{run_dir}: missing config_snapshot.yaml")
    snapshot = yaml.safe_load(snapshot_path.read_text(encoding="utf-8")) or {}
    algorithm = snapshot.get("algorithm", {})
    evaluation = snapshot.get("evaluation", {})
    total = int(algorithm.get("total_timesteps", 0))
    physical_interval = int(evaluation.get("interval_env_steps", 0))
    if total <= 0 or physical_interval <= 0:
        raise ValueError(f"{run_dir}: invalid training/evaluation budget metadata")
    if int(timesteps[-1]) < total:
        raise ValueError(
            f"{run_dir}: truncated curve ends at {int(timesteps[-1])}, "
            f"expected a forced terminal evaluation at or after {total}"
        )


def isolated_wall_time(run_dir: Path, wall_time: np.ndarray | None) -> np.ndarray | None:
    """Return wall samples only for explicitly isolated training runs."""

    if wall_time is None:
        return None
    try:
        record = json.loads(
            (run_dir / "training_record.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError, TypeError):
        return None
    return wall_time if record.get("execution_mode") == "isolated" else None


def _mean_std_ci(values: list[float]) -> tuple[float | str, float | str, float | str]:
    if not values:
        return "", "", ""
    mean = round(statistics.mean(values), 4)
    if len(values) < 2:
        return mean, "", ""
    std = statistics.stdev(values)
    return mean, round(std, 4), round(1.96 * std / math.sqrt(len(values)), 4)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--algorithms", nargs="+", default=list(DEFAULT_ALGORITHMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--figure-stem", type=Path, default=FIGURE_STEM)
    parser.add_argument(
        "--reward-figure-stem", type=Path, default=REWARD_FIGURE_STEM
    )
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Generate explicitly non-paper partial diagnostics when curves are missing",
    )
    args = parser.parse_args(argv)

    revision_cfg = load_revision_config()
    common_reward_cfg = revision_cfg["common_evaluation_reward"]
    reward_schema = str(common_reward_cfg["schema"])
    reward_algorithms = tuple(common_reward_cfg["plotted_algorithms"])
    reward_figure_algorithms = tuple(common_reward_cfg["figure_algorithms"])
    validation_seeds = tuple(revision_cfg["seeds"]["validation"])
    expected_configs = {
        (spec.algorithm, spec.train_seed): orch.materialize_config(
            spec.to_worker_dict(), validation_seeds
        )
        for spec in headline_specs(revision_cfg)
        if spec.algorithm != "dbas"
    }
    curves: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray | None, Path]]] = {}
    reward_curves: dict[
        str, list[tuple[np.ndarray, np.ndarray, np.ndarray | None, Path]]
    ] = {}
    reward_source_keys: dict[tuple[str, int], str] = {}
    source_rows: dict[str, list[dict[str, object]]] = {}
    missing: list[Path] = []
    for algorithm in args.algorithms:
        seeds = tuple(args.seeds) if args.seeds else train_seeds_for(
            algorithm, config=revision_cfg
        )
        for seed in seeds:
            run_dir = args.artifacts_root / args.pattern.format(algo=algorithm, seed=seed)
            path = run_dir / "eval" / "evaluations.npz"
            if not path.exists():
                missing.append(path)
                continue
            expected_config = expected_configs.get((algorithm, seed))
            if expected_config is None:
                print(
                    f"error: {algorithm}/seed={seed} is not a current formal headline run"
                )
                return 1
            checkpoint_name = selected_model_name(algorithm)
            try:
                provenance = validate_run_artifact(
                    run_dir,
                    expected_config=expected_config,
                    checkpoint_name=checkpoint_name,
                )
            except (FileNotFoundError, ValueError) as exc:
                print(f"error: {exc}")
                return 1
            timesteps, curve, wall_time = load_curve(path)
            try:
                validate_curve_coverage(timesteps, curve, wall_time, run_dir)
            except ValueError as exc:
                print(f"error: {exc}")
                return 1
            curves.setdefault(algorithm, []).append((timesteps, curve, wall_time, run_dir))
            if algorithm in reward_algorithms:
                try:
                    reward_steps, reward_curve, reward_source_key = (
                        load_common_reward_curve(
                            path,
                            algorithm=algorithm,
                            expected_schema=reward_schema,
                        )
                    )
                    validate_curve_coverage(
                        reward_steps, reward_curve, None, run_dir
                    )
                except (KeyError, ValueError) as exc:
                    print(f"error: {exc}")
                    return 1
                if not np.array_equal(reward_steps, timesteps):
                    print(f"error: {run_dir}: welfare/reward timesteps differ")
                    return 1
                reward_curves.setdefault(algorithm, []).append(
                    (reward_steps, reward_curve, None, run_dir)
                )
                reward_source_keys[(algorithm, seed)] = reward_source_key
            source_rows.setdefault(algorithm, []).append({
                "algorithm": algorithm,
                "train_seed": seed,
                **provenance.as_dict(),
                "evaluations_sha256": sha256_file(path),
            })
    if missing:
        status = "warning" if args.allow_partial else "error"
        print(f"{status}: {len(missing)} evaluation curves are missing")
        for path in missing[:20]:
            print(f"  {path}")
        if not args.allow_partial:
            return 1
    if not curves:
        print("error: no revision evaluation curves found")
        return 1

    convergence: list[dict[str, object]] = []
    budget_rows: list[dict[str, object]] = []
    evidence_status = "partial_nonpaper" if missing else "complete_paper"
    for algorithm, runs in curves.items():
        finals: list[float] = []
        references: list[float] = []
        aulcs: list[float] = []
        training_wall: list[float] = []
        threshold_steps = {threshold: [] for threshold in THRESHOLDS}
        threshold_wall = {threshold: [] for threshold in THRESHOLDS}
        budget_values = {budget: [] for budget in BUDGETS}

        for timesteps, curve, wall_time, run_dir in runs:
            tail = max(1, int(len(curve) * FINAL_FRACTION))
            final = float(curve[-tail:].mean())
            reference = float(curve[0])
            finals.append(final)
            references.append(reference)
            aulcs.append(normalized_aulc(timesteps, curve))
            timing_wall = isolated_wall_time(run_dir, wall_time)
            if timing_wall is not None and len(timing_wall):
                training_wall.append(float(timing_wall[-1]))
            for threshold in THRESHOLDS:
                target = progress_target(reference, final, threshold)
                index = threshold_index(curve, target) if target is not None else None
                if index is not None:
                    threshold_steps[threshold].append(int(timesteps[index]))
                    if timing_wall is not None:
                        threshold_wall[threshold].append(float(timing_wall[index]))
            for budget in BUDGETS:
                value = value_at_budget(timesteps, curve, budget)
                if value is not None:
                    budget_values[budget].append(value)

        final_mean, final_std, final_ci = _mean_std_ci(finals)
        reference_mean, reference_std, reference_ci = _mean_std_ci(references)
        aulc_mean, aulc_std, aulc_ci = _mean_std_ci(aulcs)
        source_bundle = provenance_bundle_digest(source_rows[algorithm])
        row: dict[str, object] = {
            "algorithm": algorithm,
            "algorithm_label": ALGORITHM_LABELS.get(algorithm, algorithm),
            "evidence_status": evidence_status,
            "seeds": len(runs),
            "final_welfare_mean": final_mean,
            "final_welfare_std": final_std,
            "final_welfare_ci95": final_ci,
            "reference_welfare_mean": reference_mean,
            "reference_welfare_std": reference_std,
            "reference_welfare_ci95": reference_ci,
            "positive_progress_seed_count": sum(
                final > reference
                for final, reference in zip(finals, references, strict=True)
            ),
            "normalized_aulc_mean": aulc_mean,
            "normalized_aulc_std": aulc_std,
            "normalized_aulc_ci95": aulc_ci,
            "isolated_wall_time_seed_count": len(training_wall),
            "isolated_training_wall_time_s_mean": (
                round(statistics.mean(training_wall), 2) if training_wall else ""
            ),
            "source_bundle_sha256": source_bundle,
        }
        for threshold in THRESHOLDS:
            label = int(threshold * 100)
            steps = threshold_steps[threshold]
            walls = threshold_wall[threshold]
            row[f"steps_to_{label}pct"] = round(statistics.mean(steps)) if steps else ""
            row[f"steps_to_{label}pct_seeds"] = len(steps)
            row[f"wall_s_to_{label}pct"] = (
                round(statistics.mean(walls), 2) if walls else ""
            )
        convergence.append(row)

        for budget in BUDGETS:
            mean, std, ci = _mean_std_ci(budget_values[budget])
            budget_rows.append({
                "algorithm": algorithm,
                "algorithm_label": ALGORITHM_LABELS.get(algorithm, algorithm),
                "evidence_status": evidence_status,
                "budget_steps": budget,
                "welfare_mean": mean,
                "welfare_std": std,
                "welfare_ci95": ci,
                "seeds": len(budget_values[budget]),
                "source_bundle_sha256": source_bundle,
            })

    convergence.sort(key=lambda row: -float(row["final_welfare_mean"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    convergence_path = args.output_dir / "convergence.csv"
    write_csv(convergence_path, convergence)
    budget_path = args.output_dir / "equal_budget.csv"
    write_csv(budget_path, budget_rows)
    sources_path = args.output_dir / "convergence_sources.csv"
    write_csv(
        sources_path,
        [row for algorithm in args.algorithms for row in source_rows.get(algorithm, [])],
    )
    reward_summary: list[dict[str, object]] = []
    for algorithm in reward_algorithms:
        runs = reward_curves.get(algorithm, [])
        if not runs:
            continue
        final_values = []
        for _, curve, _, _ in runs:
            tail = max(1, int(len(curve) * FINAL_FRACTION))
            final_values.append(float(curve[-tail:].mean()))
        mean, std, ci95 = _mean_std_ci(final_values)
        reward_summary.append({
            "algorithm": algorithm,
            "algorithm_label": ALGORITHM_LABELS.get(algorithm, algorithm),
            "evidence_status": evidence_status,
            "reward_schema": reward_schema,
            "seeds": len(runs),
            "final_reward_mean": mean,
            "final_reward_std": std,
            "final_reward_ci95": ci95,
            "source_bundle_sha256": provenance_bundle_digest(
                source_rows[algorithm]
            ),
        })
    reward_summary_path = args.output_dir / "reward_convergence.csv"
    write_csv(reward_summary_path, reward_summary)
    if not args.no_figure:
        render_learning_efficiency_figure(curves, budget_rows, args.figure_stem)
        render_reward_convergence_figure(
            reward_curves,
            args.reward_figure_stem,
            reward_schema=reward_schema,
            algorithms_to_plot=reward_figure_algorithms,
            source_keys=reward_source_keys,
            source_bundles={
                algorithm: provenance_bundle_digest(source_rows[algorithm])
                for algorithm in reward_curves
            },
        )
    print(
        f"wrote {convergence_path}, {budget_path}, {sources_path}, and "
        f"{reward_summary_path}"
    )
    return 0


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_curve(
    algorithm: str,
    runs: list[tuple[np.ndarray, np.ndarray, np.ndarray | None, Path]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align and aggregate one algorithm's train-seed evaluation curves."""

    min_length = min(len(run[0]) for run in runs)
    timesteps = runs[0][0][:min_length]
    if any(not np.array_equal(run[0][:min_length], timesteps) for run in runs[1:]):
        raise ValueError(f"timestep mismatch across {algorithm} train seeds")
    matrix = np.stack([run[1][:min_length] for run in runs])
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0, ddof=1) if len(runs) > 1 else np.zeros_like(mean)
    return timesteps, mean, std


def render_reward_convergence_figure(
    curves: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray | None, Path]]],
    figure_stem: Path,
    *,
    reward_schema: str,
    algorithms_to_plot: tuple[str, ...] = ("proposed", "jdls", "mappo"),
    source_keys: dict[tuple[str, int], str] | None = None,
    source_bundles: dict[str, str] | None = None,
) -> None:
    """Render the common evaluation reward as a standalone convergence plot."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.plotting.revision_style import (
        ALGORITHM_COLORS,
        FIGSIZE_SINGLE_COLUMN,
        LEGEND_STYLE,
        algorithm_style,
        apply_ieee_style,
        save_figure,
        setup_axes,
    )

    apply_ieee_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_COLUMN)
    setup_axes(ax)
    rows: list[dict[str, object]] = []
    algorithms = [
        algorithm for algorithm in algorithms_to_plot if algorithm in curves
    ]
    for algorithm in algorithms:
        timesteps, mean, std = _aggregate_curve(algorithm, curves[algorithm])
        ax.plot(
            timesteps,
            mean,
            label=ALGORITHM_LABELS.get(algorithm, algorithm),
            markevery=max(1, len(timesteps) // 8),
            **algorithm_style(algorithm),
        )
        ax.fill_between(
            timesteps,
            mean - std,
            mean + std,
            color=ALGORITHM_COLORS[algorithm],
            alpha=0.12,
            linewidth=0,
            zorder=1,
        )
        source_key_set = (
            {
                value
                for (source_algorithm, _), value in source_keys.items()
                if source_algorithm == algorithm
            }
            if source_keys else set()
        )
        rows.extend(
            {
                "algorithm": algorithm,
                "timestep": int(step),
                "common_evaluation_reward_mean": float(avg),
                "common_evaluation_reward_seed_std": float(spread),
                "train_seed_count": len(curves[algorithm]),
                "uncertainty": "seed_std",
                "reward_schema": reward_schema,
                "metric_sources": ";".join(sorted(source_key_set)),
                "source_bundle_sha256": (
                    source_bundles.get(algorithm, "") if source_bundles else ""
                ),
            }
            for step, avg, spread in zip(timesteps, mean, std, strict=True)
        )
    ax.set_xlabel(REWARD_FIGURE_XLABEL)
    ax.set_ylabel(REWARD_FIGURE_YLABEL)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.0),
        ncol=len(algorithms),
        mode="expand",
        borderaxespad=0.0,
        fontsize=6.8,
        columnspacing=0.22,
        handlelength=0.85,
        handletextpad=0.22,
        borderpad=0.25,
        **LEGEND_STYLE,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    save_figure(fig, figure_stem)
    write_csv(
        figure_stem.with_name(f"{figure_stem.name}_source.csv"), rows
    )
    plt.close(fig)


def render_learning_efficiency_figure(
    curves: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray | None, Path]]],
    budget_rows: list[dict[str, object]],
    figure_stem: Path,
) -> None:
    """Render convergence and equal-budget evidence as one two-panel figure."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.plotting.revision_style import (
        ALGORITHM_COLORS,
        FIGSIZE_DOUBLE_COLUMN,
        LEGEND_STYLE,
        algorithm_errorbar_style,
        algorithm_style,
        apply_ieee_style,
        finite_or_none,
        optional_errorbar_values,
        ordered_algorithms,
        panel_title,
        save_figure,
        setup_axes,
    )

    apply_ieee_style()
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE_COLUMN)
    for ax in axes:
        setup_axes(ax)

    source_rows: list[dict[str, object]] = []
    learning_algorithms = [
        algorithm for algorithm in ("proposed", "jdls", "mappo", "ebas")
        if algorithm in curves
    ]
    for algorithm in learning_algorithms:
        timesteps, mean, std = _aggregate_curve(algorithm, curves[algorithm])
        style = algorithm_style(algorithm)
        axes[0].plot(
            timesteps,
            mean,
            label=ALGORITHM_LABELS.get(algorithm, algorithm),
            markevery=max(1, len(timesteps) // 8),
            **style,
        )
        axes[0].fill_between(
            timesteps,
            mean - std,
            mean + std,
            color=ALGORITHM_COLORS[algorithm],
            alpha=0.12,
            linewidth=0,
            zorder=1,
        )
        source_rows.extend(
            {
                "algorithm": algorithm,
                "series_kind": "learning_curve",
                "timestep": int(step),
                "budget_steps": "",
                "social_welfare_mean": float(avg),
                "social_welfare_seed_std": float(spread),
                "social_welfare_ci95": "",
                "train_seed_count": len(curves[algorithm]),
                "uncertainty": "seed_std",
            }
            for step, avg, spread in zip(timesteps, mean, std, strict=True)
        )
    panel_title(axes[0], "a", r"Convergence ($\pm$1 SD)")
    axes[0].set_xlabel("Training Timesteps")
    axes[0].set_ylabel("Social Welfare")

    algorithms = ordered_algorithms(row["algorithm"] for row in budget_rows)
    for algorithm in algorithms:
        matches = sorted(
            (row for row in budget_rows if row["algorithm"] == algorithm),
            key=lambda row: int(row["budget_steps"]),
        )
        welfare_means = [finite_or_none(row.get("welfare_mean")) for row in matches]
        if any(value is None for value in welfare_means):
            raise ValueError(f"{algorithm} has a missing equal-budget welfare mean")
        means = [float(value) for value in welfare_means if value is not None]
        stds = [finite_or_none(row.get("welfare_std")) for row in matches]
        errors = [finite_or_none(row.get("welfare_ci95")) for row in matches]
        axes[1].errorbar(
            [int(row["budget_steps"]) / 1000 for row in matches],
            means,
            yerr=optional_errorbar_values(errors),
            label=ALGORITHM_LABELS.get(algorithm, algorithm),
            **algorithm_errorbar_style(algorithm),
        )
        source_rows.extend(
            {
                "algorithm": algorithm,
                "series_kind": "equal_budget",
                "timestep": "",
                "budget_steps": int(row["budget_steps"]),
                "social_welfare_mean": mean,
                "social_welfare_seed_std": "" if std is None else std,
                "social_welfare_ci95": "" if error is None else error,
                "train_seed_count": int(row.get("seeds") or 0),
                "uncertainty": "none" if error is None else "seed_ci95",
            }
            for row, mean, std, error in zip(
                matches, means, stds, errors, strict=True
            )
        )
    panel_title(axes[1], "b", "Equal-Budget Welfare (95% CI)")
    axes[1].set_xlabel("Training Budget (Thousand Timesteps)")
    axes[1].set_xticks([budget // 1000 for budget in BUDGETS])
    axes[1].set_ylabel("Social Welfare")

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=len(algorithms),
        columnspacing=0.8,
        handlelength=1.7,
        **LEGEND_STYLE,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    save_figure(fig, figure_stem)
    write_csv(figure_stem.with_name(f"{figure_stem.name}_source.csv"), source_rows)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
