"""Train and evaluate matched candidate-pruning sensitivity experiments.

The formal study varies only candidate admission/cap parameters while holding
the sensing-quality reference range fixed. Every distinct mechanism gets a
matched TE-PPO policy. The canonical ``r0=80, rho=4`` cell reuses the first
eight headline checkpoints, and the analytically equivalent rho-1/rho-2 cells
share one rho-2 evaluation and training identity.

Exactness label: implementation convenience only (experiment orchestration).
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from src.evaluation import orchestration as orch
from src.evaluation.revision import (
    ALL_ALGORITHMS,
    formal_algorithm_config,
    headline_run_dir,
    headline_specs,
    load_revision_config,
    pruning_overrides,
    pruning_sensitivity_artifact_root,
    pruning_sensitivity_reported_cells,
    pruning_sensitivity_run_dir,
    pruning_sensitivity_specs,
    pruning_sensitivity_training_rho,
    revision_roots,
)
from src.evaluation.sweep import (
    aggregate_sweep_rows,
    evaluate_run_dir,
    tag_episode_rows,
    write_sweep_csv,
)
from src.parallel import TaskSpec
from src.rl.eval import SWEEP_METRIC_NAMES

RUNTIME_METRIC_NAMES = (
    "candidate_pruning_time_ms",
    "follower_solve_time_ms",
    "aggregation_time_ms",
    "inference_wall_time_ms",
)
FORMAL_SENSITIVITY_METRICS = tuple(
    metric for metric in SWEEP_METRIC_NAMES if metric not in RUNTIME_METRIC_NAMES
)
RUNTIME_EVIDENCE_STATUS = "diagnostic_unwarmed_non_final"


@dataclass(frozen=True)
class PruningCell:
    radius_m: float
    diversity_factor: float

    @property
    def key(self) -> tuple[float, float]:
        return self.radius_m, self.diversity_factor


def build_cells(config: dict | None = None) -> tuple[PruningCell, ...]:
    """Return the 16 reported cells from the maintained formal contract."""

    return tuple(
        PruningCell(radius, rho) for radius, rho in pruning_sensitivity_reported_cells(config)
    )


def distinct_evaluation_cells(config: dict | None = None) -> tuple[PruningCell, ...]:
    """Return one cell per mathematically distinct pruning mechanism."""

    cfg = config or load_revision_config()
    return tuple(
        cell
        for cell in build_cells(cfg)
        if pruning_sensitivity_training_rho(cell.diversity_factor, cfg) == cell.diversity_factor
    )


def scenario_overrides_for(
    algorithm: str, cell: PruningCell, config: dict | None = None
) -> dict[str, object]:
    """Return the pure-pruning evaluation environment for one algorithm."""

    cfg = config or load_revision_config()
    override = pruning_overrides("pruning_only", cell.radius_m, cell.diversity_factor, config=cfg)
    if algorithm != "proposed":
        # Baselines do not acquire TE-PPO's candidate-selection component. The
        # sensing-quality range is fixed, so their environment is identical in
        # every formal candidate-pruning cell and their headline policies remain matched.
        return {
            "enable_candidate_pruning": False,
            "sensing_range_m": override["sensing_range_m"],
        }
    return override


def _reported_proposed_trial(
    source_rows: list[dict[str, object]],
    cell: PruningCell,
    train_seed: int,
    config: dict,
) -> list[dict[str, object]]:
    """Tag one physical evaluation for its reported sensitivity cell.

    ``rho=1`` and ``rho=2`` induce the same maintained candidate cap over the
    formal task geometry.  They therefore share one physical rho-2 evaluation;
    this function deep-copies that result before changing reporting metadata so
    the two cells remain exactly equal in every measured quantity.

    Exactness label: implementation convenience only (report construction).
    """

    trained_rho = pruning_sensitivity_training_rho(cell.diversity_factor, config)
    trial = copy.deepcopy(source_rows)
    tag_episode_rows(
        trial,
        algorithm_name="proposed",
        train_seed=train_seed,
        scenario_overrides=scenario_overrides_for("proposed", cell, config),
    )
    for row in trial:
        row.update(
            {
                "sensitivity_mode": "pruning_only",
                "sensitivity_protocol": "matched_training",
                "r0_m": cell.radius_m,
                "diversity_factor": cell.diversity_factor,
                "trained_r0_m": cell.radius_m,
                "trained_diversity_factor": trained_rho,
                "effective_candidate_cap": max(2, math.ceil(trained_rho)),
                "analytically_equivalent": cell.diversity_factor != trained_rho,
                "runtime_evidence_status": RUNTIME_EVIDENCE_STATUS,
                "evidence_status": "complete_paper",
            }
        )
    return trial


def required_checkpoint_dirs(
    algorithms: tuple[str, ...], config: dict | None = None
) -> tuple[Path, ...]:
    """Return every trained artifact consumed by formal candidate-pruning evaluation."""

    cfg = config or load_revision_config()
    seeds = tuple(cfg["pruning_sensitivity"]["train_seeds"])
    paths: list[Path] = []
    if "proposed" in algorithms:
        for radius, reported_rho in pruning_sensitivity_reported_cells(cfg):
            trained_rho = pruning_sensitivity_training_rho(reported_rho, cfg)
            for seed in seeds:
                paths.append(pruning_sensitivity_run_dir(radius, trained_rho, int(seed), cfg))
    for algorithm in algorithms:
        if algorithm in ("proposed", "dbas"):
            continue
        paths.extend(headline_run_dir(algorithm, int(seed), cfg) for seed in seeds)
    return tuple(dict.fromkeys(paths))


def _validate_checkpoints(algorithms: tuple[str, ...], config: dict) -> None:
    missing = [
        path
        for path in required_checkpoint_dirs(algorithms, config)
        if not (path / "summary.json").exists()
    ]
    if missing:
        raise FileNotFoundError(
            "missing matched pruning-sensitivity checkpoints; run the train phase first:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )


def _missing_headline_dependencies(config: dict) -> tuple[Path, ...]:
    """Return missing canonical/baseline artifacts required by formal candidate-pruning."""

    seeds = tuple(config["pruning_sensitivity"]["train_seeds"])
    return tuple(
        path
        for algorithm in ALL_ALGORITHMS
        if algorithm != "dbas"
        for seed in seeds
        for path in (headline_run_dir(algorithm, int(seed), config),)
        if not (path / "summary.json").exists()
    )


def _rank_rows(
    aggregates: list[dict[str, object]], metric: str, config: dict
) -> list[dict[str, object]]:
    """Rank every cell against the configured canonical default."""

    grouped: dict[tuple[float, float], list[dict[str, object]]] = {}
    for row in aggregates:
        key = (float(row["r0_m"]), float(row["diversity_factor"]))
        grouped.setdefault(key, []).append(row)

    sensitivity = config["pruning_sensitivity"]
    default_key = (
        float(sensitivity["canonical_radius_m"]),
        float(sensitivity["canonical_diversity_factor"]),
    )
    if default_key not in grouped:
        raise ValueError(f"canonical pruning cell {default_key} is missing")
    algorithm_sets = [{str(row["algorithm"]) for row in rows} for rows in grouped.values()]
    if not algorithm_sets or any(current != algorithm_sets[0] for current in algorithm_sets):
        raise ValueError("pruning cells do not contain a common algorithm set")

    value_key = f"{metric}_seed_mean"
    default_ranked = sorted(
        grouped[default_key], key=lambda row: float(row[value_key]), reverse=True
    )
    reference = {str(row["algorithm"]): rank for rank, row in enumerate(default_ranked, 1)}

    output: list[dict[str, object]] = []
    for key, rows in sorted(grouped.items()):
        ranked = sorted(rows, key=lambda row: float(row[value_key]), reverse=True)
        current = {str(row["algorithm"]): rank for rank, row in enumerate(ranked, 1)}
        stable = current == reference
        for algorithm, rank in current.items():
            output.append(
                {
                    "metric": metric,
                    "sensitivity_mode": "pruning_only",
                    "r0_m": key[0],
                    "diversity_factor": key[1],
                    "algorithm": algorithm,
                    "rank": rank,
                    "default_rank": reference[algorithm],
                    "full_order_stable": stable,
                    "canonical_r0_m": default_key[0],
                    "canonical_diversity_factor": default_key[1],
                }
            )
    return output


def _seed_means(
    rows: list[dict[str, object]],
    *,
    algorithm: str,
    radius: float,
    rho: float,
    metric: str,
    seeds: tuple[int, ...],
    eval_seeds: tuple[int, ...],
) -> dict[int, float]:
    result: dict[int, float] = {}
    for seed in seeds:
        matching = [
            row
            for row in rows
            if row.get("algorithm") == algorithm
            and float(row.get("r0_m", float("nan"))) == radius
            and float(row.get("diversity_factor", float("nan"))) == rho
            and int(row.get("train_seed", -1)) == seed
        ]
        actual_eval = [int(row.get("eval_seed", -1)) for row in matching]
        if set(actual_eval) != set(eval_seeds) or len(actual_eval) != len(set(actual_eval)):
            raise ValueError(
                f"{algorithm}/r0={radius:g}/rho={rho:g}/seed={seed} lacks "
                "the complete unique pruning-test bank"
            )
        values = [float(row.get(metric, float("nan"))) for row in matching]
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError(
                f"{algorithm}/r0={radius:g}/rho={rho:g}/{metric}/seed={seed} "
                "contains non-finite values"
            )
        result[seed] = sum(values) / len(values)
    return result


def _paired_delta(
    candidate: dict[int, float], reference: dict[int, float]
) -> dict[str, float | int]:
    if candidate.keys() != reference.keys() or not candidate:
        raise ValueError("paired pruning comparison requires identical nonempty seeds")
    deltas: list[float] = []
    for seed in candidate:
        denominator = reference[seed]
        if denominator == 0.0:
            raise ValueError("paired pruning comparison has a zero denominator")
        deltas.append(100.0 * (candidate[seed] - denominator) / abs(denominator))
    mean = sum(deltas) / len(deltas)
    if len(deltas) > 1:
        variance = sum((value - mean) ** 2 for value in deltas) / (len(deltas) - 1)
        std = math.sqrt(variance)
        ci95 = 1.96 * std / math.sqrt(len(deltas))
    else:
        std = ci95 = float("nan")
    observed = abs(mean)
    extreme = sum(
        abs(sum(sign * value for sign, value in zip(signs, deltas, strict=True)) / len(deltas))
        >= observed - 1e-12
        for signs in itertools.product((-1.0, 1.0), repeat=len(deltas))
    )
    return {
        "mean": mean,
        "std": std,
        "ci95": ci95,
        "paired_permutation_p": extreme / (2 ** len(deltas)),
        "pairs": len(deltas),
    }


def _paired_rows(rows: list[dict[str, object]], config: dict) -> list[dict[str, object]]:
    sensitivity = config["pruning_sensitivity"]
    seeds = tuple(int(value) for value in sensitivity["train_seeds"])
    eval_seeds = tuple(int(value) for value in config["seeds"]["pruning_test"])
    default_radius = float(sensitivity["canonical_radius_m"])
    default_rho = float(sensitivity["canonical_diversity_factor"])
    metrics = (
        "social_welfare",
        "raw_rsu_reward",
        "perception_quality_total",
        "task_success_rate",
    )
    output: list[dict[str, object]] = []
    for cell in build_cells(config):
        for metric in metrics:
            candidate = _seed_means(
                rows,
                algorithm="proposed",
                radius=cell.radius_m,
                rho=cell.diversity_factor,
                metric=metric,
                seeds=seeds,
                eval_seeds=eval_seeds,
            )
            comparisons = {
                "canonical": _seed_means(
                    rows,
                    algorithm="proposed",
                    radius=default_radius,
                    rho=default_rho,
                    metric=metric,
                    seeds=seeds,
                    eval_seeds=eval_seeds,
                ),
                "ebas": _seed_means(
                    rows,
                    algorithm="ebas",
                    radius=cell.radius_m,
                    rho=cell.diversity_factor,
                    metric=metric,
                    seeds=seeds,
                    eval_seeds=eval_seeds,
                ),
            }
            for comparison, reference in comparisons.items():
                stats = _paired_delta(candidate, reference)
                output.append(
                    {
                        "r0_m": cell.radius_m,
                        "diversity_factor": cell.diversity_factor,
                        "metric": metric,
                        "comparison": comparison,
                        "change_pct": stats["mean"],
                        "change_pct_std": stats["std"],
                        "change_pct_ci95": stats["ci95"],
                        "paired_permutation_p": stats["paired_permutation_p"],
                        "pairs": stats["pairs"],
                    }
                )
    return output


def _paper_table_rows(
    aggregates: list[dict[str, object]],
    paired_rows: list[dict[str, object]],
    rank_rows: list[dict[str, object]],
    config: dict,
) -> list[dict[str, object]]:
    """Build the compact numeric table intended for the candidate-pruning response."""

    proposed = {
        (float(row["r0_m"]), float(row["diversity_factor"])): row
        for row in aggregates
        if row["algorithm"] == "proposed"
    }
    paired = {
        (
            float(row["r0_m"]),
            float(row["diversity_factor"]),
            str(row["metric"]),
            str(row["comparison"]),
        ): row
        for row in paired_rows
    }
    ranks = {
        (
            float(row["r0_m"]),
            float(row["diversity_factor"]),
            str(row["metric"]),
        ): int(row["rank"])
        for row in rank_rows
        if row["algorithm"] == "proposed"
    }

    output: list[dict[str, object]] = []
    for cell in build_cells(config):
        key = cell.key
        row = proposed[key]
        sw_canonical = paired[(*key, "social_welfare", "canonical")]
        sw_ebas = paired[(*key, "social_welfare", "ebas")]
        rsu_canonical = paired[(*key, "raw_rsu_reward", "canonical")]
        rsu_ebas = paired[(*key, "raw_rsu_reward", "ebas")]
        output.append(
            {
                "r0_m": cell.radius_m,
                "diversity_factor": cell.diversity_factor,
                "effective_candidate_cap": int(row["effective_candidate_cap"]),
                "analytically_equivalent": bool(row["analytically_equivalent"]),
                "social_welfare_mean": float(row["social_welfare_seed_mean"]),
                "social_welfare_ci95": float(row["social_welfare_seed_ci95"]),
                "social_welfare_change_vs_canonical_pct": float(sw_canonical["change_pct"]),
                "social_welfare_change_vs_ebas_pct": float(sw_ebas["change_pct"]),
                "social_welfare_rank": ranks[(*key, "social_welfare")],
                "rsu_utility_mean": float(row["raw_rsu_reward_seed_mean"]),
                "rsu_utility_ci95": float(row["raw_rsu_reward_seed_ci95"]),
                "rsu_utility_change_vs_canonical_pct": float(rsu_canonical["change_pct"]),
                "rsu_utility_change_vs_ebas_pct": float(rsu_ebas["change_pct"]),
                "rsu_utility_rank": ranks[(*key, "raw_rsu_reward")],
                "perception_quality_mean": float(row["perception_quality_total_seed_mean"]),
                "perception_quality_ci95": float(row["perception_quality_total_seed_ci95"]),
                "task_success_rate_mean": float(row["task_success_rate_seed_mean"]),
                "task_success_rate_ci95": float(row["task_success_rate_seed_ci95"]),
            }
        )
    return output


def _finite_metric(row: dict[str, object], key: str, context: str) -> float:
    """Return one finite plotting value or fail closed on stale summaries."""

    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{context} lacks numeric {key}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{context} has non-finite {key}")
    return value


def _sensitivity_figure_source_rows(
    aggregates: list[dict[str, object]], config: dict
) -> list[dict[str, object]]:
    """Build one traceable row per distinct mechanism plus the EBAS reference.

    Exactness label: implementation convenience only (report visualization).
    The full 4-by-4 evidence remains in ``summary.csv`` and ``paper_table.csv``;
    this view collapses only rows proven analytically and numerically equivalent.
    """

    metric_columns = (
        "social_welfare_seed_mean",
        "social_welfare_seed_ci95",
        "raw_rsu_reward_seed_mean",
        "raw_rsu_reward_seed_ci95",
    )
    sensitivity = config["pruning_sensitivity"]
    expected_radii = tuple(float(value) for value in sensitivity["radius_grid_m"])
    expected_rhos = tuple(float(value) for value in sensitivity["reported_diversity_grid"])
    proposed_rows = [row for row in aggregates if row.get("algorithm") == "proposed"]
    proposed = {
        (float(row["r0_m"]), float(row["diversity_factor"])): row
        for row in proposed_rows
    }
    if len(proposed) != len(proposed_rows):
        raise ValueError("pruning sensitivity figure has duplicate proposed cells")
    expected_keys = {(radius, rho) for radius in expected_radii for rho in expected_rhos}
    if set(proposed) != expected_keys:
        missing = sorted(expected_keys.difference(proposed))
        extra = sorted(set(proposed).difference(expected_keys))
        raise ValueError(
            "pruning sensitivity figure has incomplete proposed grid: "
            f"missing={missing}, extra={extra}"
        )

    source_rows: list[dict[str, object]] = []
    for radius in expected_radii:
        by_cap: dict[int, list[tuple[float, dict[str, object]]]] = {}
        for rho in expected_rhos:
            row = proposed[(radius, rho)]
            context = f"r0={radius:g},rho={rho:g}"
            raw_cap = _finite_metric(row, "effective_candidate_cap", context)
            if not raw_cap.is_integer():
                raise ValueError(f"{context} has non-integer effective_candidate_cap")
            cap = int(raw_cap)
            by_cap.setdefault(cap, []).append((rho, row))
        for cap in sorted(by_cap):
            equivalents = by_cap[cap]
            reference_rho, reference = equivalents[0]
            context = f"r0={radius:g},N_max={cap:g}"
            values = {
                column: _finite_metric(reference, column, f"{context},rho={reference_rho:g}")
                for column in metric_columns
            }
            for rho, row in equivalents[1:]:
                for column, expected in values.items():
                    actual = _finite_metric(row, column, f"{context},rho={rho:g}")
                    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
                        raise ValueError(
                            f"equivalent pruning cells differ for {context}, "
                            f"metric={column}, rho={reference_rho:g}/{rho:g}"
                        )
            source_rows.append(
                {
                    "series_kind": "proposed",
                    "pruning_radius_m": radius,
                    "reported_diversity_factors": ";".join(
                        f"{rho:g}" for rho, _ in equivalents
                    ),
                    "effective_candidate_cap": cap,
                    "social_welfare_mean": values["social_welfare_seed_mean"],
                    "social_welfare_ci95": values["social_welfare_seed_ci95"],
                    "rsu_utility_mean": values["raw_rsu_reward_seed_mean"],
                    "rsu_utility_ci95": values["raw_rsu_reward_seed_ci95"],
                    "train_seed_count": len(sensitivity["train_seeds"]),
                    "evidence_status": "complete_paper",
                }
            )

    ebas = [row for row in aggregates if row.get("algorithm") == "ebas"]
    if not ebas:
        raise ValueError("pruning sensitivity figure requires EBAS rows")
    reference_values = {
        column: _finite_metric(ebas[0], column, "EBAS reference")
        for column in metric_columns
    }
    for row in ebas[1:]:
        for column, expected in reference_values.items():
            actual = _finite_metric(row, column, "EBAS reference")
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"EBAS reference differs across pruning cells for {column}")
    source_rows.append(
        {
            "series_kind": "ebas_reference",
            "pruning_radius_m": "",
            "reported_diversity_factors": "",
            "effective_candidate_cap": "",
            "social_welfare_mean": reference_values["social_welfare_seed_mean"],
            "social_welfare_ci95": reference_values["social_welfare_seed_ci95"],
            "rsu_utility_mean": reference_values["raw_rsu_reward_seed_mean"],
            "rsu_utility_ci95": reference_values["raw_rsu_reward_seed_ci95"],
            "train_seed_count": len(sensitivity["train_seeds"]),
            "evidence_status": "complete_paper",
        }
    )
    return source_rows


def _render_sensitivity_figure(
    aggregates: list[dict[str, object]],
    output_dir: Path,
    config: dict,
    *,
    catalog_dir: Path | None = None,
) -> None:
    """Render the paper-facing candidate-pruning sensitivity comparison with 95% CIs.

    Exactness label: implementation convenience only (report visualization).
    """

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    from src.plotting.revision_style import (
        FIGSIZE_DOUBLE_COLUMN,
        LEGEND_STYLE,
        apply_ieee_style,
        errorbar_style,
        panel_title,
        save_figure,
        setup_axes,
    )

    apply_ieee_style()

    source_rows = _sensitivity_figure_source_rows(aggregates, config)
    proposed = [row for row in source_rows if row["series_kind"] == "proposed"]
    ebas = next(row for row in source_rows if row["series_kind"] == "ebas_reference")

    sensitivity = config["pruning_sensitivity"]
    canonical_radius = float(sensitivity["canonical_radius_m"])
    canonical_rho = float(sensitivity["canonical_diversity_factor"])
    canonical_rho_label = f"{canonical_rho:g}"
    canonical_rows = [
        row
        for row in proposed
        if float(row["pruning_radius_m"]) == canonical_radius
        and canonical_rho_label in str(row["reported_diversity_factors"]).split(";")
    ]
    if len(canonical_rows) != 1:
        raise ValueError("pruning sensitivity figure cannot identify canonical display cell")
    canonical_cap = int(canonical_rows[0]["effective_candidate_cap"])
    radii = [float(value) for value in sensitivity["radius_grid_m"]]
    caps = sorted({int(row["effective_candidate_cap"]) for row in proposed})
    rho_labels: dict[int, str] = {}
    for cap in caps:
        reported = {
            str(row["reported_diversity_factors"])
            for row in proposed
            if int(row["effective_candidate_cap"]) == cap
        }
        if len(reported) != 1:
            raise ValueError(f"inconsistent diversity-factor labels for N_max={cap:g}")
        rho_labels[cap] = reported.pop().replace(";", ",")
    colors = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")
    markers = ("o", "s", "^", "D")
    offsets = np.linspace(-0.18, 0.18, len(radii))
    metrics = (
        ("social_welfare", "Social Welfare"),
        ("rsu_utility", "RSU Utility"),
    )

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE_COLUMN, sharex=True)
    radius_handles: list[Line2D] = []
    for panel, (ax, (metric, title)) in enumerate(zip(axes, metrics, strict=True)):
        values = {
            (float(row["pruning_radius_m"]), int(row["effective_candidate_cap"])): float(
                row[f"{metric}_mean"]
            )
            for row in proposed
        }
        intervals = {
            (float(row["pruning_radius_m"]), int(row["effective_candidate_cap"])): float(
                row[f"{metric}_ci95"]
            )
            for row in proposed
        }
        ebas_mean = float(ebas[f"{metric}_mean"])
        ebas_ci = float(ebas[f"{metric}_ci95"])

        ax.axhspan(
            ebas_mean - ebas_ci,
            ebas_mean + ebas_ci,
            color="#808080",
            alpha=0.08,
            linewidth=0,
            zorder=0,
        )
        ax.axhline(
            ebas_mean,
            color="#666666",
            linestyle=(0, (4, 2)),
            linewidth=0.9,
            zorder=1,
        )

        for radius, color, marker, offset in zip(radii, colors, markers, offsets, strict=True):
            means = np.asarray([values[(radius, cap)] for cap in caps])
            cis = np.asarray([intervals[(radius, cap)] for cap in caps])
            positions = np.arange(len(caps), dtype=float) + offset
            ax.errorbar(
                positions,
                means,
                yerr=cis,
                color=color,
                marker=marker,
                markersize=4.5,
                markerfacecolor="white",
                markeredgewidth=1.1,
                linestyle="none",
                zorder=2,
                **errorbar_style(color),
            )
            if ax is axes[0]:
                radius_handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=color,
                        marker=marker,
                        markerfacecolor="white",
                        markersize=4.5,
                        linestyle="none",
                        label=rf"$r_0={radius:g}$ m",
                    )
                )

        canonical_offset = offsets[radii.index(canonical_radius)]
        canonical_position = caps.index(canonical_cap) + canonical_offset
        ax.plot(
            canonical_position,
            values[(canonical_radius, canonical_cap)],
            marker="o",
            markersize=8.5,
            markerfacecolor="none",
            markeredgecolor="#111111",
            markeredgewidth=1.15,
            linestyle="none",
            zorder=3,
        )
        setup_axes(ax)
        panel_title(ax, "ab"[panel], title)
        ax.set_xticks(
            range(len(caps)),
            [
                rf"$\varrho={rho_labels[cap]}$" + "\n" + rf"$N_{{\max}}={cap:g}$"
                for cap in caps
            ],
        )
        ax.set_xlabel(r"Diversity Factor $\varrho$")
        ax.set_ylabel(title)

    reference_handles = [
        Patch(
            facecolor="#808080",
            alpha=0.12,
            edgecolor="#666666",
            label="EBAS (95% CI)",
        ),
    ]
    legend_handles = [*radius_handles, *reference_handles]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=5,
        columnspacing=0.75,
        handlelength=1.55,
        **LEGEND_STYLE,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88), w_pad=1.5)
    save_figure(fig, output_dir / "pruning_sensitivity_comparison")
    if catalog_dir is not None:
        save_figure(fig, catalog_dir / "figR_pruning_sensitivity")
        write_sweep_csv(catalog_dir / "figR_pruning_sensitivity_source.csv", source_rows)
    plt.close(fig)


def plot_existing(config: dict | None = None) -> int:
    """Regenerate the paper-facing figure from an existing formal summary.

    Exactness label: implementation convenience only (report visualization).
    """

    cfg = config or load_revision_config()
    outputs_root = revision_roots(cfg)[1]
    output_dir = outputs_root / "pruning_sensitivity" / "matched"
    summary_path = output_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"{summary_path} not found; run the controlled sensitivity evaluation first"
        )
    with summary_path.open(newline="", encoding="utf-8") as handle:
        aggregates = list(csv.DictReader(handle))
    _render_sensitivity_figure(
        aggregates,
        output_dir,
        cfg,
        catalog_dir=outputs_root / "figures",
    )
    print(f"wrote pruning sensitivity comparison figure to {output_dir}")
    return 0


def train(*, dry_run: bool, retrain_stale: bool, max_workers: int | None = None) -> int:
    cfg = load_revision_config()
    missing_dependencies = _missing_headline_dependencies(cfg)
    if missing_dependencies and not dry_run:
        print(
            "error: matched pruning sensitivity requires completed headline "
            "checkpoints:\n" + "\n".join(f"  - {path}" for path in missing_dependencies),
            file=sys.stderr,
        )
        return 1
    specs = pruning_sensitivity_specs(cfg)
    root = pruning_sensitivity_artifact_root(cfg)
    validation_seeds = tuple(cfg["seeds"]["validation"])
    plan = orch.build_train_plan(
        specs,
        root,
        validation_seeds,
        force=False,
        retrain_stale=retrain_stale,
    )
    if dry_run:
        orch.report_train_plan(plan)
        print(f"missing headline dependencies: {len(missing_dependencies)}")
        print(
            "canonical r0=80/rho=4: reuse "
            f"{len(cfg['pruning_sensitivity']['train_seeds'])} headline checkpoints"
        )
        print(
            f"reported cells={len(build_cells(cfg))}; "
            f"distinct cells={len(distinct_evaluation_cells(cfg))}"
        )
        return 0
    if any(entry.action == "blocked" for entry in plan):
        orch.report_train_plan(plan)
        print("error: pass --retrain-stale for stale/partial pruning runs", file=sys.stderr)
        return 1

    run_id = orch.new_run_id()
    manifest = revision_roots(cfg)[1] / "manifests" / f"pruning_{run_id}.jsonl"
    worker_limit = int(
        max_workers if max_workers is not None else cfg["execution"].get("max_workers", 1)
    )
    sb3_cpu_slots = int(cfg["execution"]["sb3_cpu_slots_per_task"])
    tasks: list[TaskSpec] = []
    for entry in plan:
        if entry.action not in ("train", "retrain"):
            continue
        if entry.action == "retrain":
            shutil.rmtree(entry.run_dir)
        materialized = orch.materialize_config(entry.spec.to_worker_dict(), validation_seeds)
        tasks.append(
            TaskSpec(
                func=orch.train_worker,
                args=(
                    entry.spec.to_worker_dict(),
                    validation_seeds,
                    str(root),
                    "scheduled",
                ),
                label=f"pruning/{entry.spec.run_name}",
                cpus_needed=orch.training_task_cpu_slots(materialized, sb3_cpu_slots=sb3_cpu_slots),
                vram_needed_mb=orch.MODEL_TASK_VRAM_MB,
                vram_reserve_mb=orch.MODEL_TASK_VRAM_MB,
            )
        )
    if tasks:
        try:
            orch.run_tasks_with_manifest(
                tasks,
                parallel=worker_limit > 1,
                max_workers=worker_limit,
                fail_fast=True,
                manifest=manifest,
                on_success=lambda result: orch.write_training_record(result, run_id=run_id),
            )
        except (Exception, SystemExit) as exc:
            print(f"error: pruning sensitivity training failed: {exc}", file=sys.stderr)
            return 1
    print(f"pruning sensitivity training complete; manifest: {manifest}")
    return 0


def evaluate(algorithms: tuple[str, ...] = ALL_ALGORITHMS) -> int:
    cfg = load_revision_config()
    if not {"proposed", "ebas"}.issubset(algorithms):
        raise ValueError("formal pruning sensitivity requires both proposed and ebas")
    _validate_checkpoints(algorithms, cfg)
    sensitivity = cfg["pruning_sensitivity"]
    train_seeds = tuple(int(value) for value in sensitivity["train_seeds"])
    eval_seeds = tuple(int(value) for value in cfg["seeds"]["pruning_test"])
    validation_seeds = tuple(int(value) for value in cfg["seeds"]["validation"])
    noncanonical_specs = {
        (
            float(spec.scenario_overrides["candidate_pruning_radius_m"]),
            float(spec.scenario_overrides["diversity_factor"]),
            int(spec.train_seed),
        ): spec
        for spec in pruning_sensitivity_specs(cfg)
    }
    headline = {(spec.algorithm, int(spec.train_seed)): spec for spec in headline_specs(cfg)}
    rows: list[dict[str, object]] = []

    if "proposed" in algorithms:
        evaluated: dict[tuple[float, float, int], list[dict[str, object]]] = {}
        for cell in distinct_evaluation_cells(cfg):
            for seed in train_seeds:
                key = (cell.radius_m, cell.diversity_factor, seed)
                spec = noncanonical_specs.get(key, headline[("proposed", seed)])
                expected = orch.materialize_config(spec.to_worker_dict(), validation_seeds)
                evaluated[key] = evaluate_run_dir(
                    "proposed",
                    pruning_sensitivity_run_dir(cell.radius_m, cell.diversity_factor, seed, cfg),
                    eval_seeds,
                    scenario_override=scenario_overrides_for("proposed", cell, cfg),
                    expected_config=expected,
                )

        for cell in build_cells(cfg):
            trained_rho = pruning_sensitivity_training_rho(cell.diversity_factor, cfg)
            for seed in train_seeds:
                trial = _reported_proposed_trial(
                    evaluated[(cell.radius_m, trained_rho, seed)], cell, seed, cfg
                )
                rows.extend(trial)

    # Baselines do not depend on either pruning parameter in this controlled
    # experiment. Evaluate each matched seed once, then attach the identical
    # physical results to every cell for transparent ranking and pairing.
    for algorithm in algorithms:
        if algorithm == "proposed":
            continue
        seeds = train_seeds[:1] if algorithm == "dbas" else train_seeds
        baseline_rows: list[tuple[int, list[dict[str, object]]]] = []
        for seed in seeds:
            override = scenario_overrides_for(algorithm, build_cells(cfg)[0], cfg)
            if algorithm == "dbas":
                base_config = formal_algorithm_config("dbas", cfg)
                effective = copy.deepcopy(base_config)
                effective.setdefault("scenario", {}).update(override)
                trial = evaluate_run_dir(
                    "dbas",
                    Path("."),
                    eval_seeds,
                    dbas_config=base_config,
                    scenario_override=override,
                    expected_config=effective,
                )
            else:
                spec = headline[(algorithm, seed)]
                expected = orch.materialize_config(spec.to_worker_dict(), validation_seeds)
                trial = evaluate_run_dir(
                    algorithm,
                    headline_run_dir(algorithm, seed, cfg),
                    eval_seeds,
                    scenario_override=override,
                    expected_config=expected,
                )
            baseline_rows.append((seed, trial))

        for cell in build_cells(cfg):
            for seed, source_rows in baseline_rows:
                trial = copy.deepcopy(source_rows)
                tag_episode_rows(
                    trial,
                    algorithm_name=algorithm,
                    train_seed=seed,
                    scenario_overrides=scenario_overrides_for(algorithm, cell, cfg),
                )
                for row in trial:
                    row.update(
                        {
                            "sensitivity_mode": "pruning_only",
                            "sensitivity_protocol": "matched_training",
                            "r0_m": cell.radius_m,
                            "diversity_factor": cell.diversity_factor,
                            "trained_r0_m": "",
                            "trained_diversity_factor": "",
                            "effective_candidate_cap": "",
                            "analytically_equivalent": False,
                            "runtime_evidence_status": RUNTIME_EVIDENCE_STATUS,
                            "evidence_status": "complete_paper",
                        }
                    )
                rows.extend(trial)

    aggregates = aggregate_sweep_rows(
        rows,
        group_fields=("algorithm", "sensitivity_mode", "r0_m", "diversity_factor"),
        metrics=FORMAL_SENSITIVITY_METRICS,
        uncertainty="seed",
    )
    for row in aggregates:
        row.update(
            {
                "sensitivity_protocol": "matched_training",
                "evidence_status": "complete_paper",
            }
        )
        if row["algorithm"] == "proposed":
            trained_rho = pruning_sensitivity_training_rho(float(row["diversity_factor"]), cfg)
            row.update(
                {
                    "trained_r0_m": float(row["r0_m"]),
                    "trained_diversity_factor": trained_rho,
                    "effective_candidate_cap": max(2, math.ceil(trained_rho)),
                    "analytically_equivalent": (float(row["diversity_factor"]) != trained_rho),
                }
            )
    outputs_root = revision_roots(cfg)[1]
    output_dir = outputs_root / "pruning_sensitivity" / "matched"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_sweep_csv(output_dir / "episodes.csv", rows)
    write_sweep_csv(output_dir / "summary.csv", aggregates)
    rank_rows: list[dict[str, object]] = []
    for metric in ("social_welfare", "raw_rsu_reward"):
        rank_rows.extend(_rank_rows(aggregates, metric, cfg))
    paired_rows = _paired_rows(rows, cfg)
    write_sweep_csv(output_dir / "rank_stability.csv", rank_rows)
    write_sweep_csv(output_dir / "paired_deltas.csv", paired_rows)
    write_sweep_csv(
        output_dir / "paper_table.csv",
        _paper_table_rows(aggregates, paired_rows, rank_rows, cfg),
    )
    _render_sensitivity_figure(
        aggregates,
        output_dir,
        cfg,
        catalog_dir=outputs_root / "figures",
    )
    print(f"wrote formal matched candidate-pruning outputs to {output_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("phase", choices=("plan", "train", "evaluate", "plot"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retrain-stale", action="store_true")
    parser.add_argument("--max-workers", type=int)
    args = parser.parse_args(argv)
    if args.max_workers is not None and args.max_workers < 1:
        parser.error("--max-workers must be positive")
    if args.phase == "plan":
        return train(dry_run=True, retrain_stale=False, max_workers=args.max_workers)
    if args.phase == "train":
        return train(
            dry_run=args.dry_run,
            retrain_stale=args.retrain_stale,
            max_workers=args.max_workers,
        )
    try:
        if args.phase == "plot":
            return plot_existing()
        return evaluate()
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
