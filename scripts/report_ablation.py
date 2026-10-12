"""Render the controlled five-arm ablation report.

Accept the dedicated ablation output, whose shared settings and engine-specific
differences are declared in the experiment protocol.
Exactness label: implementation convenience only.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

ARM_LABELS = {
    "full": "TE-PPO (Full)",
    "no_dynamic_pricing": "w/o Pricing",
    "no_bandwidth_allocation": "w/o Bandwidth",
    "no_analytical_best_response": "w/o Online Analytical BR",
    "no_candidate_pruning": "w/o Pruning",
}
BENEFIT_METRICS = (
    ("social_welfare", "Welfare"),
    ("raw_rsu_reward", "RSU Utility"),
    ("perception_quality_total", "Quality"),
)


def missing_ablation_arms(by_arm: dict[str, dict[str, str]]) -> set[str]:
    """Return required ablation arms absent from a parsed summary mapping.

    Exactness: implementation convenience only.  This check validates report
    completeness and does not alter any mathematical or training semantics.
    """

    return set(ARM_LABELS).difference(by_arm)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--csv", type=Path,
        default=Path("outputs/revision_2026_v5/ablation/summary.csv"),
    )
    parser.add_argument(
        "--figure-stem", type=Path,
        default=Path("outputs/revision_2026_v5/figures/figR_ablation"),
    )
    parser.add_argument(
        "--timing-csv",
        type=Path,
        default=Path(
            "outputs/revision_2026_v5/tables/"
            "ablation_inference_latency_summary.csv"
        ),
    )
    args = parser.parse_args(argv)
    if not args.csv.exists():
        print(f"error: {args.csv} not found; run the controlled ablation evaluation first")
        return 1
    if not args.timing_csv.exists():
        print(
            f"error: {args.timing_csv} not found; run the isolated ablation "
            "timing benchmark first"
        )
        return 1
    rows = list(csv.DictReader(args.csv.open(encoding="utf-8")))
    by_arm = {row["ablation_arm"]: row for row in rows}
    missing = missing_ablation_arms(by_arm)
    if missing:
        print(f"error: incomplete ablation summary; missing {sorted(missing)}")
        return 1
    timing_rows = list(csv.DictReader(args.timing_csv.open(encoding="utf-8")))
    timing_by_arm = {row["ablation_arm"]: row for row in timing_rows}
    timing_missing = missing_ablation_arms(timing_by_arm)
    if timing_missing:
        print(f"error: incomplete ablation timing; missing {sorted(timing_missing)}")
        return 1
    render(by_arm, timing_by_arm, args.figure_stem)
    for arm, row in by_arm.items():
        status = row.get("energy_per_participating_vehicle_j_status", "")
        if status and status != "estimable":
            print(
                f"note: {ARM_LABELS[arm]} energy/participant is N/A "
                f"({status})"
            )
    print(f"source: {args.csv}")
    print(f"timing source: {args.timing_csv}")
    print(f"wrote {args.figure_stem}.png / .pdf")
    return 0


def render(
    by_arm: dict[str, dict[str, str]],
    timing_by_arm: dict[str, dict[str, str]],
    figure_stem: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.plotting.revision_style import (
        ALGORITHM_COLORS,
        BAR_ERROR_STYLE,
        ERRORBAR_CAPSIZE,
        FIGSIZE_DOUBLE_COLUMN,
        LEGEND_STYLE,
        apply_ieee_style,
        panel_title,
        save_figure,
        setup_axes,
    )

    apply_ieee_style()
    arms = tuple(ARM_LABELS)
    ablated_arms = arms[1:]
    labels = ("Full", "w/o Price", "w/o BW", "w/o BR", "w/o Pruning")
    arm_colors = (
        ALGORITHM_COLORS["proposed"],
        "#7AA6C2",
        "#4F83A6",
        "#8B7BB5",
        "#7A8B99",
    )
    fig, axes = plt.subplots(
        1, 3, figsize=FIGSIZE_DOUBLE_COLUMN, layout="constrained",
        gridspec_kw={"width_ratios": (1.25, 1.0, 1.0)},
    )
    for ax in axes:
        setup_axes(ax)

    positions = np.arange(len(ablated_arms))
    width = 0.23
    metric_colors = ("#D55E00", "#0072B2", "#009E73")
    for index, ((metric, label), color) in enumerate(
        zip(BENEFIT_METRICS, metric_colors, strict=True)
    ):
        degradation = [
            -float(by_arm[arm][f"{metric}_change_vs_full_pct"])
            for arm in ablated_arms
        ]
        errors = [
            abs(float(by_arm[arm].get(f"{metric}_change_vs_full_pct_ci95") or 0.0))
            for arm in ablated_arms
        ]
        axes[0].bar(
            positions + (index - 1) * width,
            degradation,
            width,
            yerr=errors,
            capsize=ERRORBAR_CAPSIZE,
            error_kw=BAR_ERROR_STYLE,
            color=color,
            alpha=0.9,
            label=label,
        )
    axes[0].axhline(0.0, color="#4B5563", linewidth=0.7)
    axes[0].set_xticks(positions, labels[1:], rotation=18, ha="right")
    axes[0].set_ylabel("Degradation from Full (%)")
    panel_title(axes[0], "a", "Benefit Contribution")
    axes[0].legend(loc="upper left", ncol=1, **LEGEND_STYLE)

    all_positions = np.arange(len(arms))
    energy_means: list[float] = []
    energy_errors: list[float] = []
    energy_available: list[bool] = []
    for arm in arms:
        value = by_arm[arm].get("energy_per_participating_vehicle_j_seed_mean", "")
        available = value not in ("", "nan", "NaN") and np.isfinite(float(value))
        energy_available.append(bool(available))
        energy_means.append(float(value) if available else 0.0)
        interval = by_arm[arm].get("energy_per_participating_vehicle_j_seed_ci95", "")
        energy_errors.append(
            float(interval)
            if available and interval not in ("", "nan", "NaN")
            else 0.0
        )
    axes[1].bar(
        all_positions,
        energy_means,
        yerr=energy_errors,
        capsize=ERRORBAR_CAPSIZE,
        error_kw=BAR_ERROR_STYLE,
        color=arm_colors,
    )
    for position, available in zip(all_positions, energy_available, strict=True):
        if not available:
            axes[1].text(position, 0.01, "N/A", rotation=90, ha="center", va="bottom")
    axes[1].set_xticks(all_positions, labels, rotation=18, ha="right")
    axes[1].set_ylabel("Energy per Participant (J)")
    panel_title(axes[1], "b", "Vehicle Energy")

    timing_means = [float(timing_by_arm[arm]["total_cycle_ms_mean"]) for arm in arms]
    timing_errors = [
        float(timing_by_arm[arm].get("total_cycle_ms_ci95") or 0.0) for arm in arms
    ]
    axes[2].bar(
        all_positions,
        timing_means,
        yerr=timing_errors,
        capsize=ERRORBAR_CAPSIZE,
        error_kw=BAR_ERROR_STYLE,
        color=arm_colors,
    )
    axes[2].set_xticks(all_positions, labels, rotation=18, ha="right")
    axes[2].set_ylabel("Warmed Cycle Time (ms)")
    panel_title(axes[2], "c", "Computational Cost")

    save_figure(fig, figure_stem)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
