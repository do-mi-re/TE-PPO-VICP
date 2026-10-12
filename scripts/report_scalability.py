"""Render scalability and dynamic-condition figures from experiment summaries.

Exactness label: implementation convenience only (report visualization).  The
reporter preserves the maintained seed-level means and confidence intervals.
Paper-facing figures show the configured five methods; `_full` companion
figures preserve all seven methods for audit and supplementary material.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.evaluation.sweep import ALGORITHM_LABELS
from src.plotting.revision_style import (
    FIGSIZE_DOUBLE_COLUMN,
    FIGSIZE_DOUBLE_COLUMN_TALL,
    LEGEND_STYLE,
    algorithm_errorbar_style,
    algorithm_errorbar_values,
    apply_ieee_style,
    finite_or_none,
    ordered_algorithms,
    panel_title,
    save_figure,
    setup_axes,
)

METRICS = (
    ("social_welfare", "Social Welfare"),
    ("raw_rsu_reward", "RSU Utility"),
    ("task_success_rate", "Task Success Rate"),
    ("perception_quality_total", "Sensing Quality"),
)
DYNAMIC_ORDER = ("occlusion", "path_loss", "rsu_bandwidth")
PAPER_ALGORITHMS = ("proposed", "ebas", "jdls", "mappo", "dbas")


def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return list(csv.DictReader(path.open(encoding="utf-8", newline="")))


def _mean(row: dict[str, str], metric: str) -> float:
    """Use the across-training-seed mean explicitly when it is available."""

    value = finite_or_none(row.get(f"{metric}_seed_mean"))
    if value is None:
        value = finite_or_none(row.get(f"{metric}_mean"))
    if value is None:
        raise ValueError(f"row has no finite mean for {metric}")
    return value


def _ci95(row: dict[str, str], metric: str) -> float | None:
    return finite_or_none(row.get(f"{metric}_seed_ci95"))


def _figure_algorithms(
    rows: list[dict[str, str]],
    requested: tuple[str, ...] | None,
) -> list[str]:
    available = ordered_algorithms(row["algorithm"] for row in rows)
    if requested is None:
        return available
    return [algorithm for algorithm in requested if algorithm in available]


def render_scale(
    rows: list[dict[str, str]],
    axis: str,
    output: Path,
    *,
    algorithms_to_plot: tuple[str, ...] | None = PAPER_ALGORITHMS,
) -> None:
    """Render a compact four-panel scale sweep with shared visual semantics."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_ieee_style()
    algorithms = _figure_algorithms(rows, algorithms_to_plot)
    x_values = sorted({float(row["sweep_value"]) for row in rows})
    fig, axes = plt.subplots(
        2, 2, figsize=FIGSIZE_DOUBLE_COLUMN_TALL, sharex=True
    )
    for panel, (ax, (metric, label)) in enumerate(
        zip(axes.flat, METRICS, strict=True)
    ):
        setup_axes(ax)
        for algorithm in algorithms:
            series = sorted(
                (row for row in rows if row["algorithm"] == algorithm),
                key=lambda row: float(row["sweep_value"]),
            )
            errors = [_ci95(row, metric) for row in series]
            ax.errorbar(
                [float(row["sweep_value"]) for row in series],
                [_mean(row, metric) for row in series],
                yerr=algorithm_errorbar_values(algorithm, errors),
                label=ALGORITHM_LABELS.get(algorithm, algorithm),
                **algorithm_errorbar_style(algorithm),
            )
        panel_title(ax, "abcd"[panel], label)
        ax.set_xticks(x_values)
        if metric == "task_success_rate":
            ax.set_ylim(0.0, 1.05)
    fig.supxlabel("Number of Vehicles" if axis == "vehicles" else "Number of Tasks")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=len(algorithms),
        columnspacing=0.8,
        handlelength=1.8,
        **LEGEND_STYLE,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    save_figure(fig, output)
    plt.close(fig)


def _dynamic_panel_data(
    subset: list[dict[str, str]], experiment: str, algorithm: str, index: int, count: int
) -> tuple[list[float], list[float], list[float | None]]:
    series = [row for row in subset if row["algorithm"] == algorithm]
    if experiment == "occlusion":
        order = {"default": 0, "severe": 1}
        series.sort(key=lambda row: order.get(row.get("stress_level", ""), 99))
        offset = 0.34 * (index / max(1, count - 1) - 0.5)
        x = [
            float(order.get(row.get("stress_level", ""), position)) + offset
            for position, row in enumerate(series)
        ]
    else:
        series.sort(key=lambda row: float(row["sweep_value"]))
        scale = (
            1e6
            if experiment == "rsu_bandwidth"
            and max(float(row["sweep_value"]) for row in series) > 1e5
            else 1.0
        )
        x = [float(row["sweep_value"]) / scale for row in series]
    return (
        x,
        [_mean(row, "social_welfare") for row in series],
        [_ci95(row, "social_welfare") for row in series],
    )


def render_dynamic(
    rows: list[dict[str, str]],
    output_dir: Path,
    *,
    output_stem: str = "figR_dynamic_robustness",
    algorithms_to_plot: tuple[str, ...] | None = PAPER_ALGORITHMS,
) -> None:
    """Render all dynamic-condition tests as one coherent three-panel figure."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_ieee_style()
    present = {row["stress_experiment"] for row in rows}
    experiments = [name for name in DYNAMIC_ORDER if name in present]
    if experiments != list(DYNAMIC_ORDER):
        raise ValueError(f"dynamic summary must contain {DYNAMIC_ORDER}, got {sorted(present)}")
    algorithms = _figure_algorithms(rows, algorithms_to_plot)
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_DOUBLE_COLUMN, sharey=True)
    titles = {
        "occlusion": "Occlusion Severity",
        "path_loss": "Path-Loss Exponent",
        "rsu_bandwidth": "RSU Bandwidth",
    }
    xlabels = {
        "occlusion": "Occlusion Scenario",
        "path_loss": r"$\alpha$",
        "rsu_bandwidth": r"$B_0$ (MHz)",
    }
    for panel, (ax, experiment) in enumerate(zip(axes, experiments, strict=True)):
        setup_axes(ax)
        subset = [row for row in rows if row["stress_experiment"] == experiment]
        for index, algorithm in enumerate(algorithms):
            x, y, error = _dynamic_panel_data(
                subset, experiment, algorithm, index, len(algorithms)
            )
            style = algorithm_errorbar_style(algorithm)
            if experiment == "occlusion":
                style["linestyle"] = "none"
            ax.errorbar(
                x,
                y,
                yerr=algorithm_errorbar_values(algorithm, error),
                label=ALGORITHM_LABELS.get(algorithm, algorithm),
                **style,
            )
        panel_title(ax, "abc"[panel], titles[experiment])
        ax.set_xlabel(xlabels[experiment])
        if experiment == "occlusion":
            ax.set_xticks((0, 1), ("Default\n$U[0,1]$", "Severe\n$U[0.5,0.9]$"))
            ax.set_xlim(-0.45, 1.45)
        elif experiment == "path_loss":
            values = sorted({float(row["sweep_value"]) for row in subset})
            # Seven measured cells are shown, while alternating tick labels
            # keep the narrow IEEE panel legible at print size.
            ax.set_xticks(values[::2])
        else:
            raw_values = sorted({float(row["sweep_value"]) for row in subset})
            scale = 1e6 if max(raw_values) > 1e5 else 1.0
            values = [value / scale for value in raw_values]
            ax.set_xticks([value for value in values if value != 15.0])
    axes[0].set_ylabel("Social Welfare")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=len(algorithms),
        columnspacing=0.75,
        handlelength=1.7,
        **LEGEND_STYLE,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output_dir / output_stem)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--outputs-root", type=Path, default=Path("outputs/revision_2026_v5")
    )
    args = parser.parse_args(argv)
    figures = args.outputs_root / "figures"
    try:
        for axis in ("vehicles", "tasks"):
            rows = _read(args.outputs_root / "scalability" / axis / "summary.csv")
            render_scale(rows, axis, figures / f"figR_scalability_{axis}")
            render_scale(
                rows,
                axis,
                figures / f"figR_scalability_{axis}_full",
                algorithms_to_plot=None,
            )
        dynamic = _read(args.outputs_root / "dynamic_robustness" / "summary.csv")
        render_dynamic(dynamic, figures)
        render_dynamic(
            dynamic,
            figures,
            output_stem="figR_dynamic_robustness_full",
            algorithms_to_plot=None,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}")
        return 1
    print(f"wrote scalability figures to {figures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
