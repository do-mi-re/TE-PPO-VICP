"""Render the configured resource and mechanism experiment figures.

Exactness label: implementation convenience only (report visualization).
All statistics are consumed from validated analysis summaries; this script
does not evaluate or train a policy.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.evaluation.revision import load_revision_config, revision_roots
from src.evaluation.sweep import ALGORITHM_LABELS, write_sweep_csv
from src.plotting.revision_style import (
    ALGORITHM_COLORS,
    FIGSIZE_DOUBLE_COLUMN,
    LEGEND_STYLE,
    algorithm_errorbar_style,
    algorithm_errorbar_values,
    apply_ieee_style,
    errorbar_style,
    finite_or_none,
    optional_errorbar_values,
    ordered_algorithms,
    panel_title,
    save_figure,
    setup_axes,
)


def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return list(csv.DictReader(path.open(encoding="utf-8", newline="")))


def _mean(row: dict[str, str], metric: str) -> float:
    value = finite_or_none(row.get(f"{metric}_seed_mean"))
    if value is None:
        value = finite_or_none(row.get(f"{metric}_mean"))
    if value is None:
        raise ValueError(f"row has no finite mean for {metric}")
    return value


def _ci95(row: dict[str, str], metric: str) -> float | None:
    return finite_or_none(row.get(f"{metric}_seed_ci95"))


def _algorithms(rows: list[dict[str, str]], requested: tuple[str, ...] | None) -> list[str]:
    available = ordered_algorithms(row["algorithm"] for row in rows)
    return available if requested is None else [name for name in requested if name in available]


def render_resource_sensitivity(
    cpu_rows: list[dict[str, str]],
    bandwidth_rows: list[dict[str, str]],
    output: Path,
    *,
    algorithms_to_plot: tuple[str, ...] | None,
) -> list[dict[str, object]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_ieee_style()
    algorithms = _algorithms(cpu_rows, algorithms_to_plot)
    legend_rows = 2 if len(algorithms) > 5 else 1
    figsize = (
        FIGSIZE_DOUBLE_COLUMN[0],
        FIGSIZE_DOUBLE_COLUMN[1] + (0.35 if legend_rows == 2 else 0.0),
    )
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    panels = (
        (axes[0], cpu_rows, "social_welfare", "CPU Capacity", r"Average $F_i$ (Gcycles/s)"),
        (axes[1], cpu_rows, "raw_rsu_reward", "CPU Capacity", r"Average $F_i$ (Gcycles/s)"),
        (
            axes[2], bandwidth_rows, "raw_rsu_reward", "System Bandwidth",
            r"$B_0$ (MHz)",
        ),
    )
    source_rows: list[dict[str, object]] = []
    for panel_index, (ax, rows, metric, title, xlabel) in enumerate(panels):
        setup_axes(ax)
        for algorithm in algorithms:
            series = sorted(
                (row for row in rows if row["algorithm"] == algorithm),
                key=lambda row: float(row["sweep_value"]),
            )
            if not series:
                continue
            scale = (
                1e6
                if title == "System Bandwidth"
                and max(float(row["sweep_value"]) for row in series) > 1e5
                else 1.0
            )
            x = [float(row["sweep_value"]) / scale for row in series]
            y = [_mean(row, metric) for row in series]
            error = [_ci95(row, metric) for row in series]
            ax.errorbar(
                x, y, yerr=algorithm_errorbar_values(algorithm, error),
                label=ALGORITHM_LABELS.get(algorithm, algorithm),
                **algorithm_errorbar_style(algorithm),
            )
            for row, x_value, y_value, ci_value in zip(series, x, y, error, strict=True):
                source_rows.append({
                    "panel": "abc"[panel_index],
                    "panel_title": title,
                    "metric": metric,
                    "algorithm": algorithm,
                    "sweep_value": x_value,
                    "seed_mean": y_value,
                    "seed_ci95": "" if ci_value is None else ci_value,
                    "seed_replicates": row.get(f"{metric}_seed_replicates", ""),
                    "uncertainty": (
                        "none_stateless" if ci_value is None else "seed_ci95"
                    ),
                })
        panel_title(ax, "abc"[panel_index], title)
        ax.set_xlabel(xlabel)
        tick_scale = (
            1e6
            if title == "System Bandwidth"
            and max(float(row["sweep_value"]) for row in rows) > 1e5
            else 1.0
        )
        ax.set_xticks(
            sorted({float(row["sweep_value"]) / tick_scale for row in rows})
        )
        ax.set_ylabel({
            "social_welfare": "Social Welfare",
            "raw_rsu_reward": "RSU Utility",
            "task_success_rate": "Task Success Rate",
        }[metric])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.01),
        ncol=min(len(algorithms), 5), columnspacing=0.75, handlelength=1.7,
        **LEGEND_STYLE,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.83 if legend_rows == 2 else 0.90))
    save_figure(fig, output)
    plt.close(fig)
    return source_rows


def render_mechanism_analysis(
    compression_rows: list[dict[str, str]],
    pricing_rows: list[dict[str, str]],
    output: Path,
) -> list[dict[str, object]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_ieee_style()
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE_COLUMN)
    source_rows: list[dict[str, object]] = []

    ax = axes[0]
    setup_axes(ax)
    energy_series = (
        ("sensing_energy_j", "Sensing", "#6B7280", ":", "D"),
        ("computation_energy_j", "Computation", ALGORITHM_COLORS["jdls"], "--", "s"),
        ("transmission_energy_j", "Transmission", ALGORITHM_COLORS["mappo"], "-.", "^"),
        ("total_energy_j", "Total", ALGORITHM_COLORS["proposed"], "-", "*"),
    )
    ordered_compression = sorted(compression_rows, key=lambda row: float(row["sweep_value"]))
    x = [float(row["sweep_value"]) for row in ordered_compression]
    for metric, label, color, linestyle, marker in energy_series:
        y = [_mean(row, metric) for row in ordered_compression]
        error = [_ci95(row, metric) for row in ordered_compression]
        ax.errorbar(
            x, y, yerr=optional_errorbar_values(error),
            color=color, linestyle=linestyle, marker=marker,
            linewidth=2.0 if metric == "total_energy_j" else 1.25,
            markersize=6.4 if metric == "total_energy_j" else 3.6,
            label=label,
            **errorbar_style(color),
        )
        for row, x_value, y_value, ci_value in zip(
            ordered_compression, x, y, error, strict=True
        ):
            source_rows.append({
                "panel": "a", "metric": metric, "series": label,
                "sweep_value": x_value, "seed_mean": y_value,
                "seed_ci95": "" if ci_value is None else ci_value,
                "seed_replicates": row.get(f"{metric}_seed_replicates", ""),
                "uncertainty": "none" if ci_value is None else "seed_ci95",
            })
    panel_title(ax, "a", "Compression-Energy Trade-Off")
    ax.set_xlabel(r"Compression Ratio $\eta$")
    ax.set_ylabel("Energy (J)")
    preferred_eta_ticks = (1.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0)
    ax.set_xticks([value for value in preferred_eta_ticks if value in set(x)])
    ax.set_yscale("log")
    ax.legend(loc="best", ncol=2, **LEGEND_STYLE)

    ax = axes[1]
    setup_axes(ax)
    ordered_pricing = sorted(pricing_rows, key=lambda row: float(row["sweep_value"]))
    theta = [float(row["sweep_value"]) for row in ordered_pricing]
    utility = [_mean(row, "raw_rsu_reward") for row in ordered_pricing]
    error = [_ci95(row, "raw_rsu_reward") for row in ordered_pricing]
    ax.errorbar(
        theta, utility, yerr=optional_errorbar_values(error), label="TE-PPO",
        **algorithm_errorbar_style("proposed"),
    )
    capped = [
        (x_value, y_value)
        for row, x_value, y_value in zip(ordered_pricing, theta, utility, strict=True)
        if _mean(row, "price_cap_fraction") > 0.0
    ]
    if capped:
        ax.scatter(
            [value[0] for value in capped], [value[1] for value in capped],
            s=38, facecolors="none", edgecolors="#111827", linewidths=0.8,
            label="Budget Cap Active", zorder=7,
        )
    ax.axvline(1.0, color="#6B7280", linestyle=(0, (3, 2)), linewidth=0.9)
    panel_title(ax, "b", "Pricing Sensitivity")
    ax.set_xlabel(r"Price Scaling Factor $\vartheta$")
    ax.set_ylabel("RSU Utility")
    preferred_theta_ticks = (0.5, 0.7, 1.0, 1.3, 1.5)
    ax.set_xticks([value for value in preferred_theta_ticks if value in set(theta)])
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda index: labels[index] != "TE-PPO")
    ax.legend(
        [handles[index] for index in order],
        [labels[index] for index in order],
        loc="best",
        **LEGEND_STYLE,
    )
    for row, x_value, y_value, ci_value in zip(
        ordered_pricing, theta, utility, error, strict=True
    ):
        source_rows.append({
            "panel": "b", "metric": "raw_rsu_reward", "series": "TE-PPO",
            "sweep_value": x_value, "seed_mean": y_value,
            "seed_ci95": "" if ci_value is None else ci_value,
            "seed_replicates": row.get("raw_rsu_reward_seed_replicates", ""),
            "uncertainty": "none" if ci_value is None else "seed_ci95",
            "price_cap_fraction_seed_mean": _mean(row, "price_cap_fraction"),
        })
    fig.tight_layout()
    save_figure(fig, output)
    plt.close(fig)
    return source_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--outputs-root", type=Path)
    args = parser.parse_args(argv)
    cfg = load_revision_config()
    outputs = args.outputs_root or revision_roots(cfg)[1]
    analysis_root = outputs / "main_text_analysis"
    figures = outputs / "figures"
    try:
        cpu = _read(outputs / "cpu_robustness" / "formal" / "summary.csv")
        dynamic = _read(outputs / "dynamic_robustness" / "summary.csv")
        bandwidth = [
            row for row in dynamic
            if row.get("stress_experiment") == "rsu_bandwidth"
        ]
        if not bandwidth:
            raise ValueError("dynamic robustness summary has no RSU-bandwidth rows")
        compression = _read(analysis_root / "compression" / "summary.csv")
        pricing = _read(analysis_root / "pricing" / "summary.csv")
        paper_algorithms = tuple(cfg["main_text_analysis"]["paper_algorithms"])
        resource_source = render_resource_sensitivity(
            cpu, bandwidth, figures / "figR_resource_sensitivity",
            algorithms_to_plot=paper_algorithms,
        )
        render_resource_sensitivity(
            cpu, bandwidth, figures / "figR_resource_sensitivity_full",
            algorithms_to_plot=None,
        )
        mechanism_source = render_mechanism_analysis(
            compression, pricing, figures / "figR_mechanism_analysis"
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"error: {exc}")
        return 1
    write_sweep_csv(figures / "figR_resource_sensitivity_source.csv", resource_source)
    write_sweep_csv(figures / "figR_mechanism_analysis_source.csv", mechanism_source)
    print(f"wrote formal main-text figures to {figures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
