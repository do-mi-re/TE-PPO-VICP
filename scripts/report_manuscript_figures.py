"""Render the nine manuscript experiment figures from formal experiment data.

Exactness label: implementation convenience only (report visualization).
This module reads validated aggregate/source CSVs and never evaluates or trains
a policy.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Callable, Iterable
from pathlib import Path

from src.evaluation.provenance import sha256_file
from src.evaluation.revision import load_revision_config, revision_roots
from src.evaluation.sweep import ALGORITHM_LABELS, write_sweep_csv
from src.plotting.revision_style import (
    ALGORITHM_COLORS,
    FIGSIZE_SINGLE_COLUMN,
    LEGEND_STYLE,
    algorithm_errorbar_style,
    algorithm_errorbar_values,
    algorithm_style,
    apply_ieee_style,
    errorbar_style,
    finite_or_none,
    optional_errorbar_values,
    save_figure,
    setup_axes,
)

MANUSCRIPT_STEMS = tuple(f"fig{index}" for index in range(1, 10))
FULL_STEMS = ("fig2_full", "fig3_full", "fig4_full", "fig5_full", "fig6_full", "fig9_full")
MANUSCRIPT_LEGEND_FONTSIZE = 6.8


def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def _mean(row: dict[str, str], metric: str) -> float:
    value = finite_or_none(row.get(f"{metric}_seed_mean"))
    if value is None:
        value = finite_or_none(row.get(f"{metric}_mean"))
    if value is None:
        raise ValueError(f"row has no finite mean for {metric}")
    return value


def _ci95(row: dict[str, str], metric: str) -> float | None:
    return finite_or_none(row.get(f"{metric}_seed_ci95"))


def _algorithm_label(algorithm: str) -> str:
    """Return the compact method label used by manuscript figures."""

    if algorithm == "proposed":
        return "TE-PPO"
    return ALGORITHM_LABELS.get(algorithm, algorithm)


def _require_grid(
    rows: Iterable[dict[str, str]], expected: Iterable[float], label: str
) -> None:
    actual = {float(row["sweep_value"]) for row in rows}
    target = {float(value) for value in expected}
    if actual != target:
        raise ValueError(f"{label} grid mismatch: expected {sorted(target)}, got {sorted(actual)}")


def _bandwidth_scale(rows: Iterable[dict[str, str]]) -> float:
    """Return the unit scale for summaries that may store bandwidth in Hz or MHz."""

    values = [float(row["sweep_value"]) for row in rows]
    if not values:
        raise ValueError("bandwidth source is empty")
    return 1e6 if max(values) > 1e5 else 1.0


def _legend(
    ax: object,
    algorithm_count: int,
    *,
    columns: int | None = None,
) -> None:
    """Render a compact row, expanding multi-entry legends to the axes width."""

    ncol = min(algorithm_count, columns or algorithm_count)
    legend_kwargs = dict(
        ncol=ncol,
        fontsize=MANUSCRIPT_LEGEND_FONTSIZE,
        columnspacing=0.22,
        handlelength=0.85,
        handletextpad=0.22,
        borderpad=0.25,
        **LEGEND_STYLE,
    )
    if algorithm_count == 1:
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            **legend_kwargs,
        )
        return
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.0),
        mode="expand",
        borderaxespad=0.0,
        **legend_kwargs,
    )


def render_convergence(
    rows: list[dict[str, str]],
    output: Path,
    *,
    algorithms: tuple[str, ...],
) -> list[dict[str, object]]:
    """Render Fig. 1 using mean welfare and one-standard-deviation bands."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_ieee_style()
    selected = [row for row in rows if row.get("series_kind") == "learning_curve"]
    available = {row["algorithm"] for row in selected}
    missing = set(algorithms) - available
    if missing:
        raise ValueError(f"convergence source is missing algorithms: {sorted(missing)}")
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_COLUMN)
    setup_axes(ax)
    source_rows: list[dict[str, object]] = []
    for algorithm in algorithms:
        series = sorted(
            (row for row in selected if row["algorithm"] == algorithm),
            key=lambda row: int(row["timestep"]),
        )
        steps = [int(row["timestep"]) for row in series]
        x = [step / 1000.0 for step in steps]
        means = [float(row["social_welfare_mean"]) for row in series]
        stds = [float(row["social_welfare_seed_std"]) for row in series]
        ax.plot(
            x,
            means,
            label=_algorithm_label(algorithm),
            markevery=max(1, len(x) // 7),
            **algorithm_style(algorithm),
        )
        ax.fill_between(
            x,
            [mean - std for mean, std in zip(means, stds, strict=True)],
            [mean + std for mean, std in zip(means, stds, strict=True)],
            color=ALGORITHM_COLORS[algorithm],
            alpha=0.12,
            linewidth=0,
            zorder=1,
        )
        source_rows.extend(
            {
                "algorithm": algorithm,
                "timestep": step,
                "plotted_timestep_thousands": plotted,
                "social_welfare_mean": mean,
                "social_welfare_seed_std": std,
                "train_seed_count": int(row["train_seed_count"]),
                "uncertainty": "seed_std",
            }
            for row, step, plotted, mean, std in zip(
                series, steps, x, means, stds, strict=True
            )
        )
    ax.set_xlabel(r"Training Timesteps ($\times 10^3$)")
    ax.set_ylabel("Social Welfare")
    _legend(ax, len(algorithms))
    fig.tight_layout()
    save_figure(fig, output)
    plt.close(fig)
    return source_rows


def render_algorithm_sweep(
    rows: list[dict[str, str]],
    output: Path,
    *,
    algorithms: tuple[str, ...],
    metric: str,
    xlabel: str,
    ylabel: str,
    x_transform: Callable[[float], float] = lambda value: value,
) -> list[dict[str, object]]:
    """Render a measured-grid algorithm comparison with train-seed 95% CIs."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_ieee_style()
    available = {row["algorithm"] for row in rows}
    missing = set(algorithms) - available
    if missing:
        raise ValueError(f"{output.name} source is missing algorithms: {sorted(missing)}")
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_COLUMN)
    setup_axes(ax)
    source_rows: list[dict[str, object]] = []
    for algorithm in algorithms:
        series = sorted(
            (row for row in rows if row["algorithm"] == algorithm),
            key=lambda row: float(row["sweep_value"]),
        )
        raw_x = [float(row["sweep_value"]) for row in series]
        x = [x_transform(value) for value in raw_x]
        y = [_mean(row, metric) for row in series]
        errors = [_ci95(row, metric) for row in series]
        ax.errorbar(
            x,
            y,
            yerr=algorithm_errorbar_values(algorithm, errors),
            label=_algorithm_label(algorithm),
            **algorithm_errorbar_style(algorithm),
        )
        source_rows.extend(
            {
                "algorithm": algorithm,
                "sweep_value": raw,
                "plotted_x": plotted,
                "metric": metric,
                "seed_mean": mean,
                "seed_ci95": "" if error is None else error,
                "seed_replicates": row.get(f"{metric}_seed_replicates", ""),
                "uncertainty": "none_stateless" if error is None else "seed_ci95",
            }
            for row, raw, plotted, mean, error in zip(
                series, raw_x, x, y, errors, strict=True
            )
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted({x_transform(float(row["sweep_value"])) for row in rows}))
    _legend(ax, len(algorithms))
    fig.tight_layout()
    save_figure(fig, output)
    plt.close(fig)
    return source_rows


def render_compression(
    rows: list[dict[str, str]], output: Path
) -> list[dict[str, object]]:
    """Render Fig. 7's paper-exact energy-component decomposition."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_ieee_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_COLUMN)
    setup_axes(ax)
    series_specs = (
        ("sensing_energy_j", "Sensing", "#6B7280", ":", "D"),
        ("computation_energy_j", "Computation", ALGORITHM_COLORS["jdls"], "--", "s"),
        ("transmission_energy_j", "Transmission", ALGORITHM_COLORS["mappo"], "-.", "^"),
        ("total_energy_j", "Total", ALGORITHM_COLORS["proposed"], "-", "*"),
    )
    ordered = sorted(rows, key=lambda row: float(row["sweep_value"]))
    x = [float(row["sweep_value"]) for row in ordered]
    source_rows: list[dict[str, object]] = []
    for metric, label, color, linestyle, marker in series_specs:
        y = [_mean(row, metric) for row in ordered]
        errors = [_ci95(row, metric) for row in ordered]
        ax.errorbar(
            x,
            y,
            yerr=optional_errorbar_values(errors),
            color=color,
            linestyle=linestyle,
            marker=marker,
            linewidth=2.0 if metric == "total_energy_j" else 1.2,
            markersize=6.4 if metric == "total_energy_j" else 3.6,
            label=label,
            **errorbar_style(color),
        )
        source_rows.extend(
            {
                "series": label,
                "metric": metric,
                "compression_ratio": value,
                "seed_mean": mean,
                "seed_ci95": "" if error is None else error,
                "seed_replicates": row.get(f"{metric}_seed_replicates", ""),
                "uncertainty": "none" if error is None else "seed_ci95",
            }
            for row, value, mean, error in zip(ordered, x, y, errors, strict=True)
        )
    ax.set_xlabel(r"Compression Ratio $\eta$")
    ax.set_ylabel("Energy (J)")
    ax.set_yscale("log")
    ax.set_xticks([value for value in (1, 5, 10, 20, 30, 40, 50) if value in set(x)])
    _legend(ax, len(series_specs))
    fig.tight_layout()
    save_figure(fig, output)
    plt.close(fig)
    return source_rows


def render_pricing(rows: list[dict[str, str]], output: Path) -> list[dict[str, object]]:
    """Render Fig. 8 with explicit price-cap markers."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_ieee_style()
    ordered = sorted(rows, key=lambda row: float(row["sweep_value"]))
    x = [float(row["sweep_value"]) for row in ordered]
    y = [_mean(row, "raw_rsu_reward") for row in ordered]
    errors = [_ci95(row, "raw_rsu_reward") for row in ordered]
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_COLUMN)
    setup_axes(ax)
    ax.errorbar(
        x,
        y,
        yerr=optional_errorbar_values(errors),
        label="TE-PPO",
        **algorithm_errorbar_style("proposed"),
    )
    capped = [
        (value, mean)
        for row, value, mean in zip(ordered, x, y, strict=True)
        if _mean(row, "price_cap_fraction") > 0.0
    ]
    if capped:
        ax.scatter(
            [point[0] for point in capped],
            [point[1] for point in capped],
            s=34,
            facecolors="none",
            edgecolors="#111827",
            linewidths=0.75,
            label="Budget Cap Active",
            zorder=7,
        )
    ax.axvline(1.0, color="#6B7280", linestyle=(0, (3, 2)), linewidth=0.9)
    ax.set_xlabel(r"Price Scaling Factor $\vartheta$")
    ax.set_ylabel("RSU Utility")
    preferred_ticks = (0.5, 0.7, 1.0, 1.3, 1.5)
    ax.set_xticks([value for value in preferred_ticks if value in set(x)])
    _legend(ax, 2 if capped else 1)
    fig.tight_layout()
    save_figure(fig, output)
    plt.close(fig)
    return [
        {
            "algorithm": "proposed",
            "price_scale": value,
            "raw_rsu_reward_seed_mean": mean,
            "raw_rsu_reward_seed_ci95": "" if error is None else error,
            "seed_replicates": row.get("raw_rsu_reward_seed_replicates", ""),
            "price_cap_fraction_seed_mean": _mean(row, "price_cap_fraction"),
            "uncertainty": "none" if error is None else "seed_ci95",
        }
        for row, value, mean, error in zip(ordered, x, y, errors, strict=True)
    ]


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def render_manuscript_figures(outputs: Path, cfg: dict[str, object]) -> list[dict[str, object]]:
    """Render all nine figures and return their auditable manifest rows."""

    contract = cfg["manuscript_figures"]  # type: ignore[index]
    scalability = cfg["scalability"]  # type: ignore[index]
    main_text = cfg["main_text_analysis"]  # type: ignore[index]
    output_dir = outputs / str(contract["output_subdir"])
    source_dir = output_dir / "source_data"
    convergence_path = outputs / "figures" / "figR_learning_efficiency_source.csv"
    vehicle_path = outputs / "scalability" / "vehicles" / "summary.csv"
    dynamic_path = outputs / "dynamic_robustness" / "summary.csv"
    cpu_path = outputs / "cpu_robustness" / "formal" / "summary.csv"
    compression_path = outputs / "main_text_analysis" / "compression" / "summary.csv"
    pricing_path = outputs / "main_text_analysis" / "pricing" / "summary.csv"

    convergence = _read(convergence_path)
    vehicles = _read(vehicle_path)
    dynamic = _read(dynamic_path)
    bandwidth = [row for row in dynamic if row.get("stress_experiment") == "rsu_bandwidth"]
    cpu = _read(cpu_path)
    compression = _read(compression_path)
    pricing = _read(pricing_path)
    _require_grid(vehicles, scalability["vehicle_grid"], "vehicle scalability")
    bandwidth_scale = _bandwidth_scale(bandwidth)
    _require_grid(
        (
            {**row, "sweep_value": str(float(row["sweep_value"]) / bandwidth_scale)}
            for row in bandwidth
        ),
        (float(value) / 1e6 for value in scalability["rsu_bandwidth_grid_hz"]),
        "bandwidth robustness",
    )
    expected_cpu = [
        (float(low) + float(high)) / 2e9
        for low, high in main_text["cpu_capacity_ranges_cycles_per_s"]
    ]
    _require_grid(cpu, expected_cpu, "CPU robustness")
    _require_grid(compression, main_text["compression_ratio_grid"], "compression")
    _require_grid(pricing, main_text["price_scale_grid"], "pricing")

    paper_algorithms = tuple(contract["paper_algorithms"])
    full_algorithms = tuple(contract["full_algorithms"])
    convergence_algorithms = tuple(contract["convergence_algorithms"])
    definitions = (
        ("fig1", "Training Convergence", convergence_path, lambda stem: render_convergence(convergence, stem, algorithms=convergence_algorithms)),
        ("fig2", "Social Welfare versus Number of Vehicles", vehicle_path, lambda stem: render_algorithm_sweep(vehicles, stem, algorithms=paper_algorithms, metric="social_welfare", xlabel="Number of Vehicles", ylabel="Social Welfare")),
        ("fig3", "Social Welfare versus System Bandwidth", dynamic_path, lambda stem: render_algorithm_sweep(bandwidth, stem, algorithms=paper_algorithms, metric="social_welfare", xlabel=r"System Bandwidth $B_0$ (MHz)", ylabel="Social Welfare", x_transform=lambda value: value / bandwidth_scale)),
        ("fig4", "Social Welfare versus CPU Capacity", cpu_path, lambda stem: render_algorithm_sweep(cpu, stem, algorithms=paper_algorithms, metric="social_welfare", xlabel=r"Average $F_i$ (Gcycles/s)", ylabel="Social Welfare")),
        ("fig5", "RSU Utility versus Number of Vehicles", vehicle_path, lambda stem: render_algorithm_sweep(vehicles, stem, algorithms=paper_algorithms, metric="raw_rsu_reward", xlabel="Number of Vehicles", ylabel="RSU Utility")),
        ("fig6", "RSU Utility versus System Bandwidth", dynamic_path, lambda stem: render_algorithm_sweep(bandwidth, stem, algorithms=paper_algorithms, metric="raw_rsu_reward", xlabel=r"System Bandwidth $B_0$ (MHz)", ylabel="RSU Utility", x_transform=lambda value: value / bandwidth_scale)),
        ("fig7", "Compression-Energy Trade-Off", compression_path, lambda stem: render_compression(compression, stem)),
        ("fig8", "Pricing Sensitivity", pricing_path, lambda stem: render_pricing(pricing, stem)),
        ("fig9", "Sensing Quality versus Number of Vehicles", vehicle_path, lambda stem: render_algorithm_sweep(vehicles, stem, algorithms=paper_algorithms, metric="perception_quality_total", xlabel="Number of Vehicles", ylabel="Effective Sensing Quality")),
    )
    full_definitions = (
        ("fig2_full", "Social Welfare versus Number of Vehicles", vehicle_path, vehicles, "social_welfare", "Number of Vehicles", "Social Welfare", lambda value: value),
        ("fig3_full", "Social Welfare versus System Bandwidth", dynamic_path, bandwidth, "social_welfare", r"System Bandwidth $B_0$ (MHz)", "Social Welfare", lambda value: value / bandwidth_scale),
        ("fig4_full", "Social Welfare versus CPU Capacity", cpu_path, cpu, "social_welfare", r"Average $F_i$ (Gcycles/s)", "Social Welfare", lambda value: value),
        ("fig5_full", "RSU Utility versus Number of Vehicles", vehicle_path, vehicles, "raw_rsu_reward", "Number of Vehicles", "RSU Utility", lambda value: value),
        ("fig6_full", "RSU Utility versus System Bandwidth", dynamic_path, bandwidth, "raw_rsu_reward", r"System Bandwidth $B_0$ (MHz)", "RSU Utility", lambda value: value / bandwidth_scale),
        ("fig9_full", "Sensing Quality versus Number of Vehicles", vehicle_path, vehicles, "perception_quality_total", "Number of Vehicles", "Effective Sensing Quality", lambda value: value),
    )
    manifest: list[dict[str, object]] = []

    def record(
        stem: str,
        title: str,
        input_path: Path,
        plotted: list[dict[str, object]],
    ) -> None:
        source_path = source_dir / f"{stem}.csv"
        write_sweep_csv(source_path, plotted)
        png_path = output_dir / f"{stem}.png"
        pdf_path = output_dir / f"{stem}.pdf"
        manifest.append(
            {
                "figure": stem,
                "title": title,
                "input_csv": _relative(input_path, outputs),
                "input_csv_sha256": sha256_file(input_path),
                "source_csv": _relative(source_path, outputs),
                "source_csv_sha256": sha256_file(source_path),
                "png_sha256": sha256_file(png_path),
                "pdf_sha256": sha256_file(pdf_path),
                "contract_version": int(contract["contract_version"]),
                "exactness_label": "implementation convenience only",
                "layout": "single_column",
                "legend_layout": "single_row_adaptive",
                "figure_width_in": FIGSIZE_SINGLE_COLUMN[0],
                "figure_height_in": FIGSIZE_SINGLE_COLUMN[1],
            }
        )

    for stem, title, input_path, renderer in definitions:
        record(stem, title, input_path, renderer(output_dir / stem))
    for stem, title, input_path, rows, metric, xlabel, ylabel, transform in full_definitions:
        plotted = render_algorithm_sweep(
            rows,
            output_dir / stem,
            algorithms=full_algorithms,
            metric=metric,
            xlabel=xlabel,
            ylabel=ylabel,
            x_transform=transform,
        )
        record(stem, title, input_path, plotted)
    write_sweep_csv(output_dir / "manifest.csv", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--outputs-root", type=Path)
    args = parser.parse_args(argv)
    cfg = load_revision_config()
    outputs = args.outputs_root or revision_roots(cfg)[1]
    try:
        manifest = render_manuscript_figures(outputs, cfg)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        print(f"error: {exc}")
        return 1
    print(f"wrote {len(manifest)} manuscript figure variants to {outputs / str(cfg['manuscript_figures']['output_subdir'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
