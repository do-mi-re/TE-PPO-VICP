"""Shared IEEE-style presentation helpers for experiment figures.

Exactness label: implementation convenience only (report visualization and
derived-artifact hygiene).  This module does not change model, training, or
statistical semantics.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

ALGORITHM_ORDER = (
    "proposed",
    "ebas",
    "jdls",
    "mappo",
    "dbas",
    "rdus",
    "spis",
)

ALGORITHM_COLORS = {
    "proposed": "#D55E00",
    "ebas": "#CC79A7",
    "jdls": "#0072B2",
    "mappo": "#009E73",
    "dbas": "#E69F00",
    "rdus": "#8C6D31",
    "spis": "#6B7280",
}

ALGORITHM_MARKERS = {
    "proposed": "*",
    "ebas": "o",
    "jdls": "s",
    "mappo": "^",
    "dbas": "D",
    "rdus": "P",
    "spis": "v",
}

ALGORITHM_LINESTYLES: dict[str, Any] = {
    "proposed": "-",
    "ebas": (0, (5, 2)),
    "jdls": "--",
    "mappo": "-.",
    "dbas": ":",
    "rdus": (0, (1, 1)),
    "spis": (0, (3, 1, 1, 1)),
}

FIGURE_DPI = 300
FIGSIZE_SINGLE_COLUMN = (3.5, 2.55)
FIGSIZE_DOUBLE_COLUMN = (7.16, 3.05)
FIGSIZE_DOUBLE_COLUMN_TALL = (7.16, 5.15)

LEGEND_STYLE: dict[str, Any] = {
    "frameon": True,
    "framealpha": 0.94,
    "edgecolor": "#D1D5DB",
    "fancybox": False,
}

ERRORBAR_CAPSIZE = 3.5
ERRORBAR_STYLE: dict[str, Any] = {
    "capsize": ERRORBAR_CAPSIZE,
    "capthick": 1.0,
    "elinewidth": 1.0,
    "barsabove": True,
}
BAR_ERROR_STYLE: dict[str, Any] = {
    "ecolor": "#374151",
    "elinewidth": 1.0,
    "capthick": 1.0,
}


def finite_or_none(value: object) -> float | None:
    """Return a finite float, preserving missing/non-finite report values.

    Missing uncertainty is not zero uncertainty.  Reporters use this helper so
    stateless methods do not acquire fabricated zero-width confidence bars.
    """

    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def optional_errorbar_values(
    values: Iterable[float | None],
) -> list[float] | None:
    """Convert optional CI values to Matplotlib ``yerr`` semantics.

    ``None`` means that no interval is available.  An entirely missing series
    returns ``None`` so Matplotlib draws no caps.  For a partially missing
    series, NaN suppresses only the unavailable points without inventing a
    zero-width interval.
    """

    converted: list[float] = []
    has_interval = False
    for value in values:
        if value is None:
            converted.append(float("nan"))
            continue
        number = float(value)
        if not math.isfinite(number):
            converted.append(float("nan"))
            continue
        if number < 0.0:
            raise ValueError(f"confidence-interval half-width must be nonnegative: {number}")
        converted.append(number)
        has_interval = True
    return converted if has_interval else None


def algorithm_errorbar_values(
    algorithm: str,
    values: Iterable[float | None],
) -> list[float] | None:
    """Validate the paper-wide train-seed CI contract for one algorithm."""

    intervals = list(values)
    if algorithm == "dbas":
        if any(value is not None for value in intervals):
            raise ValueError("stateless DBAS must not carry a train-seed interval")
    elif any(value is None for value in intervals):
        raise ValueError(f"{algorithm} is missing a train-seed confidence interval")
    return optional_errorbar_values(intervals)


def apply_ieee_style() -> None:
    """Install compact, PDF-safe defaults for experiment figures."""

    matplotlib.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "legend.fontsize": 7.2,
            "legend.title_fontsize": 7.5,
            "lines.linewidth": 1.2,
            "lines.markersize": 4.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            # Keep the physical 7.16-inch double-column contract exact.  A
            # tight bounding box would silently shrink the PDF media box.
            "savefig.bbox": None,
        }
    )


def ordered_algorithms(algorithms: Iterable[str]) -> list[str]:
    """Return algorithms in the paper-wide visual order."""

    values = set(algorithms)
    known = [algorithm for algorithm in ALGORITHM_ORDER if algorithm in values]
    return known + sorted(values.difference(known))


def algorithm_style(algorithm: str) -> dict[str, Any]:
    """Return a consistent line/marker style, emphasizing TE-PPO and EBAS."""

    if algorithm == "proposed":
        linewidth, markersize, zorder, alpha = 2.15, 6.4, 5, 1.0
    elif algorithm == "ebas":
        linewidth, markersize, zorder, alpha = 1.65, 4.4, 4, 1.0
    else:
        linewidth, markersize, zorder, alpha = 1.15, 3.6, 2, 1.0
    return {
        "color": ALGORITHM_COLORS.get(algorithm, "#374151"),
        "marker": ALGORITHM_MARKERS.get(algorithm, "o"),
        "linestyle": ALGORITHM_LINESTYLES.get(algorithm, "-"),
        "linewidth": linewidth,
        "markersize": markersize,
        "alpha": alpha,
        "zorder": zorder,
    }


def _darken_color(color: str, factor: float = 0.72) -> tuple[float, float, float]:
    """Return a darker RGB error-bar color without changing data semantics."""

    from matplotlib.colors import to_rgb

    return tuple(channel * factor for channel in to_rgb(color))


def errorbar_style(color: str) -> dict[str, Any]:
    """Return print-visible error-bar kwargs for a curve of ``color``."""

    return {**ERRORBAR_STYLE, "ecolor": _darken_color(color)}


def algorithm_errorbar_style(algorithm: str) -> dict[str, Any]:
    """Combine the shared algorithm identity with visible uncertainty styling."""

    style = algorithm_style(algorithm)
    return {**style, **errorbar_style(str(style["color"]))}


def setup_axes(ax: Any, *, grid_axis: str = "y") -> None:
    """Apply restrained axes styling suitable for IEEE print layouts."""

    ax.set_axisbelow(True)
    ax.grid(
        axis=grid_axis,
        color="#D1D5DB",
        linewidth=0.55,
        linestyle=(0, (2, 2)),
        alpha=0.8,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_title(ax: Any, letter: str, title: str) -> None:
    """Set a compact panel title with a stable subfigure identifier."""

    ax.set_title(f"({letter}) {title}", loc="left", pad=3.0, fontweight="semibold")


def save_figure(fig: Any, stem: Path) -> None:
    """Save matching high-resolution raster and vector report artifacts."""

    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=FIGURE_DPI)
    fig.savefig(stem.with_suffix(".pdf"))
