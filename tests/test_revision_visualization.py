"""Regression tests for paper-facing visual reports.

Exactness label: implementation convenience only (report visualization).
"""

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from scripts.report_convergence import (
    render_learning_efficiency_figure,
    render_reward_convergence_figure,
)
from scripts.report_scalability import (
    PAPER_ALGORITHMS,
    _dynamic_panel_data,
    _figure_algorithms,
    render_dynamic,
    render_scale,
)
from src.plotting.revision_style import (
    BAR_ERROR_STYLE,
    ERRORBAR_CAPSIZE,
    algorithm_errorbar_style,
    algorithm_errorbar_values,
    algorithm_style,
    optional_errorbar_values,
    ordered_algorithms,
)


def _dynamic_row(
    algorithm: str,
    experiment: str,
    level: str,
    value: float,
    mean: float,
) -> dict[str, str]:
    return {
        "algorithm": algorithm,
        "stress_experiment": experiment,
        "stress_level": level,
        "sweep_value": str(value),
        "social_welfare_seed_mean": str(mean),
        "social_welfare_seed_ci95": "0.2",
    }


def test_algorithm_order_and_emphasis_are_stable() -> None:
    assert ordered_algorithms(("spis", "jdls", "proposed", "ebas")) == [
        "proposed",
        "ebas",
        "jdls",
        "spis",
    ]
    assert algorithm_style("proposed")["linewidth"] > algorithm_style("ebas")["linewidth"]
    assert algorithm_style("ebas")["linewidth"] > algorithm_style("jdls")["linewidth"]


def test_reward_convergence_uses_common_metric_and_seed_std(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    steps = np.array([3_072, 6_144, 9_216])
    curves = {
        algorithm: [
            (steps, np.array([1.0, 2.0, 3.0]) + index, None, tmp_path),
            (steps, np.array([1.2, 2.2, 3.2]) + index, None, tmp_path),
        ]
        for index, algorithm in enumerate(("proposed", "jdls", "mappo", "ebas"))
    }
    stem = tmp_path / "figR_reward_convergence"
    monkeypatch.setattr(plt, "close", lambda _figure: None)
    render_reward_convergence_figure(
        curves,
        stem,
        reward_schema="common_evaluation_reward_v1",
        source_keys={
            (algorithm, seed): "common_evaluation_reward"
            for algorithm in curves
            for seed in (1, 2)
        },
        source_bundles={algorithm: f"digest-{algorithm}" for algorithm in curves},
    )

    assert stem.with_suffix(".png").exists()
    assert stem.with_suffix(".pdf").exists()
    with stem.with_name(f"{stem.name}_source.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert {row["algorithm"] for row in rows} == {"proposed", "jdls", "mappo"}
    assert {row["reward_schema"] for row in rows} == {
        "common_evaluation_reward_v1"
    }
    assert {row["uncertainty"] for row in rows} == {"seed_std"}
    assert {int(row["train_seed_count"]) for row in rows} == {2}
    ax = plt.gcf().axes[0]
    assert ax.get_xlabel() == "Training Timesteps"
    assert ax.get_ylabel() == "Training Reward"
    assert [text.get_text() for text in ax.get_legend().get_texts()] == [
        "TE-PPO (Proposed)", "JDLS", "MAPS"
    ]
    plt.gcf().clear()


def test_algorithm_errorbars_are_visible_above_markers() -> None:
    for algorithm in ("proposed", "ebas", "jdls", "mappo", "dbas", "rdus", "spis"):
        style = algorithm_errorbar_style(algorithm)
        assert style["barsabove"] is True
        assert style["capsize"] == ERRORBAR_CAPSIZE
        assert style["capthick"] >= 1.0
        assert style["elinewidth"] >= 1.0
        assert 2 * style["capsize"] > style["markersize"]
        assert style["alpha"] == 1.0
    assert BAR_ERROR_STYLE == {
        "ecolor": "#374151",
        "elinewidth": 1.0,
        "capthick": 1.0,
    }


def test_optional_errorbars_preserve_missing_uncertainty() -> None:
    assert optional_errorbar_values([None, None]) is None
    partial = optional_errorbar_values([0.2, None])
    assert partial is not None
    assert partial[0] == 0.2
    assert math.isnan(partial[1])
    with pytest.raises(ValueError, match="must be nonnegative"):
        optional_errorbar_values([0.2, -0.1])


def test_algorithm_errorbars_enforce_stateless_dbas_contract() -> None:
    assert algorithm_errorbar_values("dbas", [None, None]) is None
    with pytest.raises(ValueError, match="stateless DBAS"):
        algorithm_errorbar_values("dbas", [0.0, 0.0])
    with pytest.raises(ValueError, match="missing a train-seed"):
        algorithm_errorbar_values("proposed", [0.2, None])


def test_scalability_report_draws_no_zero_width_caps_for_dbas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from matplotlib.axes import Axes

    rows: list[dict[str, str]] = []
    for algorithm in ("proposed", "dbas"):
        for value in (10.0, 20.0):
            row = {"algorithm": algorithm, "sweep_value": str(value)}
            for metric, mean in (
                ("social_welfare", 10.0),
                ("raw_rsu_reward", 8.0),
                ("task_success_rate", 0.9),
                ("perception_quality_total", 100.0),
            ):
                row[f"{metric}_seed_mean"] = str(mean)
                row[f"{metric}_seed_ci95"] = (
                    "0.2" if algorithm == "proposed" else "nan"
                )
            rows.append(row)

    calls: list[tuple[str, object]] = []
    original_errorbar = Axes.errorbar

    def capture_errorbar(self, *args, **kwargs):
        calls.append((str(kwargs.get("label")), kwargs.get("yerr")))
        return original_errorbar(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "errorbar", capture_errorbar)
    render_scale(
        rows,
        "vehicles",
        tmp_path / "scale",
        algorithms_to_plot=("proposed", "dbas"),
    )

    assert [yerr for label, yerr in calls if label == "DBAS"] == [None] * 4
    proposed = [yerr for label, yerr in calls if label == "TE-PPO (Proposed)"]
    assert proposed == [[0.2, 0.2]] * 4


def test_scalability_paper_and_full_algorithm_sets_are_explicit() -> None:
    rows = [
        {"algorithm": algorithm}
        for algorithm in ("proposed", "jdls", "mappo", "dbas", "ebas", "spis", "rdus")
    ]
    assert _figure_algorithms(rows, PAPER_ALGORITHMS) == list(PAPER_ALGORITHMS)
    assert _figure_algorithms(rows, None) == [
        "proposed", "ebas", "jdls", "mappo", "dbas", "rdus", "spis"
    ]


def test_dynamic_bandwidth_accepts_mhz_or_hz_without_double_scaling() -> None:
    mhz = [
        _dynamic_row("proposed", "rsu_bandwidth", "10_mhz", 10.0, 1.0),
        _dynamic_row("proposed", "rsu_bandwidth", "20_mhz", 20.0, 2.0),
    ]
    hz = [
        _dynamic_row("proposed", "rsu_bandwidth", "10_mhz", 10e6, 1.0),
        _dynamic_row("proposed", "rsu_bandwidth", "20_mhz", 20e6, 2.0),
    ]
    assert _dynamic_panel_data(mhz, "rsu_bandwidth", "proposed", 0, 1)[0] == [10.0, 20.0]
    assert _dynamic_panel_data(hz, "rsu_bandwidth", "proposed", 0, 1)[0] == [10.0, 20.0]


def test_dynamic_report_combines_panels(tmp_path: Path) -> None:
    rows: list[dict[str, str]] = []
    for algorithm, offset in (("proposed", 2.0), ("ebas", 1.0)):
        rows.extend(
            (
                _dynamic_row(algorithm, "occlusion", "default", 0.5, 10.0 + offset),
                _dynamic_row(algorithm, "occlusion", "severe", 0.7, 9.0 + offset),
                _dynamic_row(algorithm, "path_loss", "alpha_2.5", 2.5, 10.0 + offset),
                _dynamic_row(algorithm, "path_loss", "alpha_3.5", 3.5, 9.5 + offset),
                _dynamic_row(algorithm, "rsu_bandwidth", "10_mhz", 10.0, 9.0 + offset),
                _dynamic_row(algorithm, "rsu_bandwidth", "20_mhz", 20.0, 10.0 + offset),
            )
        )
    render_dynamic(rows, tmp_path)

    assert (tmp_path / "figR_dynamic_robustness.png").exists()
    assert (tmp_path / "figR_dynamic_robustness.pdf").exists()

def test_learning_report_combines_panels(
    tmp_path: Path,
) -> None:
    steps = np.asarray([10_000, 20_000, 40_000, 60_000])
    curves = {}
    algorithms = ("proposed", "jdls", "mappo", "ebas", "rdus", "spis")
    for index, algorithm in enumerate(algorithms):
        base = np.asarray([index, index + 1, index + 2, index + 3], dtype=float)
        curves[algorithm] = [
            (steps, base, None, tmp_path),
            (steps, base + 0.2, None, tmp_path),
        ]
    budget_rows = [
        {
            "algorithm": algorithm,
            "budget_steps": budget,
            "welfare_mean": float(index + budget / 20_000),
            "welfare_ci95": 0.1,
        }
        for index, algorithm in enumerate(curves)
        for budget in (20_000, 40_000, 60_000)
    ]
    render_learning_efficiency_figure(
        curves, budget_rows, tmp_path / "figR_learning_efficiency"
    )

    assert (tmp_path / "figR_learning_efficiency.png").exists()
    assert (tmp_path / "figR_learning_efficiency.pdf").exists()
    source_path = tmp_path / "figR_learning_efficiency_source.csv"
    assert source_path.exists()
    with source_path.open(encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    assert {row["series_kind"] for row in source_rows} == {
        "learning_curve",
        "equal_budget",
    }
    learning_rows = [
        row for row in source_rows if row["series_kind"] == "learning_curve"
    ]
    assert {row["algorithm"] for row in learning_rows} == {
        "proposed",
        "jdls",
        "mappo",
        "ebas",
    }
    assert len([row for row in learning_rows if row["algorithm"] == "ebas"]) == len(
        steps
    )
    equal_budget_rows = [
        row for row in source_rows if row["series_kind"] == "equal_budget"
    ]
    assert {row["algorithm"] for row in equal_budget_rows} == set(algorithms)
