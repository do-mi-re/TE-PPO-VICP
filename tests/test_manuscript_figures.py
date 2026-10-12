"""Regression tests for manuscript-compatible formal experiment figures."""

import csv
from pathlib import Path

from scripts.report_manuscript_figures import (
    FULL_STEMS,
    MANUSCRIPT_STEMS,
    _algorithm_label,
    _legend,
    render_manuscript_figures,
)
from scripts.validate_revision_outputs import (
    _png_dimensions,
    _validate_manuscript_figures,
)
from src.evaluation.provenance import sha256_file
from src.evaluation.sweep import write_sweep_csv
from src.plotting.revision_style import FIGURE_DPI

ALGORITHMS = ("proposed", "ebas", "jdls", "mappo", "dbas", "rdus", "spis")


def _metric_row(algorithm: str, x: float) -> dict[str, object]:
    row: dict[str, object] = {
        "algorithm": algorithm,
        "sweep_value": x,
    }
    for metric, base in (
        ("social_welfare", 10.0),
        ("raw_rsu_reward", 8.0),
        ("perception_quality_total", 100.0),
    ):
        row[f"{metric}_seed_mean"] = base + x / 100.0
        row[f"{metric}_seed_ci95"] = "nan" if algorithm == "dbas" else 0.2
        row[f"{metric}_seed_replicates"] = 1 if algorithm == "dbas" else 5
    return row


def _write_fixture_bundle(root: Path) -> dict[str, object]:
    paper = ("proposed", "ebas", "jdls", "mappo", "dbas")
    cfg: dict[str, object] = {
        "manuscript_figures": {
            "contract_version": 3,
            "output_subdir": "figures/manuscript",
            "convergence_algorithms": ["proposed", "jdls", "mappo", "ebas"],
            "paper_algorithms": list(paper),
            "full_algorithms": list(ALGORITHMS),
        },
        "scalability": {
            "vehicle_grid": [10, 20],
            "rsu_bandwidth_grid_hz": [5e6, 10e6],
        },
        "main_text_analysis": {
            "cpu_capacity_ranges_cycles_per_s": [[2.5e9, 7.5e9], [7.5e9, 12.5e9]],
            "compression_ratio_grid": [1, 5, 10],
            "price_scale_grid": [0.5, 1.0, 1.5],
        },
    }
    convergence = [
        {
            "algorithm": algorithm,
            "series_kind": "learning_curve",
            "timestep": step,
            "social_welfare_mean": 5 + step / 10_000,
            "social_welfare_seed_std": 0.2,
            "train_seed_count": 8,
        }
        for algorithm in ("proposed", "jdls", "mappo", "ebas")
        for step in (3072, 6144, 9216)
    ]
    vehicles = [_metric_row(algorithm, x) for algorithm in ALGORITHMS for x in (10, 20)]
    bandwidth = [
        {
            **_metric_row(algorithm, x),
            "stress_experiment": "rsu_bandwidth",
            "stress_level": f"{x / 1e6:g}_mhz",
        }
        for algorithm in ALGORITHMS
        for x in (5e6, 10e6)
    ]
    cpu = [_metric_row(algorithm, x) for algorithm in ALGORITHMS for x in (5, 10)]
    compression = []
    for eta in (1, 5, 10):
        row: dict[str, object] = {"algorithm": "proposed", "sweep_value": eta}
        for metric, value in (
            ("sensing_energy_j", 0.02),
            ("computation_energy_j", 0.001 * eta),
            ("transmission_energy_j", 0.1 / eta),
            ("total_energy_j", 0.02 + 0.001 * eta + 0.1 / eta),
        ):
            row[f"{metric}_seed_mean"] = value
            row[f"{metric}_seed_ci95"] = 0.001
            row[f"{metric}_seed_replicates"] = 8
        compression.append(row)
    pricing = [
        {
            "algorithm": "proposed",
            "sweep_value": theta,
            "raw_rsu_reward_seed_mean": 10 - abs(theta - 1),
            "raw_rsu_reward_seed_ci95": 0.1,
            "raw_rsu_reward_seed_replicates": 8,
            "price_cap_fraction_seed_mean": 0.2 if theta > 1 else 0.0,
        }
        for theta in (0.5, 1.0, 1.5)
    ]
    write_sweep_csv(root / "figures" / "figR_learning_efficiency_source.csv", convergence)
    write_sweep_csv(root / "scalability" / "vehicles" / "summary.csv", vehicles)
    write_sweep_csv(root / "dynamic_robustness" / "summary.csv", bandwidth)
    write_sweep_csv(root / "cpu_robustness" / "formal" / "summary.csv", cpu)
    write_sweep_csv(root / "main_text_analysis" / "compression" / "summary.csv", compression)
    write_sweep_csv(root / "main_text_analysis" / "pricing" / "summary.csv", pricing)
    return cfg


def test_manuscript_bundle_renders_and_validates(tmp_path: Path) -> None:
    cfg = _write_fixture_bundle(tmp_path)
    manifest = render_manuscript_figures(tmp_path, cfg)
    expected = set(MANUSCRIPT_STEMS + FULL_STEMS)
    assert {row["figure"] for row in manifest} == expected
    root = tmp_path / "figures" / "manuscript"
    for stem in expected:
        assert (root / f"{stem}.png").is_file()
        assert (root / f"{stem}.pdf").is_file()
        assert (root / "source_data" / f"{stem}.csv").is_file()
        expected_inches = (3.5, 2.55)
        expected_pixels = tuple(round(value * FIGURE_DPI) for value in expected_inches)
        assert _png_dimensions(root / f"{stem}.png") == expected_pixels
    by_figure = {row["figure"]: row for row in manifest}
    for stem in expected:
        assert by_figure[stem]["layout"] == "single_column"
        assert by_figure[stem]["legend_layout"] == "single_row_adaptive"
        assert by_figure[stem]["figure_width_in"] == 3.5
        assert by_figure[stem]["figure_height_in"] == 2.55
    issues: list[str] = []
    _validate_manuscript_figures(tmp_path, cfg, issues)
    assert issues == []


def test_compact_legend_uses_one_row_and_readable_font() -> None:
    class FakeAxes:
        def __init__(self) -> None:
            self.legend_args: tuple[object, ...] = ()
            self.legend_kwargs: dict[str, object] = {}

        def legend(self, *args: object, **kwargs: object) -> None:
            self.legend_args = args
            self.legend_kwargs = kwargs

    ax = FakeAxes()
    _legend(ax, 7)

    assert ax.legend_args == ()
    assert ax.legend_kwargs["loc"] == "lower left"
    assert ax.legend_kwargs["bbox_to_anchor"] == (0.0, 1.02, 1.0, 0.0)
    assert ax.legend_kwargs["mode"] == "expand"
    assert ax.legend_kwargs["borderaxespad"] == 0.0
    assert ax.legend_kwargs["ncol"] == 7
    assert ax.legend_kwargs["fontsize"] == 6.8
    assert ax.legend_kwargs["columnspacing"] == 0.22
    assert ax.legend_kwargs["handlelength"] == 0.85

    single = FakeAxes()
    _legend(single, 1)

    assert single.legend_args == ()
    assert single.legend_kwargs["loc"] == "lower center"
    assert single.legend_kwargs["bbox_to_anchor"] == (0.5, 1.02)
    assert single.legend_kwargs["ncol"] == 1
    assert "mode" not in single.legend_kwargs
    assert "borderaxespad" not in single.legend_kwargs


def test_manuscript_algorithm_labels_are_compact() -> None:
    assert _algorithm_label("proposed") == "TE-PPO"
    assert _algorithm_label("ebas") == "EBAS"


def test_manuscript_validator_rejects_stale_input_digest(tmp_path: Path) -> None:
    cfg = _write_fixture_bundle(tmp_path)
    render_manuscript_figures(tmp_path, cfg)
    source = tmp_path / "scalability" / "vehicles" / "summary.csv"
    source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    issues: list[str] = []
    _validate_manuscript_figures(tmp_path, cfg, issues)
    assert any("input digest mismatch" in issue for issue in issues)


def test_manuscript_validator_rejects_modified_figure(tmp_path: Path) -> None:
    cfg = _write_fixture_bundle(tmp_path)
    render_manuscript_figures(tmp_path, cfg)
    figure = tmp_path / "figures" / "manuscript" / "fig2.png"
    figure.write_bytes(figure.read_bytes() + b"modified")
    issues: list[str] = []
    _validate_manuscript_figures(tmp_path, cfg, issues)
    assert any("png figure digest mismatch" in issue for issue in issues)


def test_manuscript_validator_rejects_fabricated_dbas_interval(
    tmp_path: Path,
) -> None:
    cfg = _write_fixture_bundle(tmp_path)
    manifest = render_manuscript_figures(tmp_path, cfg)
    source = tmp_path / "figures" / "manuscript" / "source_data" / "fig2.csv"
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row["algorithm"] == "dbas":
            row["seed_ci95"] = "0.0"
            row["uncertainty"] = "seed_ci95"
    write_sweep_csv(source, rows)
    for record in manifest:
        if record["figure"] == "fig2":
            record["source_csv_sha256"] = sha256_file(source)
    write_sweep_csv(
        tmp_path / "figures" / "manuscript" / "manifest.csv", manifest
    )

    issues: list[str] = []
    _validate_manuscript_figures(tmp_path, cfg, issues)

    assert any("DBAS must not carry" in issue for issue in issues)
