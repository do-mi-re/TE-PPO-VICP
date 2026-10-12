from __future__ import annotations

import csv
import re

import pytest

from scripts.run_pruning_sensitivity import (
    FORMAL_SENSITIVITY_METRICS,
    RUNTIME_METRIC_NAMES,
    PruningCell,
    _paired_delta,
    _paper_table_rows,
    _rank_rows,
    _render_sensitivity_figure,
    _reported_proposed_trial,
    _sensitivity_figure_source_rows,
    build_cells,
    distinct_evaluation_cells,
    required_checkpoint_dirs,
    scenario_overrides_for,
)
from src.evaluation.revision import (
    load_revision_config,
    pruning_sensitivity_run_dir,
)
from src.plotting.revision_style import FIGSIZE_DOUBLE_COLUMN
from src.sim.scenario import candidate_limit


def test_formal_grid_has_one_mode_and_16_reported_cells():
    cells = build_cells()
    assert len(cells) == 16
    assert {cell.radius_m for cell in cells} == {60.0, 80.0, 100.0, 120.0}
    assert {cell.diversity_factor for cell in cells} == {1.0, 2.0, 3.0, 4.0}
    assert len(distinct_evaluation_cells()) == 12


def test_formal_sensitivity_decouples_sensing_quality():
    override = scenario_overrides_for("proposed", PruningCell(60.0, 3.0))
    assert override["candidate_pruning_radius_m"] == 60.0
    assert override["candidate_limit_range_m"] == 60.0
    assert override["sensing_range_m"] == 80.0


def test_baselines_keep_the_identical_physical_environment():
    left = scenario_overrides_for("jdls", PruningCell(60.0, 1.0))
    right = scenario_overrides_for("jdls", PruningCell(120.0, 4.0))
    assert (
        left
        == right
        == {
            "enable_candidate_pruning": False,
            "sensing_range_m": 80.0,
        }
    )


@pytest.mark.parametrize("radius", (60.0, 80.0, 100.0, 120.0))
@pytest.mark.parametrize("length", (15.0, 25.0, 35.0))
def test_rho_one_and_two_are_exact_cap_equivalents(radius, length):
    assert candidate_limit(length, radius, 1.0) == 2
    assert candidate_limit(length, radius, 2.0) == 2


def test_checkpoint_requirements_use_matched_cells_and_headline_seeds():
    paths = required_checkpoint_dirs(("proposed", "ebas", "dbas"))
    assert len(paths) == 96 + 8
    assert sum("/headline/" in str(path) for path in paths) == 16
    assert sum("/pruning_sensitivity/matched/" in str(path) for path in paths) == 88


def test_only_canonical_cell_resolves_to_headline():
    cfg = load_revision_config()
    canonical = pruning_sensitivity_run_dir(80.0, 4.0, 123, cfg)
    noncanonical = pruning_sensitivity_run_dir(80.0, 3.0, 123, cfg)
    assert "/headline/" in str(canonical)
    assert "/pruning_sensitivity/matched/" in str(noncanonical)


def test_rank_reference_comes_from_configured_canonical_cell():
    cfg = load_revision_config()
    aggregates = []
    for cell in build_cells(cfg):
        # Reverse the ordering only at rho=2. A stale hard-coded rho=2 default
        # would therefore produce the wrong reference ranks.
        proposed = 1.0 if cell.diversity_factor == 2.0 else 3.0
        ebas = 2.0
        for algorithm, value in (("proposed", proposed), ("ebas", ebas)):
            aggregates.append(
                {
                    "algorithm": algorithm,
                    "r0_m": cell.radius_m,
                    "diversity_factor": cell.diversity_factor,
                    "social_welfare_seed_mean": value,
                }
            )
    rows = _rank_rows(aggregates, "social_welfare", cfg)
    default = [row for row in rows if row["r0_m"] == 80.0 and row["diversity_factor"] == 4.0]
    assert {row["algorithm"]: row["default_rank"] for row in default} == {
        "proposed": 1,
        "ebas": 2,
    }


def test_paired_delta_uses_within_seed_ratios():
    seeds = (123, 231, 312, 413, 524, 635, 746, 857)
    stats = _paired_delta(
        {seed: float(index * 12) for index, seed in enumerate(seeds, 1)},
        {seed: float(index * 10) for index, seed in enumerate(seeds, 1)},
    )
    assert stats["mean"] == pytest.approx(20.0)
    assert stats["std"] == pytest.approx(0.0)
    assert stats["pairs"] == 8
    assert stats["paired_permutation_p"] == pytest.approx(2 / 2**8)


def test_formal_summary_excludes_unwarmed_runtime_metrics():
    assert set(RUNTIME_METRIC_NAMES).isdisjoint(FORMAL_SENSITIVITY_METRICS)


def test_equivalent_reported_cells_clone_one_physical_evaluation():
    cfg = load_revision_config()
    source = [{"eval_seed": 40000, "social_welfare": 12.5, "raw_rsu_reward": 9.0}]

    rho_one = _reported_proposed_trial(source, PruningCell(80.0, 1.0), 123, cfg)
    rho_two = _reported_proposed_trial(source, PruningCell(80.0, 2.0), 123, cfg)

    assert source == [{"eval_seed": 40000, "social_welfare": 12.5, "raw_rsu_reward": 9.0}]
    assert rho_one[0]["social_welfare"] == rho_two[0]["social_welfare"] == 12.5
    assert rho_one[0]["raw_rsu_reward"] == rho_two[0]["raw_rsu_reward"] == 9.0
    assert rho_one[0]["trained_diversity_factor"] == 2.0
    assert rho_two[0]["trained_diversity_factor"] == 2.0
    assert rho_one[0]["analytically_equivalent"] is True
    assert rho_two[0]["analytically_equivalent"] is False


def test_paper_sensitivity_figure_writes_png_and_pdf(tmp_path):
    cfg = load_revision_config()
    aggregates = []
    for cell in build_cells(cfg):
        for algorithm, offset in (("proposed", 1.0), ("ebas", 0.0)):
            aggregates.append(
                {
                    "algorithm": algorithm,
                    "r0_m": cell.radius_m,
                    "diversity_factor": cell.diversity_factor,
                    "social_welfare_seed_mean": 10.0 + offset,
                    "social_welfare_seed_ci95": 0.2,
                    "raw_rsu_reward_seed_mean": 8.0 + offset,
                    "raw_rsu_reward_seed_ci95": 0.3,
                    "effective_candidate_cap": max(2, int(cell.diversity_factor)),
                }
            )

    catalog = tmp_path / "figures"
    _render_sensitivity_figure(aggregates, tmp_path, cfg, catalog_dir=catalog)

    assert (tmp_path / "pruning_sensitivity_comparison.png").stat().st_size > 0
    assert (tmp_path / "pruning_sensitivity_comparison.pdf").stat().st_size > 0
    assert (catalog / "figR_pruning_sensitivity.png").stat().st_size > 0
    pdf_path = catalog / "figR_pruning_sensitivity.pdf"
    assert pdf_path.stat().st_size > 0
    media_box = re.search(
        rb"/MediaBox\s*\[\s*0\s+0\s+([0-9.]+)\s+([0-9.]+)\s*\]",
        pdf_path.read_bytes(),
    )
    assert media_box is not None
    width_points, height_points = (float(value) for value in media_box.groups())
    assert width_points / 72 == pytest.approx(FIGSIZE_DOUBLE_COLUMN[0])
    assert height_points / 72 == pytest.approx(FIGSIZE_DOUBLE_COLUMN[1])
    with (catalog / "figR_pruning_sensitivity_source.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 13
    assert sum(row["series_kind"] == "proposed" for row in rows) == 12
    assert sum(row["series_kind"] == "ebas_reference" for row in rows) == 1


def _figure_aggregates(*, unequal_equivalent_cell: bool = False) -> list[dict[str, object]]:
    cfg = load_revision_config()
    aggregates: list[dict[str, object]] = []
    for cell in build_cells(cfg):
        cap = max(2, int(cell.diversity_factor))
        for algorithm, offset in (("proposed", 1.0), ("ebas", 0.0)):
            mechanism_value = cap if algorithm == "proposed" else 0
            welfare = 10.0 + offset + mechanism_value
            if (
                unequal_equivalent_cell
                and algorithm == "proposed"
                and cell.radius_m == 80.0
                and cell.diversity_factor == 1.0
            ):
                welfare += 0.1
            aggregates.append(
                {
                    "algorithm": algorithm,
                    "r0_m": cell.radius_m,
                    "diversity_factor": cell.diversity_factor,
                    "effective_candidate_cap": cap,
                    "social_welfare_seed_mean": welfare,
                    "social_welfare_seed_ci95": 0.2,
                    "raw_rsu_reward_seed_mean": 8.0 + offset + mechanism_value,
                    "raw_rsu_reward_seed_ci95": 0.3,
                }
            )
    return aggregates


def test_sensitivity_figure_collapses_only_equal_rho_one_and_two_cells():
    rows = _sensitivity_figure_source_rows(_figure_aggregates(), load_revision_config())

    proposed = [row for row in rows if row["series_kind"] == "proposed"]
    assert len(proposed) == 12
    cap_two = [row for row in proposed if row["effective_candidate_cap"] == 2]
    assert len(cap_two) == 4
    assert {row["reported_diversity_factors"] for row in cap_two} == {"1;2"}


def test_sensitivity_figure_rejects_unequal_equivalent_cells():
    with pytest.raises(ValueError, match="equivalent pruning cells differ"):
        _sensitivity_figure_source_rows(
            _figure_aggregates(unequal_equivalent_cell=True),
            load_revision_config(),
        )


def test_sensitivity_figure_rejects_duplicate_proposed_cells():
    aggregates = _figure_aggregates()
    duplicate = next(row for row in aggregates if row["algorithm"] == "proposed")

    with pytest.raises(ValueError, match="duplicate proposed cells"):
        _sensitivity_figure_source_rows(
            [*aggregates, dict(duplicate)],
            load_revision_config(),
        )


def test_paper_table_has_one_numeric_row_per_reported_cell():
    cfg = load_revision_config()
    aggregates = []
    paired = []
    ranks = []
    for cell in build_cells(cfg):
        aggregates.append(
            {
                "algorithm": "proposed",
                "r0_m": cell.radius_m,
                "diversity_factor": cell.diversity_factor,
                "effective_candidate_cap": max(2, int(cell.diversity_factor)),
                "analytically_equivalent": cell.diversity_factor == 1.0,
                "social_welfare_seed_mean": 10.0,
                "social_welfare_seed_ci95": 0.5,
                "raw_rsu_reward_seed_mean": 8.0,
                "raw_rsu_reward_seed_ci95": 0.4,
                "perception_quality_total_seed_mean": 100.0,
                "perception_quality_total_seed_ci95": 2.0,
                "task_success_rate_seed_mean": 1.0,
                "task_success_rate_seed_ci95": 0.0,
            }
        )
        for metric in ("social_welfare", "raw_rsu_reward"):
            ranks.append(
                {
                    "algorithm": "proposed",
                    "r0_m": cell.radius_m,
                    "diversity_factor": cell.diversity_factor,
                    "metric": metric,
                    "rank": 1,
                }
            )
            for comparison in ("canonical", "ebas"):
                paired.append(
                    {
                        "r0_m": cell.radius_m,
                        "diversity_factor": cell.diversity_factor,
                        "metric": metric,
                        "comparison": comparison,
                        "change_pct": 1.0,
                    }
                )

    rows = _paper_table_rows(aggregates, paired, ranks, cfg)

    assert len(rows) == 16
    assert {row["effective_candidate_cap"] for row in rows} == {2, 3, 4}
    assert all(row["social_welfare_rank"] == 1 for row in rows)
