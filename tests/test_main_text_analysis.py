import math

import numpy as np

from scripts.report_main_text_figures import (
    render_mechanism_analysis,
    render_resource_sensitivity,
)
from scripts.run_main_text_analysis import (
    _compression_rows_from_info,
    _scaled_prices,
    summarize_across_training_seeds,
    summarize_by_seed,
)
from scripts.validate_revision_outputs import _validate_seed_reduction
from src.sim.scenario import ScenarioConfig


def test_price_scaling_clips_and_reports_component_fraction():
    prices, fraction = _scaled_prices(np.array([0.1, 0.25, 0.3]), 1.5, 0.3)
    assert prices == (0.15000000000000002, 0.3, 0.3)
    assert fraction == 2 / 3


def test_seed_summary_weights_evaluation_scenarios_equally():
    rows = [
        {"algorithm": "proposed", "sweep_value": 1.0, "train_seed": 1, "eval_seed": 10, "m": 0.0},
        {"algorithm": "proposed", "sweep_value": 1.0, "train_seed": 1, "eval_seed": 11, "m": 10.0},
        {"algorithm": "proposed", "sweep_value": 1.0, "train_seed": 1, "eval_seed": 11, "m": 20.0},
    ]
    summary = summarize_by_seed(
        rows, group_fields=("algorithm", "sweep_value"), metrics=("m",)
    )
    assert summary[0]["m"] == 7.5
    assert summary[0]["evaluation_scenario_count"] == 2
    assert summary[0]["source_row_count"] == 3


def test_final_summary_weights_training_seeds_equally_after_scenario_reduction():
    rows = [
        {"algorithm": "proposed", "sweep_value": 1.0, "train_seed": 1, "eval_seed": 10, "m": 0.0},
        {"algorithm": "proposed", "sweep_value": 1.0, "train_seed": 1, "eval_seed": 11, "m": 10.0},
        {"algorithm": "proposed", "sweep_value": 1.0, "train_seed": 1, "eval_seed": 11, "m": 20.0},
        {"algorithm": "proposed", "sweep_value": 1.0, "train_seed": 2, "eval_seed": 10, "m": 100.0},
    ]
    seed_rows = summarize_by_seed(
        rows, group_fields=("algorithm", "sweep_value"), metrics=("m",)
    )
    summary = summarize_across_training_seeds(
        seed_rows, group_fields=("algorithm", "sweep_value"), metrics=("m",)
    )
    assert summary[0]["m_seed_mean"] == (7.5 + 100.0) / 2
    assert summary[0]["m_seed_replicates"] == 2


def test_validator_rejects_incorrect_seed_reduction():
    seed_rows = [
        {"algorithm": "proposed", "sweep_value": "1", "train_seed": "1", "m": "1"},
        {"algorithm": "proposed", "sweep_value": "1", "train_seed": "2", "m": "3"},
    ]
    expected_std = math.sqrt(2.0)
    summary = [{
        "algorithm": "proposed",
        "sweep_value": "1",
        "m_seed_mean": "2",
        "m_seed_std": str(expected_std),
        "m_seed_ci95": str(1.96 * expected_std / math.sqrt(2.0)),
        "m_seed_replicates": "2",
    }]
    issues: list[str] = []
    _validate_seed_reduction(
        summary, seed_rows, metrics=("m",), label="test", issues=issues
    )
    assert issues == []

    summary[0]["m_seed_mean"] = "2.5"
    _validate_seed_reduction(
        summary, seed_rows, metrics=("m",), label="test", issues=issues
    )
    assert "incorrect m seed mean" in issues[-1]


def test_compression_counterfactual_uses_exact_energy_decomposition():
    config = ScenarioConfig(num_vehicles=2, pricing_budget_max=0.3)
    info = {
        "observation_snapshot": {
            "channels": np.array([1e-8, 1e-8]),
            "vehicles": {
                "cpu_cycles_per_s": np.array([15e9, 15e9]),
                "sensor_power_w": np.array([7.5, 7.5]),
            },
            "tasks": {"tau_max_s": np.array([0.1, 0.3, 0.5])},
        },
        "selected_tasks": np.array([0, -1]),
        "follower_sensing_durations_s": np.array([0.02, 0.0]),
        "follower_compression_ratios": np.array([10.0, 0.0]),
        "leader_action": {"bandwidth": np.array([0.5, 0.5])},
    }
    rows = _compression_rows_from_info(
        info,
        train_seed=123,
        eval_seed=80000,
        eta_grid=(1.0, 10.0, 50.0),
        scenario_config=config,
        provenance={
            "config_fingerprint": "f" * 64,
            "checkpoint_sha256": "c" * 64,
            "normalization_sha256": None,
            "selection_sha256": "s" * 64,
            "artifact_kind": "trained_checkpoint",
        },
    )
    assert len(rows) == 3
    assert [row["compression_ratio"] for row in rows] == [1.0, 10.0, 50.0]
    assert [row["computation_energy_j"] for row in rows] == sorted(
        row["computation_energy_j"] for row in rows
    )
    assert [row["transmission_energy_j"] for row in rows] == sorted(
        (row["transmission_energy_j"] for row in rows), reverse=True
    )
    for row in rows:
        assert math.isclose(
            row["total_energy_j"],
            row["sensing_energy_j"]
            + row["computation_energy_j"]
            + row["transmission_energy_j"],
            rel_tol=1e-12,
        )


def test_main_text_renderers_write_png_and_pdf(tmp_path):
    algorithms = ("proposed", "ebas", "jdls", "mappo", "dbas")
    cpu = [
        {
            "algorithm": algorithm,
            "sweep_value": str(x),
            "social_welfare_seed_mean": str(10 + x / 10),
            "social_welfare_seed_ci95": "nan" if algorithm == "dbas" else "0.2",
            "social_welfare_seed_replicates": "1" if algorithm == "dbas" else "8",
            "raw_rsu_reward_seed_mean": str(8 + x / 10),
            "raw_rsu_reward_seed_ci95": "nan" if algorithm == "dbas" else "0.15",
            "raw_rsu_reward_seed_replicates": "1" if algorithm == "dbas" else "8",
            "task_success_rate_seed_mean": "0.9",
            "task_success_rate_seed_ci95": "nan" if algorithm == "dbas" else "0.02",
            "task_success_rate_seed_replicates": "1" if algorithm == "dbas" else "8",
        }
        for algorithm in algorithms
        for x in (5, 10, 15, 20, 25)
    ]
    resource = tmp_path / "resource"
    bandwidth = [
        {
            "algorithm": algorithm,
            "sweep_value": str(x * 1e6),
            "raw_rsu_reward_seed_mean": str(7 + x / 10),
            "raw_rsu_reward_seed_ci95": "nan" if algorithm == "dbas" else "0.1",
            "raw_rsu_reward_seed_replicates": "1" if algorithm == "dbas" else "5",
        }
        for algorithm in algorithms
        for x in (5, 10, 20, 30, 40)
    ]
    source = render_resource_sensitivity(
        cpu, bandwidth, resource, algorithms_to_plot=algorithms
    )
    assert resource.with_suffix(".png").exists()
    assert resource.with_suffix(".pdf").exists()
    assert len(source) == 5 * (5 + 5 + 5)
    panel_c = [row for row in source if row["panel"] == "c"]
    assert {row["panel_title"] for row in panel_c} == {"System Bandwidth"}
    assert {row["metric"] for row in panel_c} == {"raw_rsu_reward"}
    assert {row["sweep_value"] for row in panel_c} == {5.0, 10.0, 20.0, 30.0, 40.0}
    dbas_rows = [row for row in source if row["algorithm"] == "dbas"]
    assert dbas_rows
    assert {row["seed_ci95"] for row in dbas_rows} == {""}
    assert {row["uncertainty"] for row in dbas_rows} == {"none_stateless"}

    compression = []
    for eta in (1, 2, 5, 10):
        row = {"sweep_value": str(eta)}
        for metric, value in (
            ("sensing_energy_j", 0.01),
            ("computation_energy_j", 0.001 * eta),
            ("transmission_energy_j", 0.1 / eta),
            ("total_energy_j", 0.01 + 0.001 * eta + 0.1 / eta),
        ):
            row[f"{metric}_seed_mean"] = str(value)
            row[f"{metric}_seed_ci95"] = "0.001"
            row[f"{metric}_seed_replicates"] = "8"
        compression.append(row)
    pricing = [
        {
            "sweep_value": str(theta),
            "raw_rsu_reward_seed_mean": str(10 - abs(theta - 1)),
            "raw_rsu_reward_seed_ci95": "0.1",
            "raw_rsu_reward_seed_replicates": "8",
            "price_cap_fraction_seed_mean": "0.1" if theta > 1.0 else "0.0",
        }
        for theta in (0.5, 0.9, 1.0, 1.1, 1.5)
    ]
    mechanism = tmp_path / "mechanism"
    mechanism_source = render_mechanism_analysis(compression, pricing, mechanism)
    assert mechanism.with_suffix(".png").exists()
    assert mechanism.with_suffix(".pdf").exists()
    assert len(mechanism_source) == 4 * len(compression) + len(pricing)
