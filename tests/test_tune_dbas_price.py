from __future__ import annotations

import subprocess
import sys

import pytest
import yaml

from scripts.tune_dbas_price import _calibration_scenarios, choose_best_candidate
from src.evaluation import revision, sweep


def test_choose_best_candidate_maximizes_social_welfare() -> None:
    rows = [
        {
            "price": 0.01,
            "fixed_price_fraction": 0.0333333333,
            "social_welfare_mean": 5.0,
            "raw_rsu_reward_mean": 4.0,
            "task_success_rate_mean": 1.0,
            "selected_vehicle_count_mean": 3.0,
        },
        {
            "price": 0.03,
            "fixed_price_fraction": 0.1,
            "social_welfare_mean": 7.0,
            "raw_rsu_reward_mean": 2.0,
            "task_success_rate_mean": 0.9,
            "selected_vehicle_count_mean": 2.0,
        },
    ]

    assert choose_best_candidate(rows)["price"] == 0.03


def test_choose_best_candidate_rejects_low_success_and_zero_service() -> None:
    rows = [
        {
            "price": 0.01,
            "fixed_price_fraction": 0.0333333333,
            "social_welfare_mean": 20.0,
            "raw_rsu_reward_mean": 20.0,
            "task_success_rate_mean": 0.5,
            "selected_vehicle_count_mean": 3.0,
        },
        {
            "price": 0.02,
            "fixed_price_fraction": 0.0666666667,
            "social_welfare_mean": 18.0,
            "raw_rsu_reward_mean": 18.0,
            "task_success_rate_mean": 0.9,
            "selected_vehicle_count_mean": 0.0,
        },
        {
            "price": 0.04,
            "fixed_price_fraction": 0.1333333333,
            "social_welfare_mean": 10.0,
            "raw_rsu_reward_mean": 8.0,
            "task_success_rate_mean": 0.75,
            "selected_vehicle_count_mean": 1.0,
        },
    ]

    assert choose_best_candidate(rows)["price"] == 0.04


def test_choose_best_candidate_tiebreaks_by_rsu_utility_then_lower_price() -> None:
    rows = [
        {
            "price": 0.05,
            "fixed_price_fraction": 0.1666666667,
            "social_welfare_mean": 10.0,
            "raw_rsu_reward_mean": 8.0,
            "task_success_rate_mean": 1.0,
            "selected_vehicle_count_mean": 1.0,
        },
        {
            "price": 0.04,
            "fixed_price_fraction": 0.1333333333,
            "social_welfare_mean": 10.0,
            "raw_rsu_reward_mean": 9.0,
            "task_success_rate_mean": 1.0,
            "selected_vehicle_count_mean": 1.0,
        },
        {
            "price": 0.03,
            "fixed_price_fraction": 0.1,
            "social_welfare_mean": 10.0,
            "raw_rsu_reward_mean": 9.0,
            "task_success_rate_mean": 1.0,
            "selected_vehicle_count_mean": 1.0,
        },
    ]

    assert choose_best_candidate(rows)["price"] == 0.03


def test_tune_dbas_price_script_runs_as_file() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/tune_dbas_price.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Tune DBAS fixed price locally" in result.stdout


def test_calibration_scenarios_deduplicate_default_equivalents() -> None:
    base_config = {
        "scenario": {
            "num_vehicles": 20,
            "pricing_budget_max": 0.3,
            "spatial_potential_max": 50.0,
        }
    }

    scenarios = _calibration_scenarios(base_config)
    names = [name for name, _ in scenarios]

    assert "default" in names
    assert "vehicles_20" not in names
    assert "bandwidth_20000000" not in names


def test_dbas_base_config_uses_configured_fixed_price() -> None:
    with open("configs/algorithms/dbas.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    assert config["algorithm"]["fixed_price_fraction"] == pytest.approx(0.04 / 0.3)
    assert (
        config["algorithm"]["fixed_price_fraction"]
        * config["scenario"]["pricing_budget_max"]
    ) == pytest.approx(0.04)


@pytest.mark.parametrize("changed_input", ("seeds", "prices", "base_config", "implementation", "missing_digest"))
def test_formal_calibration_rejects_changed_inputs(tmp_path, monkeypatch, changed_input):
    config = revision.load_revision_config()
    path = tmp_path / "selected.yaml"
    config["baseline_calibration"]["dbas_selection_file"] = str(path)
    record = {
        "schema_version": 1,
        "algorithm.fixed_price_fraction": 0.04 / 0.3,
        "validation_seed_count": len(config["seeds"]["validation"]),
        "calibration_contract_digest": revision.dbas_calibration_contract_digest(config),
    }
    path.write_text(yaml.safe_dump(record), encoding="utf-8")
    assert revision.load_dbas_calibration(config, required=True) == {
        "algorithm.fixed_price_fraction": pytest.approx(0.04 / 0.3)
    }

    if changed_input == "seeds":
        config["seeds"]["validation"] = tuple(seed + 1 for seed in config["seeds"]["validation"])
    elif changed_input == "prices":
        config["baseline_calibration"]["dbas_price_grid"] = (0.01, 0.02)
    elif changed_input == "base_config":
        base = sweep.load_algorithm_config("dbas")
        base["scenario"]["num_vehicles"] += 1
        monkeypatch.setattr(sweep, "load_algorithm_config", lambda _: base)
    elif changed_input == "implementation":
        monkeypatch.setattr(revision, "sha256_file", lambda _: "changed-source")
    else:
        record.pop("calibration_contract_digest")
        path.write_text(yaml.safe_dump(record), encoding="utf-8")

    with pytest.raises(ValueError, match="contract is missing or stale"):
        revision.load_dbas_calibration(config, required=True)


def test_local_calibration_overrides_do_not_claim_the_formal_contract():
    config = revision.load_revision_config()
    formal = revision.dbas_calibration_contract_digest(config)
    assert revision.dbas_calibration_contract_digest(
        config, price_grid=(0.01,),
    ) != formal
    assert revision.dbas_calibration_contract_digest(
        config, validation_seeds=tuple(config["seeds"]["validation"][:1]),
    ) != formal
