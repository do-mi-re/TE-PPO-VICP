"""CPU DBAS calibration selection and artifact-identity checks."""

from pathlib import Path

import pytest
import yaml

from scripts import run_cpu_robustness_experiment as experiment
from src.evaluation import revision, sweep
from src.evaluation.revision import (
    load_cpu_robustness_dbas_calibration,
    load_revision_config,
)


def _config(tmp_path):
    cfg = load_revision_config()
    cfg["paths"] = {
        "artifacts_root": str(tmp_path / "artifacts"),
        "outputs_root": str(tmp_path / "outputs"),
    }
    cfg["baseline_calibration"]["dbas_price_grid"] = (0.01, 0.02, 0.03)
    cfg["cpu_robustness"]["dbas_selection_file"] = str(tmp_path / "selected.yaml")
    return cfg


@pytest.mark.parametrize("changed_input", ("prices", "seeds", "base_config", "implementation"))
def test_calibration_uses_maximin_and_rejects_a_failed_band(tmp_path, monkeypatch, changed_input):
    cfg = _config(tmp_path)
    bands = cfg["cpu_robustness"]["cpu_capacity_ranges_cycles_per_s"]
    validation_seeds = tuple(cfg["seeds"]["validation"])

    def evaluate(algorithm, run_dir, seeds, *, dbas_config, scenario_override):
        assert algorithm == "dbas"
        assert seeds == validation_seeds
        price = round(dbas_config["algorithm"]["fixed_price_fraction"] * 0.3, 2)
        band = tuple(scenario_override["cpu_cycles_per_s_range"])
        first_band = band == tuple(bands[0])
        welfare = {0.01: 100.0, 0.02: 4.0 if first_band else 20.0, 0.03: 5.0}[price]
        return [{
            "social_welfare": welfare,
            "raw_rsu_reward": welfare / 2,
            "task_success_rate": 0.5 if price == 0.01 and first_band else 0.75,
            "selected_vehicle_count": 1.0,
        }]

    monkeypatch.setattr(experiment, "evaluate_run_dir", evaluate)
    experiment.calibrate_dbas(cfg)
    selected = yaml.safe_load(Path(cfg["cpu_robustness"]["dbas_selection_file"]).read_text())
    assert selected["price"] == pytest.approx(0.03)
    assert selected["minimum_social_welfare"] == pytest.approx(5.0)
    assert load_cpu_robustness_dbas_calibration(cfg, required=True) == {
        "algorithm.fixed_price_fraction": pytest.approx(0.1)
    }

    if changed_input == "prices":
        cfg["baseline_calibration"]["dbas_price_grid"] = (0.01, 0.02)
    elif changed_input == "seeds":
        cfg["seeds"]["validation"] = tuple(seed + 1 for seed in validation_seeds)
    elif changed_input == "base_config":
        base = sweep.load_algorithm_config("dbas")
        base["scenario"]["num_vehicles"] += 1
        monkeypatch.setattr(sweep, "load_algorithm_config", lambda _: base)
    else:
        monkeypatch.setattr(revision, "sha256_file", lambda _: "changed-source")
    with pytest.raises(ValueError, match="contract is stale"):
        load_cpu_robustness_dbas_calibration(cfg, required=True)


@pytest.mark.parametrize("failure", ("success", "participation"))
def test_calibration_fails_when_no_price_meets_service_constraints(
    tmp_path, monkeypatch, failure,
):
    cfg = _config(tmp_path)
    monkeypatch.setattr(experiment, "evaluate_run_dir", lambda *args, **kwargs: [{
        "social_welfare": 100.0,
        "raw_rsu_reward": 50.0,
        "task_success_rate": 0.74 if failure == "success" else 1.0,
        "selected_vehicle_count": 0.0 if failure == "participation" else 1.0,
    }])
    with pytest.raises(ValueError, match="per-band service gate"):
        experiment.calibrate_dbas(cfg)
    assert not Path(cfg["cpu_robustness"]["dbas_selection_file"]).exists()
