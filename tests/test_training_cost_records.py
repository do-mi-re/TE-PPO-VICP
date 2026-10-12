import csv
import json
import sys

import pytest

import run as runmod
from scripts import report_training_cost
from scripts.report_training_cost import load_current_runs


def test_training_record_includes_scale_and_budget_provenance(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config_snapshot.yaml").write_text(
        "scenario:\n  num_vehicles: 40\n  urgency_profiles: [[a, 1, 1], [b, 1, 1]]\n"
        "algorithm:\n  total_timesteps: 50000\n  device: cpu\n"
    )
    runmod._write_training_record({
        "result": {
            "run_dir": str(run_dir), "run_name": "x", "algorithm": "proposed",
            "train_seed": 1, "config_digest": "abc",
            "execution_mode": "isolated",
            "hardware": {
                "cpu_model": "Example CPU", "physical_cpu_count": 48,
                "logical_cpu_count": 96, "memory_total_gib": 128.0,
                "gpu_count": 4, "gpus": [{"name": "RTX 3090"}],
            },
        },
        "elapsed_s": 2.0,
    }, run_id="test")
    record = json.loads((run_dir / "training_record.json").read_text())
    assert record["num_vehicles"] == 40
    assert record["num_tasks"] == 2
    assert record["total_timesteps"] == 50000
    assert record["schema_version"] == 2
    assert record["timing_clock"] == "time.monotonic"
    assert record["hardware_id"]
    assert record["execution_mode"] == "isolated"
    assert record["hardware"]["physical_cpu_count"] == 48
    assert record["hardware"]["gpu_count"] == 4


def test_strict_current_runs_reject_mismatched_training_fingerprint(tmp_path):
    artifacts = tmp_path / "artifacts"
    run_dir = artifacts / "proposed_s1"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text("{}")
    (run_dir / "config_fingerprint.txt").write_text("expected")
    (run_dir / "training_record.json").write_text(json.dumps({
        "schema_version": 2,
        "run_id": "new",
        "run_name": "proposed_s1",
        "algorithm": "proposed",
        "train_seed": 1,
        "elapsed_s": 20.0,
        "config_digest": "stale",
    }))

    assert load_current_runs(
        artifacts
    ) == []


def test_artifact_path_not_run_name_is_current_record_identity(tmp_path):
    artifacts = tmp_path / "artifacts"
    for purpose in ("headline", "scalability"):
        run_dir = artifacts / purpose / "proposed_s1"
        run_dir.mkdir(parents=True)
        (run_dir / "summary.json").write_text("{}")
        (run_dir / "config_fingerprint.txt").write_text("current")
        (run_dir / "training_record.json").write_text(json.dumps({
            "schema_version": 2,
            "run_id": purpose,
            "run_name": "proposed_s1",
            "algorithm": "proposed",
            "train_seed": 1,
            "elapsed_s": 10.0,
            "config_digest": "current",
            "timing_clock": "time.monotonic",
            "hardware_id": "example",
            "hardware": {"cpu_model": "Example CPU"},
        }))

    runs = load_current_runs(artifacts)

    assert len(runs) == 2


@pytest.mark.parametrize("schema_version, elapsed_s", (
    (1, 10.0), (3, 10.0), (2.5, 10.0),
    (2, float("nan")), (2, float("inf")), (2, 0.0), (2, -1.0),
))
def test_invalid_timing_records_are_excluded(tmp_path, schema_version, elapsed_s):
    run_dir = tmp_path / "proposed_s1"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text("{}")
    (run_dir / "config_fingerprint.txt").write_text("current")
    (run_dir / "training_record.json").write_text(json.dumps({
        "schema_version": schema_version,
        "run_name": "proposed_s1",
        "algorithm": "proposed",
        "elapsed_s": elapsed_s,
        "config_digest": "current",
    }))

    assert load_current_runs(tmp_path) == []


@pytest.mark.parametrize("payload", ([], None))
def test_malformed_timing_record_does_not_abort_report(tmp_path, payload):
    run_dir = tmp_path / "proposed_s1"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text("{}")
    (run_dir / "training_record.json").write_text(json.dumps(payload))

    assert load_current_runs(tmp_path) == []


@pytest.mark.parametrize("duration_scale", (1.0, 1.0 / 3600))
def test_cost_report_separates_shapes_and_totals_all_algorithm_groups(
    tmp_path, monkeypatch, capsys, duration_scale,
):
    artifacts = tmp_path / "artifacts"
    output = tmp_path / "tables"
    for name, algorithm, vehicles, hardware, elapsed in (
        ("p10_s1", "proposed", 10, "hardware-a", 3600.0),
        ("p10_s2", "proposed", 10, "hardware-a", 7200.0),
        ("p60_s1", "proposed", 60, "hardware-a", 3600.0),
        ("p10b_s1", "proposed", 10, "hardware-b", 3600.0),
        ("m10_s1", "mappo", 10, "hardware-a", 7200.0),
    ):
        run_dir = artifacts / name
        run_dir.mkdir(parents=True)
        (run_dir / "summary.json").write_text("{}")
        (run_dir / "config_fingerprint.txt").write_text("fixture")
        (run_dir / "training_record.json").write_text(json.dumps({
            "schema_version": 2, "config_digest": "fixture", "run_name": name,
            "algorithm": algorithm, "num_vehicles": vehicles, "num_tasks": 3,
            "purpose": "sweep_metric", "total_timesteps": 600000,
            "hardware_id": hardware, "execution_mode": "scheduled",
            "elapsed_s": elapsed * duration_scale,
        }))
    monkeypatch.setattr(sys, "argv", [
        "report_training_cost", "--artifacts-root", str(artifacts),
        "--output-dir", str(output),
    ])

    assert report_training_cost.main() == 0

    console = capsys.readouterr().out
    assert f"proposed accounts for {5.0 * duration_scale:.1f} h (71% of the total)" in console
    assert "hardware=hardware-a" in console and "hardware=hardware-b" in console
    with (output / "training_cost.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    small = next(row for row in rows if row["algorithm"] == "proposed"
                 and row["num_vehicles"] == "10" and row["hardware_id"] == "hardware-a")
    assert int(small["runs"]) == 2
    assert float(small["median_s"]) == pytest.approx(5400.0 * duration_scale)
    large = next(row for row in rows if row["num_vehicles"] == "60")
    assert int(large["runs"]) == 1
    assert float(large["median_s"]) == pytest.approx(3600.0 * duration_scale)
