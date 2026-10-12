"""Configuration fingerprint, selected-checkpoint, and worker-record checks.

Mock training isolates artifact lifecycle from numerical training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

import run as runmod
from src.evaluation.provenance import (
    digest_payload,
    sha256_file,
)
from src.evaluation.tasks import RunSpec


def _make_trained_run_dir(root: Path, run_name: str = "proposed_s1") -> Path:
    run_dir = root / run_name
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text("{}")
    return run_dir


def _complete_selected_bundle(run_dir: Path, config: dict) -> None:
    """Populate the minimum fail-closed selected-checkpoint artifact."""

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text("{}")
    (run_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(config))
    digest = runmod._train_config_digest(config)
    assert digest is not None
    (run_dir / runmod._TRAIN_FINGERPRINT_FILE).write_text(digest)
    model = run_dir / "selected_model.zip"
    model.write_bytes(b"checkpoint")
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(exist_ok=True)
    np.savez(
        eval_dir / "evaluations.npz",
        timesteps=np.array([10]),
        social_welfare=np.array([[2.0]]),
    )
    (run_dir / "selection.json").write_text(json.dumps({
        "selection_metric": "social_welfare",
        "selected_timestep": 10,
        "selected_metric_value": 2.0,
        "validation_seeds": [],
        "config_fingerprint": digest,
        "model_file": model.name,
        "model_sha256": sha256_file(model),
        "normalization_file": None,
        "normalization_sha256": None,
    }))


# --------------------------------------------------------------------------- #
# Pure freshness classifier
# --------------------------------------------------------------------------- #

def test_absent_when_no_summary(tmp_path):
    run_dir = tmp_path / "proposed_s1"
    run_dir.mkdir()
    assert runmod._train_freshness(run_dir, "abc") == "absent"


def test_fresh_when_fingerprint_matches(tmp_path):
    run_dir = _make_trained_run_dir(tmp_path)
    (run_dir / runmod._TRAIN_FINGERPRINT_FILE).write_text("abc")
    assert runmod._train_freshness(run_dir, "abc") == "fresh"


def test_stale_when_fingerprint_differs(tmp_path):
    run_dir = _make_trained_run_dir(tmp_path)
    (run_dir / runmod._TRAIN_FINGERPRINT_FILE).write_text("OLD-config-digest")
    assert runmod._train_freshness(run_dir, "new-config-digest") == "stale"


def test_stale_when_fingerprint_absent(tmp_path):
    """A summary without its required fingerprint is stale."""
    run_dir = _make_trained_run_dir(tmp_path)  # no sidecar written
    assert runmod._train_freshness(run_dir, "any-digest") == "stale"


def test_stale_when_expected_unhashable(tmp_path):
    """An unhashable (None-digest) config never matches -> retrain (safe)."""
    run_dir = _make_trained_run_dir(tmp_path)
    (run_dir / runmod._TRAIN_FINGERPRINT_FILE).write_text("abc")
    assert runmod._train_freshness(run_dir, None) == "stale"


# --------------------------------------------------------------------------- #
# Digest is config-sensitive
# --------------------------------------------------------------------------- #

def test_digest_changes_with_config():
    base = {"scenario": {"num_vehicles": 20}, "algorithm": {"total_timesteps": 384000}}
    changed = {"scenario": {"num_vehicles": 20}, "algorithm": {"total_timesteps": 600000}}
    assert runmod._train_config_digest(base) != runmod._train_config_digest(changed)


def test_digest_changes_with_algorithm_implementation_version():
    unversioned = {
        "scenario": {"num_vehicles": 20},
        "algorithm": {"name": "jdls"},
    }
    versioned = {
        "scenario": {"num_vehicles": 20},
        "algorithm": {
            "name": "jdls",
            "implementation_version": "test_implementation_v1",
        },
    }
    assert runmod._train_config_digest(unversioned) != runmod._train_config_digest(
        versioned
    )


def test_digest_stable_for_equal_config_regardless_of_key_order():
    a = {"scenario": {"num_vehicles": 20}, "reward": {"omega_fail": 2.0}}
    b = {"reward": {"omega_fail": 2.0}, "scenario": {"num_vehicles": 20}}
    assert runmod._train_config_digest(a) == runmod._train_config_digest(b)


def test_digest_none_for_unserializable():
    # The defensive guard: json.dumps raises ValueError on a circular reference
    # (default=str can't rescue it) -> None -> always retrain (safe direction).
    circular: dict = {}
    circular["self"] = circular
    assert runmod._train_config_digest(circular) is None


def test_digest_payload_expands_scenario_defaults():
    """Energy params that live ONLY as ScenarioConfig defaults must reach the
    digest payload, so an in-place edit of one forces a retrain."""
    payload = digest_payload({"scenario": {"num_vehicles": 20}})
    assert payload["scenario"]["num_vehicles"] == 20
    # These never appear in any algorithm YAML — they're dataclass defaults.
    assert "switched_capacitance_coeff" in payload["scenario"]
    assert "tx_power_w" in payload["scenario"]


def test_digest_sensitive_to_scenario_energy_default():
    """A config that names no energy params still fingerprints on the resolved
    ScenarioConfig, so changing an energy default (kappa/tx_power) changes the
    digest. Emulated by pinning kappa to a non-default value."""
    from src.sim.scenario import ScenarioConfig

    other_kappa = ScenarioConfig().switched_capacitance_coeff * 10
    a = {"scenario": {"num_vehicles": 20}}
    b = {"scenario": {"num_vehicles": 20, "switched_capacitance_coeff": other_kappa}}
    assert runmod._train_config_digest(a) != runmod._train_config_digest(b)


def test_digest_payload_passthrough_for_scenarioless_config():
    """Non-dict / scenario-less configs are returned unchanged (keeps the DBAS
    eval-cache and circular-ref guards behaving as before)."""
    assert digest_payload({"version": 1}) == {"version": 1}
    assert digest_payload(None) is None


# --------------------------------------------------------------------------- #
# Worker writes the sidecar (only after a successful train)
# --------------------------------------------------------------------------- #

@pytest.fixture
def fake_train(monkeypatch, tmp_path):
    """Patch the heavy bits so ``_train_worker`` runs hermetically.

    ``_materialize_config`` -> a fixed dict; ``train_from_config`` -> writes a
    summary.json and returns its run_dir, mirroring the real contract.
    """
    cfg = {"scenario": {"num_vehicles": 20}, "algorithm": {"total_timesteps": 600000}}
    monkeypatch.setattr(runmod, "_materialize_config", lambda *a, **k: cfg)

    run_dir = tmp_path / "proposed_s1"

    def fake_train_from_config(config, output_root):
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "summary.json").write_text("{}")
        return {"run_dir": str(run_dir)}

    import src.rl.train as trainmod
    monkeypatch.setattr(trainmod, "train_from_config", fake_train_from_config)
    return cfg, run_dir


def _spec_dict(run_name: str = "proposed_s1") -> dict:
    return {
        "algorithm": "proposed",
        "run_name": run_name,
        "train_seed": 1,
        "run_name_suffix": "",
        "scenario_overrides": None,
    }


def _run_spec(run_name_suffix: str = "unit") -> RunSpec:
    return RunSpec(
        algorithm="proposed",
        train_seed=1,
        run_name_suffix=run_name_suffix,
        scenario_overrides_items=(),
        sweep_param=None,
        purpose="baseline_line",
    )


def test_worker_writes_matching_fingerprint(fake_train):
    cfg, run_dir = fake_train
    runmod._train_worker(_spec_dict(), (40000,), str(run_dir.parent))

    fingerprint = (run_dir / runmod._TRAIN_FINGERPRINT_FILE).read_text().strip()
    assert fingerprint == runmod._train_config_digest(cfg)
    # End-to-end: a freshly trained run is classified fresh by the gate.
    assert runmod._train_freshness(run_dir, runmod._train_config_digest(cfg)) == "fresh"


def test_worker_fingerprint_then_config_change_is_stale(fake_train, monkeypatch):
    cfg, run_dir = fake_train
    runmod._train_worker(_spec_dict(), (40000,), str(run_dir.parent))

    # Simulate an in-place config edit (same run_name, different values).
    new_cfg = {"scenario": {"num_vehicles": 20}, "algorithm": {"total_timesteps": 384000}}
    assert runmod._train_freshness(run_dir, runmod._train_config_digest(new_cfg)) == "stale"


def test_build_train_plan_classifies_existing_fresh_run(tmp_path, monkeypatch):
    spec = _run_spec()
    run_dir = spec.run_dir(tmp_path)
    config = {"scenario": {}}
    _complete_selected_bundle(run_dir, config)
    monkeypatch.setattr(runmod, "_materialize_config", lambda *a, **k: config)

    plan = runmod._build_train_plan(
        [spec], tmp_path, (40_000,),
        force=False, retrain_stale=False,
    )

    assert plan[0].freshness == "fresh"
    assert plan[0].action == "skip"


def test_build_train_plan_blocks_stale_and_partial_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(runmod, "_materialize_config", lambda *a, **k: {"scenario": {}})
    stale = _run_spec("stale")
    stale_dir = stale.run_dir(tmp_path)
    stale_dir.mkdir()
    (stale_dir / "summary.json").write_text("{}")
    (stale_dir / runmod._TRAIN_FINGERPRINT_FILE).write_text("old")
    partial = _run_spec("partial")
    partial.run_dir(tmp_path).mkdir()

    plan = runmod._build_train_plan(
        [stale, partial], tmp_path, (40_000,),
        force=False, retrain_stale=False,
    )

    assert [(entry.freshness, entry.action) for entry in plan] == [
        ("stale", "blocked"), ("partial", "blocked")
    ]


def test_retrain_stale_does_not_retrain_fresh_run(tmp_path, monkeypatch):
    config = {"scenario": {}}
    monkeypatch.setattr(runmod, "_materialize_config", lambda *a, **k: config)
    fresh = _run_spec("fresh")
    fresh_dir = fresh.run_dir(tmp_path)
    _complete_selected_bundle(fresh_dir, config)
    stale = _run_spec("stale2")
    stale_dir = stale.run_dir(tmp_path)
    stale_dir.mkdir()
    (stale_dir / "summary.json").write_text("{}")

    plan = runmod._build_train_plan(
        [fresh, stale], tmp_path, (40_000,),
        force=False, retrain_stale=True,
    )

    assert [entry.action for entry in plan] == ["skip", "retrain"]


def test_training_record_is_written_atomically(tmp_path):
    run_dir = tmp_path / "proposed_s1"
    run_dir.mkdir()
    result = {
        "status": "ok",
        "elapsed_s": 12.5,
        "result": {
            "run_dir": str(run_dir), "run_name": "proposed_s1",
            "algorithm": "proposed", "train_seed": 1,
            "config_digest": "abc",
        },
    }

    runmod._write_training_record(result, run_id="invocation-1")

    import json
    record = json.loads((run_dir / "training_record.json").read_text())
    assert record["elapsed_s"] == 12.5
    assert record["config_digest"] == "abc"
    assert record["run_id"] == "invocation-1"
    assert not list(run_dir.glob(".training_record.*.tmp"))


def test_serial_manifest_keeps_success_before_later_failure(tmp_path):
    import json

    from src.parallel import TaskSpec

    def succeed():
        return {"value": 1}

    def fail():
        raise RuntimeError("boom")

    manifest = tmp_path / "invocation.jsonl"
    persisted: list[str] = []
    tasks = [
        TaskSpec(func=succeed, label="train/good"),
        TaskSpec(func=fail, label="train/bad"),
    ]

    with pytest.raises(RuntimeError, match="boom"):
        runmod._run_tasks_with_manifest(
            tasks,
            parallel=False,
            max_workers=None,
            fail_fast=True,
            manifest=manifest,
            on_success=lambda result: persisted.append(result["label"]),
        )

    events = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert [(event["event"], event["label"]) for event in events] == [
        ("done", "train/good"),
        ("failed", "train/bad"),
    ]
    assert persisted == ["train/good"]
