from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.evaluation.provenance import (
    TRAIN_FINGERPRINT_FILE,
    config_digest,
    provenance_bundle_digest,
    sha256_file,
    validate_run_artifact,
)
from src.evaluation.sweep import evaluate_run_dir


def test_trained_evaluation_requires_expected_config(tmp_path):
    with pytest.raises(ValueError, match="expected_config is required"):
        evaluate_run_dir("proposed", tmp_path, (40_000,))


def _artifact(
    tmp_path: Path,
    config: dict,
    *,
    model_filename: str = "selected_model.zip",
) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(json.dumps({"ok": True}))
    (run_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(config))
    digest = config_digest(config)
    assert digest is not None
    (run_dir / TRAIN_FINGERPRINT_FILE).write_text(digest)
    model = run_dir / model_filename
    model.write_bytes(b"checkpoint")
    normalization_file = None
    normalization_sha256 = None
    if config.get("normalization", {}).get("norm_obs", False):
        normalization = run_dir / "selected_vecnormalize.pkl"
        normalization.write_bytes(b"normalization")
        normalization_file = normalization.name
        normalization_sha256 = sha256_file(normalization)
    eval_dir = run_dir / "eval"
    eval_dir.mkdir()
    np.savez(
        eval_dir / "evaluations.npz",
        timesteps=np.array([10]),
        social_welfare=np.array([[1.0, 3.0]]),
    )
    (run_dir / "selection.json").write_text(json.dumps({
        "selection_metric": "social_welfare",
        "selected_timestep": 10,
        "selected_metric_value": 2.0,
        "validation_seeds": config.get("evaluation", {}).get("seed_list", []),
        "config_fingerprint": digest,
        "model_file": model_filename,
        "model_sha256": sha256_file(model),
        "normalization_file": normalization_file,
        "normalization_sha256": normalization_sha256,
    }))
    return run_dir


def test_validate_run_artifact_requires_three_matching_config_identities(tmp_path):
    config = {
        "scenario": {"num_vehicles": 20},
        "provenance": {"revision_id": "r3", "training_semantics_version": 1},
    }
    run_dir = _artifact(tmp_path, config)

    result = validate_run_artifact(run_dir, expected_config=config)

    assert result.config_fingerprint == config_digest(config)
    assert len(result.checkpoint_sha256) == 64


def test_jdls_artifact_does_not_require_normalization_state(tmp_path):
    config = {
        "algorithm": {"name": "jdls"},
        "normalization": {"norm_obs": False, "norm_reward": False},
        "scenario": {"num_vehicles": 20},
        "provenance": {"revision_id": "r5", "training_semantics_version": 4},
    }
    run_dir = _artifact(tmp_path, config, model_filename="selected_model.pt")

    result = validate_run_artifact(run_dir, expected_config=config)

    assert result.normalization_sha256 is None
    assert not (run_dir / "selected_vecnormalize.pkl").exists()


@pytest.mark.parametrize("corruption", ["expected", "snapshot", "stored"])
def test_validate_run_artifact_rejects_stale_or_tampered_config(tmp_path, corruption):
    config = {
        "scenario": {"num_vehicles": 20},
        "provenance": {"revision_id": "r3", "training_semantics_version": 1},
    }
    run_dir = _artifact(tmp_path, config)
    expected = config
    if corruption == "expected":
        expected = {**config, "provenance": {**config["provenance"],
                    "training_semantics_version": 2}}
    elif corruption == "snapshot":
        changed = {**config, "scenario": {"num_vehicles": 30}}
        (run_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(changed))
    else:
        (run_dir / TRAIN_FINGERPRINT_FILE).write_text("0" * 64)

    with pytest.raises(ValueError, match="stale or inconsistent"):
        validate_run_artifact(
            run_dir, expected_config=expected, checkpoint_name="selected_model.zip"
        )


@pytest.mark.parametrize("filename", ["selected_model.zip", "selection.json"])
def test_validate_run_artifact_rejects_tampered_selected_bundle(tmp_path, filename):
    config = {
        "algorithm": {"name": "proposed"},
        "scenario": {"num_vehicles": 20},
        "provenance": {"revision_id": "r4", "training_semantics_version": 3},
    }
    run_dir = _artifact(tmp_path, config)
    with (run_dir / filename).open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError):
        validate_run_artifact(run_dir, expected_config=config)


def test_validate_run_artifact_rejects_mismatched_selected_normalization(tmp_path):
    config = {
        "algorithm": {"name": "proposed"},
        "normalization": {"norm_obs": True},
        "scenario": {"num_vehicles": 20},
        "provenance": {"revision_id": "r4", "training_semantics_version": 3},
    }
    run_dir = _artifact(tmp_path, config)
    with (run_dir / "selected_vecnormalize.pkl").open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(ValueError, match="normalization hash mismatch"):
        validate_run_artifact(run_dir, expected_config=config)


def test_formal_freshness_rejects_incomplete_selected_bundle(tmp_path):
    config = {
        "algorithm": {"name": "proposed"},
        "scenario": {"num_vehicles": 20},
        "provenance": {"revision_id": "r4", "training_semantics_version": 3},
    }
    run_dir = _artifact(tmp_path, config)
    digest = config_digest(config)
    assert digest is not None
    (run_dir / "selection.json").unlink()

    from src.evaluation.provenance import train_freshness

    assert train_freshness(
        run_dir, digest, expected_config=config
    ) == "partial"


def test_semantics_version_participates_in_digest():
    base = {
        "scenario": {},
        "provenance": {"revision_id": "r3", "training_semantics_version": 1},
    }
    changed = {
        **base,
        "provenance": {"revision_id": "r3", "training_semantics_version": 2},
    }
    assert config_digest(base) != config_digest(changed)


def test_provenance_bundle_digest_is_stable_across_csv_scalar_types():
    numeric = [{
        "algorithm": "proposed",
        "train_seed": 123,
        "config_fingerprint": "config",
        "checkpoint_sha256": "checkpoint",
        "normalization_sha256": None,
        "selection_sha256": "selection",
        "artifact_kind": "trained_checkpoint",
        "evaluations_sha256": "curve",
    }]
    strings = [{
        key: "" if value is None else str(value)
        for key, value in numeric[0].items()
    }]

    assert provenance_bundle_digest(numeric) == provenance_bundle_digest(strings)
