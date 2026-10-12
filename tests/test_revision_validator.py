from __future__ import annotations

import csv
from pathlib import Path

import yaml

from scripts.validate_revision_outputs import (
    _validate_expected_row_provenance,
    _validate_hardware_payload,
    _validate_priority_success_metrics,
    _validate_pruning_equivalent_cells,
    _validate_row_provenance_fields,
    validate_final,
    validate_preflight,
)
from src.evaluation.revision import dbas_calibration_contract_digest, load_revision_config
from src.rl.eval import SWEEP_METRIC_NAMES


def test_revision_preflight_protocol_is_self_consistent():
    assert validate_preflight(load_revision_config()) == []


def test_row_provenance_rejects_unknown_artifact_kind():
    issues: list[str] = []
    _validate_row_provenance_fields(
        [
            {
                "config_fingerprint": "abc",
                "checkpoint_sha256": "def",
                "artifact_kind": "shape_only",
            }
        ],
        "headline",
        issues,
    )
    assert issues == ["headline: invalid artifact_kind"]


def test_expected_provenance_rejects_nonempty_but_stale_digests():
    from src.evaluation.provenance import ArtifactProvenance

    issues: list[str] = []
    _validate_expected_row_provenance(
        [
            {
                "algorithm": "jdls",
                "train_seed": "123",
                "config_fingerprint": "old-config",
                "checkpoint_sha256": "old-checkpoint",
                "artifact_kind": "trained_checkpoint",
            },
            {
                "algorithm": "jdls",
                "train_seed": "123",
                "config_fingerprint": "old-config",
                "checkpoint_sha256": "old-checkpoint",
                "artifact_kind": "trained_checkpoint",
            },
        ],
        "headline",
        lambda _row: ArtifactProvenance(
            config_fingerprint="current-config",
            checkpoint_sha256="current-checkpoint",
        ),
        issues,
    )

    assert issues == [
        "headline: algorithm=jdls, train_seed=123, expected_config=current-conf "
        "has stale provenance fields [config_fingerprint, checkpoint_sha256] "
        "across 2 row(s)"
    ]


def test_priority_success_metrics_validate_exact_decomposition():
    issues: list[str] = []
    _validate_priority_success_metrics(
        [
            {
                "task_count_high": 1,
                "task_failure_count_high": 0,
                "task_success_rate_high": 1.0,
                "task_count_mid": 1,
                "task_failure_count_mid": 0,
                "task_success_rate_mid": 1.0,
                "task_count_low": 1,
                "task_failure_count_low": 1,
                "task_success_rate_low": 0.0,
                "task_failure_count": 1,
                "task_success_rate": 2.0 / 3.0,
            }
        ],
        "headline",
        issues,
    )
    assert issues == []


def test_priority_success_metrics_reject_inconsistent_aggregate():
    issues: list[str] = []
    _validate_priority_success_metrics(
        [
            {
                "task_count_high": 1,
                "task_failure_count_high": 0,
                "task_success_rate_high": 1.0,
                "task_count_mid": 1,
                "task_failure_count_mid": 0,
                "task_success_rate_mid": 1.0,
                "task_count_low": 1,
                "task_failure_count_low": 1,
                "task_success_rate_low": 0.0,
                "task_failure_count": 0,
                "task_success_rate": 1.0,
            }
        ],
        "headline",
        issues,
    )
    assert len(issues) == 1
    assert "priority failures do not sum to aggregate" in issues[0]


def test_pruning_equivalence_validator_compares_measured_metrics():
    common = {metric: 1.0 for metric in SWEEP_METRIC_NAMES}
    identity = {
        "algorithm": "proposed",
        "r0_m": 80.0,
        "train_seed": 123,
        "eval_seed": 40000,
    }
    rho_one = {**common, **identity, "diversity_factor": 1.0}
    rho_two = {**common, **identity, "diversity_factor": 2.0}
    issues: list[str] = []

    _validate_pruning_equivalent_cells([rho_one, rho_two], issues)

    assert issues == []

    rho_two["social_welfare"] = 1.1
    _validate_pruning_equivalent_cells([rho_one, rho_two], issues)
    assert issues == [
        "pruning sensitivity rho=1/rho=2 values differ for "
        "(80.0, 123, 40000), metric=social_welfare"
    ]


def test_preflight_rejects_validation_test_leakage():
    config = load_revision_config()
    config["seeds"]["test"] = config["seeds"]["validation"]
    assert "evaluation seed banks validation and test overlap" in validate_preflight(config)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_final_validator_rejects_duplicate_headline_seed_identity(tmp_path):
    config = load_revision_config()
    config["paths"]["outputs_root"] = str(tmp_path / "outputs")
    calibration = tmp_path / "dbas_selected.yaml"
    calibration.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "algorithm.fixed_price_fraction": 0.1,
                "validation_seed_count": len(config["seeds"]["validation"]),
                "calibration_contract_digest": dbas_calibration_contract_digest(config),
            }
        ),
        encoding="utf-8",
    )
    config["baseline_calibration"]["dbas_selection_file"] = str(calibration)
    row = {
        "algorithm": "proposed",
        "train_seed": 123,
        "eval_seed": 40000,
        "social_welfare": 1.0,
        "raw_rsu_reward": 1.0,
        "perception_quality_total": 1.0,
    }
    _write_csv(
        Path(config["paths"]["outputs_root"]) / "headline" / "episodes.csv",
        [row, row],
    )

    issues = validate_final(config)

    assert "headline proposed: duplicate train/eval seed rows" in issues


def test_final_validator_rejects_shape_only_timing_rows(tmp_path):
    config = load_revision_config()
    config["paths"]["outputs_root"] = str(tmp_path / "outputs")
    calibration = tmp_path / "dbas_selected.yaml"
    calibration.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "algorithm.fixed_price_fraction": 0.1,
                "validation_seed_count": len(config["seeds"]["validation"]),
                "calibration_contract_digest": dbas_calibration_contract_digest(config),
            }
        ),
        encoding="utf-8",
    )
    config["baseline_calibration"]["dbas_selection_file"] = str(calibration)
    _write_csv(
        Path(config["paths"]["outputs_root"]) / "tables" / "inference_latency.csv",
        [
            {
                "num_vehicles": 10,
                "num_tasks": 3,
                "benchmark_kind": "shape_controlled_microbenchmark",
                "config_fingerprint": "",
                "checkpoint_sha256": "",
            }
        ],
    )

    issues = validate_final(config)

    assert "inference timing contains non-checkpoint evidence" in issues


def test_hardware_validator_requires_full_physical_gpu_inventory():
    issues: list[str] = []
    _validate_hardware_payload(
        {
            "hardware_schema_version": 1,
            "hostname": "server",
            "cpu_model": "Example CPU",
            "physical_cpu_count": 48,
            "logical_cpu_count": 96,
            "memory_total_gib": 128,
            "python": "3.12",
            "torch": "2.5",
            "git_commit": "abc123",
            "physical_gpu_count": 4,
            "physical_gpus": [
                {"name": "RTX 3090", "memory_total_gib": 24, "driver": "555"} for _ in range(4)
            ],
        },
        "server",
        issues,
    )
    assert issues == []

    issues = []
    _validate_hardware_payload(
        {
            "hardware_schema_version": 1,
            "hostname": "server",
            "cpu_model": "Example CPU",
            "physical_cpu_count": 48,
            "logical_cpu_count": 96,
            "memory_total_gib": 128,
            "python": "3.12",
            "torch": "2.5",
            "git_commit": "abc123",
            "physical_gpu_count": 0,
            "physical_gpus": [],
        },
        "server",
        issues,
    )
    assert "server: incomplete physical GPU inventory" in issues
