"""Fail-closed provenance for trained research artifacts.

Exactness label: implementation convenience only.  This module never changes
the simulator or learning problem; it proves that an evaluation consumed the
checkpoint trained from the currently expected materialized configuration.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

TRAIN_FINGERPRINT_FILE = "config_fingerprint.txt"

PROVENANCE_BUNDLE_FIELDS: tuple[str, ...] = (
    "algorithm",
    "train_seed",
    "config_fingerprint",
    "checkpoint_sha256",
    "normalization_sha256",
    "selection_sha256",
    "artifact_kind",
    "evaluations_sha256",
)

@dataclass(frozen=True)
class ArtifactProvenance:
    config_fingerprint: str
    checkpoint_sha256: str
    normalization_sha256: str | None = None
    selection_sha256: str = ""
    artifact_kind: str = "trained_checkpoint"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def digest_payload(config: dict[str, Any]) -> dict[str, Any]:
    """Expand effective scenario defaults before hashing a materialized config."""

    if not isinstance(config, dict):
        return config
    from src.rl.training_config import resolve_effective_training_config

    normalized = resolve_effective_training_config(config)
    scenario = normalized.get("scenario")
    if not isinstance(scenario, dict):
        return normalized
    from src.sim.scenario import ScenarioConfig

    try:
        resolved = asdict(ScenarioConfig(**scenario))
    except (TypeError, ValueError):
        resolved = asdict(ScenarioConfig())
        resolved.update(scenario)
    return {**normalized, "scenario": resolved}


def config_digest(config: dict[str, Any]) -> str | None:
    """Return the canonical SHA-256 for one complete training configuration."""

    try:
        payload = json.dumps(digest_payload(config), sort_keys=True, default=str)
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_train_fingerprint(run_dir: str | Path) -> str | None:
    try:
        value = (Path(run_dir) / TRAIN_FINGERPRINT_FILE).read_text(
            encoding="utf-8"
        ).strip()
    except OSError:
        return None
    return value or None


def train_freshness(
    run_dir: str | Path,
    expected_digest: str | None,
    *,
    expected_config: dict[str, Any] | None = None,
) -> str:
    """Classify a cached training run without accepting incomplete evidence.

    Passing only a digest performs a lightweight fingerprint check used by
    callers and unit tests.  Formal orchestration also passes the materialized
    config, which lets this gate distinguish an incomplete selected-checkpoint
    bundle from a genuinely reusable run.
    """

    path = Path(run_dir)
    if not (path / "summary.json").exists():
        return "absent"
    stored = read_train_fingerprint(path)
    if expected_digest is None or stored != expected_digest:
        return "stale"
    if expected_config is None:
        return "fresh"
    algorithm = str(expected_config.get("algorithm", {}).get("name", "proposed"))
    if algorithm == "dbas":
        return "fresh"
    try:
        validate_run_artifact(path, expected_config=expected_config)
    except FileNotFoundError:
        return "partial"
    except ValueError:
        return "stale"
    return "fresh"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance_bundle_digest(rows: list[dict[str, Any]]) -> str:
    """Hash a canonical per-seed source manifest independently of CSV typing."""

    normalized = [
        {
            field: "" if row.get(field) is None else str(row.get(field, ""))
            for field in PROVENANCE_BUNDLE_FIELDS
        }
        for row in rows
    ]
    payload = json.dumps(
        sorted(normalized, key=lambda row: (row["algorithm"], row["train_seed"])),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_run_artifact(
    run_dir: str | Path,
    *,
    expected_config: dict[str, Any],
    checkpoint_name: str | None = None,
) -> ArtifactProvenance:
    """Validate expected config, saved snapshot, fingerprint, and checkpoint.

    A stored fingerprint is not accepted merely because it is present: all
    three independently derived configuration identities must agree.
    """

    from src.rl.checkpoint_selection import (
        SELECTED_VECNORMALIZE,
        SELECTION_MANIFEST,
        cpu_validation_ranges,
        robust_checkpoint_score,
        selected_model_name,
        validation_grid_digest,
    )

    path = Path(run_dir)
    algorithm = str(expected_config.get("algorithm", {}).get("name", "proposed"))
    expected_checkpoint_name = selected_model_name(algorithm)
    if checkpoint_name is not None and checkpoint_name != expected_checkpoint_name:
        raise ValueError(
            f"formal artifact must use {expected_checkpoint_name}, not {checkpoint_name}"
        )
    required = (
        path / "summary.json",
        path / "config_snapshot.yaml",
        path / TRAIN_FINGERPRINT_FILE,
        path / expected_checkpoint_name,
        path / SELECTION_MANIFEST,
        path / "eval" / "evaluations.npz",
    )
    missing = [item for item in required if not item.exists()]
    if missing:
        raise FileNotFoundError(
            f"incomplete trained artifact under {path}: "
            + ", ".join(str(item) for item in missing)
        )

    expected = config_digest(expected_config)
    if expected is None:
        raise ValueError(f"expected configuration for {path} is not digestible")
    snapshot = yaml.safe_load(
        (path / "config_snapshot.yaml").read_text(encoding="utf-8")
    ) or {}
    snapshot_digest = config_digest(snapshot)
    stored = read_train_fingerprint(path)
    if stored != expected or snapshot_digest != expected:
        raise ValueError(
            f"stale or inconsistent artifact under {path}: "
            f"expected={expected}, snapshot={snapshot_digest}, stored={stored}"
        )

    try:
        selection = json.loads(
            (path / SELECTION_MANIFEST).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid selection manifest under {path}") from exc
    checkpoint_path = path / expected_checkpoint_name
    checkpoint_hash = sha256_file(checkpoint_path)
    if selection.get("selection_metric") != "social_welfare":
        raise ValueError(f"invalid selection metric under {path}")
    expected_ranges = cpu_validation_ranges(expected_config.get("evaluation", {}))
    expected_rule = (
        "maximin_cpu_band_social_welfare"
        if expected_ranges else "mean_social_welfare"
    )
    if selection.get("selection_rule", "mean_social_welfare") != expected_rule:
        raise ValueError(f"invalid selection rule under {path}")
    if selection.get("config_fingerprint") != expected:
        raise ValueError(f"selection manifest config mismatch under {path}")
    if selection.get("model_file") != expected_checkpoint_name:
        raise ValueError(f"selection manifest model mismatch under {path}")
    if selection.get("model_sha256") != checkpoint_hash:
        raise ValueError(f"selected checkpoint hash mismatch under {path}")
    expected_seeds = [
        int(value) for value in expected_config.get("evaluation", {}).get("seed_list", [])
    ]
    if selection.get("validation_seeds") != expected_seeds:
        raise ValueError(f"selection validation seeds mismatch under {path}")
    try:
        with np.load(path / "eval" / "evaluations.npz", allow_pickle=False) as data:
            timesteps = np.asarray(data["timesteps"], dtype=np.int64)
            welfare = np.asarray(data["social_welfare"], dtype=np.float64)
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"invalid validation trajectory under {path}") from exc
    selected_timestep = int(selection.get("selected_timestep", -1))
    matches = np.flatnonzero(timesteps == selected_timestep)
    if len(matches) != 1:
        raise ValueError(f"selected timestep is absent or duplicated under {path}")
    selected_values = [float(value) for value in welfare[int(matches[0])]]
    if expected_ranges:
        if int(selection.get("schema_version", 0)) != 2:
            raise ValueError(f"CPU robust selection schema mismatch under {path}")
        if selection.get("cpu_validation_ranges") != [
            list(value) for value in expected_ranges
        ]:
            raise ValueError(f"CPU validation ranges mismatch under {path}")
        if selection.get("validation_grid_digest") != validation_grid_digest(
            expected_ranges
        ):
            raise ValueError(f"CPU validation grid digest mismatch under {path}")
        score = robust_checkpoint_score(
            selected_values,
            cell_count=len(expected_ranges),
            seeds_per_cell=len(expected_seeds),
        )
        if not np.isclose(
            score.minimum_cell_social_welfare,
            float(selection.get("selected_metric_value", np.nan)),
            rtol=1e-10,
            atol=1e-10,
        ):
            raise ValueError(f"selected maximin metric mismatch under {path}")
        if not np.isclose(
            score.macro_mean_social_welfare,
            float(selection.get("selected_macro_mean_social_welfare", np.nan)),
            rtol=1e-10,
            atol=1e-10,
        ):
            raise ValueError(f"selected macro metric mismatch under {path}")
        if not np.allclose(
            score.cell_social_welfare_means,
            selection.get("selected_cell_social_welfare_means", ()),
            rtol=1e-10,
            atol=1e-10,
        ):
            raise ValueError(f"selected CPU cell metrics mismatch under {path}")
        try:
            with np.load(path / "eval" / "evaluations.npz", allow_pickle=False) as data:
                stored_ranges = np.asarray(data["cpu_validation_ranges"], dtype=np.float64)
                cell_indices = np.asarray(data["validation_cell_index"], dtype=np.int64)
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError(f"CPU validation trajectory metadata missing under {path}") from exc
        if not np.array_equal(stored_ranges, np.asarray(expected_ranges)):
            raise ValueError(f"CPU validation trajectory ranges mismatch under {path}")
        expected_indices = np.tile(
            np.repeat(np.arange(len(expected_ranges)), len(expected_seeds)),
            (len(timesteps), 1),
        )
        if not np.array_equal(cell_indices, expected_indices):
            raise ValueError(f"CPU validation cell identities mismatch under {path}")
    else:
        selected_mean = float(np.mean(selected_values))
        if not np.isclose(
            selected_mean, float(selection.get("selected_metric_value", np.nan)),
            rtol=1e-10, atol=1e-10,
        ):
            raise ValueError(f"selected metric does not match validation data under {path}")

    normalization_hash: str | None = None
    norm_required = bool(expected_config.get("normalization", {}).get("norm_obs", False))
    if norm_required:
        norm_path = path / SELECTED_VECNORMALIZE
        if selection.get("normalization_file") != SELECTED_VECNORMALIZE or not norm_path.exists():
            raise FileNotFoundError(f"selected normalization state missing under {path}")
        normalization_hash = sha256_file(norm_path)
        if selection.get("normalization_sha256") != normalization_hash:
            raise ValueError(f"selected normalization hash mismatch under {path}")
    elif selection.get("normalization_file") is not None:
        raise ValueError(f"unexpected selected normalization under {path}")

    return ArtifactProvenance(
        config_fingerprint=expected,
        checkpoint_sha256=checkpoint_hash,
        normalization_sha256=normalization_hash,
        selection_sha256=sha256_file(path / SELECTION_MANIFEST),
    )


CONTRACT_FILENAME = "evaluation_contract.json"
FORMAL_STATUS = "formal_revision_evaluation"


def evaluation_contract_payload(contract: dict[str, Any]) -> dict[str, Any]:
    """Return the canonical sidecar payload for one formal evaluation."""

    body = {
        "schema_version": 1,
        "evidence_status": FORMAL_STATUS,
        **contract,
    }
    normalized = json.loads(json.dumps(body, sort_keys=True, default=str))
    digest = config_digest(normalized)
    if digest is None:
        raise ValueError("evaluation contract is not digestible")
    return {**normalized, "contract_digest": digest}


def write_evaluation_contract(output_dir: Path, contract: dict[str, Any]) -> Path:
    """Atomically write the sidecar after both formal CSVs are complete."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / CONTRACT_FILENAME
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(evaluation_contract_payload(contract), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path
