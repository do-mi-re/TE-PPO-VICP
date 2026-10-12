"""Fail-closed completeness validator for the IoT-J revision evidence bundle.

Exactness label: implementation convenience only.  Validation checks evidence
identity and completeness without changing model or training semantics.
"""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import json
import math
import statistics
import struct
from pathlib import Path
from typing import Any

import yaml

from src.evaluation import orchestration as orch
from src.evaluation.ablation import (
    ABLATION_ARMS,
    ABLATION_CONDITIONAL_RATIO_METRIC,
    ABLATION_ENGINES,
    ABLATION_UNCONDITIONAL_METRICS,
    ablation_run_dir,
    build_ablation_config,
    calibrated_price_from_run,
    conditional_ratio_summary,
    paired_ablation_ratio_delta_summary,
)
from src.evaluation.provenance import (
    CONTRACT_FILENAME,
    FORMAL_STATUS,
    ArtifactProvenance,
    config_digest,
    provenance_bundle_digest,
    sha256_file,
    validate_run_artifact,
)
from src.evaluation.revision import (
    ALL_ALGORITHMS,
    STOCHASTIC_ALGORITHMS,
    cpu_robustness_run_dir,
    cpu_robustness_specs,
    formal_algorithm_config,
    headline_run_dir,
    headline_specs,
    load_cpu_robustness_dbas_calibration,
    load_dbas_calibration,
    load_revision_config,
    pruning_overrides,
    pruning_sensitivity_reported_cells,
    pruning_sensitivity_run_dir,
    pruning_sensitivity_specs,
    pruning_sensitivity_training_rho,
    revision_roots,
    scalability_specs,
    timing_scalability_specs,
    train_seeds_for,
)
from src.evaluation.sweep import load_algorithm_config
from src.plotting.revision_style import (
    FIGSIZE_SINGLE_COLUMN,
    FIGURE_DPI,
)
from src.rl.checkpoint_selection import selected_model_name
from src.rl.eval import SWEEP_METRIC_NAMES

ProvenanceCache = dict[tuple[str, str, str], ArtifactProvenance]
REPORT_METRICS = ("social_welfare", "raw_rsu_reward", "perception_quality_total")


def _png_dimensions(path: Path) -> tuple[int, int]:
    """Read PNG dimensions from its fixed IHDR header without image dependencies."""

    header = path.read_bytes()[:24]
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a valid PNG")
    return struct.unpack(">II", header[16:24])


def _checkpoint_name(algorithm: str) -> str:
    return selected_model_name(algorithm)


def _optional_float(value: object) -> float:
    if value in (None, ""):
        return math.nan
    return float(value)


def _validate_seed_reduction(
    summary_rows: list[dict[str, str]],
    seed_rows: list[dict[str, str]],
    *,
    metrics: tuple[str, ...],
    label: str,
    issues: list[str],
) -> None:
    """Require final statistics to use equally weighted training-seed rows."""

    grouped: dict[tuple[str, float], list[dict[str, str]]] = {}
    try:
        for row in seed_rows:
            key = (row["algorithm"], float(row["sweep_value"]))
            grouped.setdefault(key, []).append(row)
        reported = {
            (row["algorithm"], float(row["sweep_value"])): row
            for row in summary_rows
        }
        if set(grouped) != set(reported):
            raise ValueError("summary and seed-summary cells differ")
        for key, group in grouped.items():
            if len({int(row["train_seed"]) for row in group}) != len(group):
                raise ValueError(f"duplicate training seed in cell {key}")
            for metric in metrics:
                values = [_optional_float(row[metric]) for row in group]
                values = [value for value in values if math.isfinite(value)]
                if not values:
                    raise ValueError(f"no finite {metric} values in cell {key}")
                expected_mean = statistics.fmean(values)
                expected_std = statistics.stdev(values) if len(values) > 1 else math.nan
                expected_ci = (
                    1.96 * expected_std / math.sqrt(len(values))
                    if len(values) > 1
                    else math.nan
                )
                row = reported[key]
                actual_mean = _optional_float(row[f"{metric}_seed_mean"])
                actual_std = _optional_float(row[f"{metric}_seed_std"])
                actual_ci = _optional_float(row[f"{metric}_seed_ci95"])
                actual_n = int(row[f"{metric}_seed_replicates"])
                if not math.isclose(
                    actual_mean, expected_mean, rel_tol=1e-10, abs_tol=1e-12
                ):
                    raise ValueError(f"incorrect {metric} seed mean in cell {key}")
                if actual_n != len(values):
                    raise ValueError(f"incorrect {metric} seed count in cell {key}")
                for actual, expected, statistic_name in (
                    (actual_std, expected_std, "std"),
                    (actual_ci, expected_ci, "ci95"),
                ):
                    if math.isnan(expected):
                        if not math.isnan(actual):
                            raise ValueError(
                                f"incorrect {metric} seed {statistic_name} in cell {key}"
                            )
                    elif not math.isclose(
                        actual, expected, rel_tol=1e-10, abs_tol=1e-12
                    ):
                        raise ValueError(
                            f"incorrect {metric} seed {statistic_name} in cell {key}"
                        )
    except (KeyError, TypeError, ValueError) as exc:
        issues.append(f"{label}: invalid training-seed reduction: {exc}")


def _trained_provenance(
    algorithm: str,
    run_dir: Path,
    expected_config: dict[str, Any],
    cache: ProvenanceCache,
) -> ArtifactProvenance:
    digest = config_digest(expected_config)
    if digest is None:
        raise ValueError("expected training configuration is not digestible")
    checkpoint = _checkpoint_name(algorithm)
    key = (str(run_dir), digest, checkpoint)
    if key not in cache:
        cache[key] = validate_run_artifact(
            run_dir,
            expected_config=expected_config,
            checkpoint_name=checkpoint,
        )
    return cache[key]


def _stateless_provenance(config: dict[str, Any]) -> ArtifactProvenance:
    digest = config_digest(config)
    if digest is None:
        raise ValueError("expected stateless configuration is not digestible")
    return ArtifactProvenance(
        config_fingerprint=digest,
        checkpoint_sha256="stateless",
        selection_sha256="stateless",
        artifact_kind="stateless_config",
    )


def _compare_provenance(
    row: dict[str, str],
    expected: ArtifactProvenance,
    label: str,
    issues: list[str],
) -> None:
    for field in _provenance_mismatch_fields(row, expected):
        issues.append(f"{label}: {field} does not match current config/artifact")


def _provenance_mismatch_fields(
    row: dict[str, str], expected: ArtifactProvenance
) -> tuple[str, ...]:
    """Return stale provenance fields without changing validation semantics."""

    mismatches: list[str] = []
    for field, value in expected.as_dict().items():
        actual = row.get(field)
        if value in (None, "") and actual in (None, ""):
            continue
        if str(actual) != str(value):
            mismatches.append(field)
    return tuple(mismatches)


def _headline_expectations(
    cfg: dict[str, Any],
    cache: ProvenanceCache,
) -> dict[tuple[str, int], ArtifactProvenance]:
    validation_seeds = tuple(cfg["seeds"]["validation"])
    expectations: dict[tuple[str, int], ArtifactProvenance] = {}
    for spec in headline_specs(cfg):
        key = (spec.algorithm, int(spec.train_seed))
        if spec.algorithm == "dbas":
            expectations[key] = _stateless_provenance(formal_algorithm_config("dbas", cfg))
        else:
            expected_config = orch.materialize_config(
                spec.to_worker_dict(), validation_seeds
            )
            expectations[key] = _trained_provenance(
                spec.algorithm,
                headline_run_dir(spec.algorithm, spec.train_seed, cfg),
                expected_config,
                cache,
            )
    return expectations


def _ablation_expectations(
    cfg: dict[str, Any],
    cache: ProvenanceCache,
) -> dict[tuple[str, int], ArtifactProvenance]:
    base_config = formal_algorithm_config("proposed", cfg)
    expectations: dict[tuple[str, int], ArtifactProvenance] = {}
    for seed in cfg["seeds"]["ablation_train"]:
        seed = int(seed)
        full_dir = ablation_run_dir("full", seed, cfg)
        fixed_price = calibrated_price_from_run(full_dir)
        for arm in ABLATION_ARMS:
            expected_config = build_ablation_config(
                arm,
                seed,
                fixed_price=fixed_price if arm == "no_dynamic_pricing" else None,
                revision_config=cfg,
                base_config=base_config,
            )
            engine = ABLATION_ENGINES[arm]
            expectations[(arm, seed)] = _trained_provenance(
                engine, ablation_run_dir(arm, seed, cfg), expected_config, cache
            )
    return expectations


def _pruning_stateless_config(row: dict[str, str], cfg: dict[str, Any]) -> dict[str, Any]:
    override = pruning_overrides(
        row["sensitivity_mode"],
        float(row["r0_m"]),
        float(row["diversity_factor"]),
        config=cfg,
    )
    # DBAS does not acquire TE-PPO's pruning component during a fair replay.
    override = {
        "enable_candidate_pruning": False,
        "sensing_range_m": override["sensing_range_m"],
    }
    config = copy.deepcopy(formal_algorithm_config("dbas", cfg))
    config.setdefault("scenario", {}).update(override)
    return config


def _dynamic_override_map(cfg: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    scale = cfg["scalability"]
    overrides = {
        ("occlusion", str(label)): {"occlusion_range": tuple(float(value) for value in band)}
        for label, band in scale["occlusion_bands"].items()
    }
    overrides.update(
        {
            ("path_loss", f"alpha_{float(value):g}"): {"path_loss_exponent": float(value)}
            for value in scale["path_loss_grid"]
        }
    )
    overrides.update(
        {
            ("rsu_bandwidth", f"{float(value) / 1e6:g}MHz"): {"system_bandwidth_hz": float(value)}
            for value in scale["rsu_bandwidth_grid_hz"]
        }
    )
    return overrides


def _validate_cpu_robustness(
    cfg: dict[str, Any], outputs: Path, issues: list[str]
) -> None:
    """Validate the dedicated CPU-domain-randomized evidence and gates."""

    contract = cfg["cpu_robustness"]
    root = outputs / "cpu_robustness" / "formal"
    rows = _rows(root / "episodes.csv", issues)
    summary = _rows(root / "summary.csv", issues)
    seed_summary = _rows(root / "seed_summary.csv", issues)
    ranges = tuple(contract["cpu_capacity_ranges_cycles_per_s"])
    values = tuple((low + high) / 2e9 for low, high in ranges)
    train_seeds = tuple(cfg["seeds"][contract["training_seed_bank"]])
    eval_seeds = tuple(cfg["seeds"][contract["evaluation_seed_bank"]])
    _validate_evaluation_contract(
        root,
        label="CPU robustness",
        expected={
            "revision_id": cfg["revision_id"],
            "revision_schema_version": int(cfg["schema_version"]),
            "cpu_robustness_protocol_version": int(contract["protocol_version"]),
            "experiment": "cpu_robustness_formal",
            "training_seed_bank": contract["training_seed_bank"],
            "training_seeds": list(train_seeds),
            "evaluation_seed_bank": contract["evaluation_seed_bank"],
            "evaluation_seeds": list(eval_seeds),
            "cpu_capacity_ranges_cycles_per_s": [list(value) for value in ranges],
            "training_distribution": "uniform_episode_level_cpu_regime",
            "checkpoint_selection": contract["checkpoint_selection"],
            "aggregation_unit": "training_seed_mean_over_evaluation_scenarios",
        },
        issues=issues,
    )
    _validate_priority_success_metrics(rows, "CPU robustness", issues)

    validation_seeds = tuple(cfg["seeds"][contract["validation_seed_bank"]])
    specs = cpu_robustness_specs(cfg)
    expected_configs = {
        (spec.algorithm, int(spec.train_seed)): orch.materialize_config(
            spec.to_worker_dict(), validation_seeds
        )
        for spec in specs
    }
    provenance: dict[tuple[str, int], ArtifactProvenance] = {}
    for spec in specs:
        if spec.algorithm == "dbas":
            continue
        try:
            provenance[(spec.algorithm, int(spec.train_seed))] = validate_run_artifact(
                cpu_robustness_run_dir(spec.algorithm, spec.train_seed, cfg),
                expected_config=expected_configs[(spec.algorithm, int(spec.train_seed))],
            )
        except (FileNotFoundError, ValueError) as exc:
            issues.append(f"CPU robustness artifact invalid: {exc}")
    try:
        dbas_config = expected_configs[("dbas", train_seeds[0])]
        for dotted, value in load_cpu_robustness_dbas_calibration(
            cfg, required=True
        ).items():
            node = dbas_config
            parts = dotted.split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        issues.append(f"CPU robustness DBAS calibration invalid: {exc}")
        dbas_config = {}

    def expected_row_provenance(row: dict[str, str]) -> ArtifactProvenance:
        algorithm = row["algorithm"]
        seed = int(row["train_seed"])
        if algorithm != "dbas":
            return provenance[(algorithm, seed)]
        effective = copy.deepcopy(dbas_config)
        effective.setdefault("scenario", {}).update({
            "cpu_cycles_per_s_range": (
                float(row["cpu_range_low_cycles_per_s"]),
                float(row["cpu_range_high_cycles_per_s"]),
            ),
            "cpu_cycles_per_s_regimes": None,
            "cpu_regime_probabilities": None,
        })
        return _stateless_provenance(effective)

    expected_trained_artifacts = (len(ALL_ALGORITHMS) - 1) * len(train_seeds)
    if len(provenance) == expected_trained_artifacts and dbas_config:
        _validate_expected_row_provenance(
            rows, "CPU robustness", expected_row_provenance, issues
        )
    expected = {
        (algorithm, seed, value, eval_seed)
        for algorithm in ALL_ALGORITHMS
        for seed in ((train_seeds[0],) if algorithm == "dbas" else train_seeds)
        for value in values
        for eval_seed in eval_seeds
    }
    try:
        actual = [
            (
                row["algorithm"], int(row["train_seed"]),
                float(row["sweep_value"]), int(row["eval_seed"]),
            )
            for row in rows
        ]
        if set(actual) != expected or len(actual) != len(set(actual)):
            issues.append("CPU robustness Cartesian product is incomplete or duplicated")
        if any(
            row.get("analysis_kind") != "cpu_domain_randomized_policy_evaluation"
            for row in rows
        ):
            issues.append("CPU robustness rows have an invalid evidence label")
        if len(summary) != len(ALL_ALGORITHMS) * len(values):
            issues.append("CPU robustness summary has an invalid cell count")
        expected_seed_rows = (
            (len(ALL_ALGORITHMS) - 1) * len(train_seeds) + 1
        ) * len(values)
        if len(seed_summary) != expected_seed_rows:
            issues.append("CPU robustness seed summary has an invalid cell count")
    except (KeyError, TypeError, ValueError):
        issues.append("CPU robustness output contains invalid identities")
    _validate_seed_reduction(
        summary,
        seed_summary,
        metrics=(
            "social_welfare", "raw_rsu_reward", "task_success_rate",
            "task_success_rate_high", "bandwidth_nonadmitted_ratio",
            "perception_quality_total",
        ),
        label="CPU robustness",
        issues=issues,
    )


def _validate_main_text_analysis(
    cfg: dict[str, Any],
    outputs: Path,
    headline: dict[tuple[str, int], ArtifactProvenance],
    issues: list[str],
) -> None:
    """Validate fixed-policy and counterfactual main-text evidence."""

    analysis = cfg["main_text_analysis"]
    root = outputs / "main_text_analysis"
    eval_bank = str(analysis["evaluation_seed_bank"])
    eval_seeds = tuple(int(value) for value in cfg["seeds"][eval_bank])
    train_seeds = tuple(int(value) for value in cfg["seeds"]["headline_train"])
    common_contract = {
        "revision_id": cfg["revision_id"],
        "revision_schema_version": int(cfg["schema_version"]),
        "local_contract_version": int(analysis["contract_version"]),
        "evaluation_seed_bank": eval_bank,
        "evaluation_seeds": list(eval_seeds),
        "training_seeds": list(train_seeds),
        "aggregation_unit": "training_seed_mean_over_evaluation_scenarios",
    }

    pricing_root = root / "pricing"
    pricing_rows = _rows(pricing_root / "episodes.csv", issues)
    pricing_summary = _rows(pricing_root / "summary.csv", issues)
    pricing_seed_summary = _rows(pricing_root / "seed_summary.csv", issues)
    theta_grid = tuple(float(value) for value in analysis["price_scale_grid"])
    _validate_evaluation_contract(
        pricing_root,
        label="main-text pricing sensitivity",
        expected={
            **common_contract,
            "experiment": "main_text_pricing_sensitivity",
            "algorithm": "proposed",
            "price_scale_grid": list(theta_grid),
            "price_transform": "clip(theta * learned_price, 0, pricing_budget_max)",
            "fixed_quantity": "learned_bandwidth_and_sampled_scenario",
        },
        issues=issues,
    )
    _validate_priority_success_metrics(pricing_rows, "main-text pricing sensitivity", issues)
    _validate_expected_row_provenance(
        pricing_rows,
        "main-text pricing sensitivity",
        lambda row: headline[("proposed", int(row["train_seed"]))],
        issues,
    )
    expected_pricing = {
        (seed, theta, eval_seed)
        for seed in train_seeds for theta in theta_grid for eval_seed in eval_seeds
    }
    try:
        actual_pricing = [
            (int(row["train_seed"]), float(row["sweep_value"]), int(row["eval_seed"]))
            for row in pricing_rows
        ]
        if set(actual_pricing) != expected_pricing or len(actual_pricing) != len(set(actual_pricing)):
            issues.append("main-text pricing Cartesian product is incomplete or duplicated")
        if len(pricing_summary) != len(theta_grid):
            issues.append("main-text pricing summary has an invalid cell count")
        if len(pricing_seed_summary) != len(theta_grid) * len(train_seeds):
            issues.append("main-text pricing seed summary has an invalid cell count")
        for row in pricing_rows:
            cap_fraction = float(row["price_cap_fraction"])
            if not 0.0 <= cap_fraction <= 1.0:
                raise ValueError
            if float(row["sweep_value"]) == 1.0 and not math.isclose(
                float(row["raw_rsu_reward"]),
                float(row["theta_one_reference_raw_rsu_reward"]),
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                issues.append("main-text theta=1 does not reproduce RSU utility")
                break
    except (KeyError, TypeError, ValueError):
        issues.append("main-text pricing output contains invalid values")
    _validate_seed_reduction(
        pricing_summary,
        pricing_seed_summary,
        metrics=("raw_rsu_reward", "task_success_rate", "price_cap_fraction"),
        label="main-text pricing sensitivity",
        issues=issues,
    )

    compression_root = root / "compression"
    compression_rows = _rows(compression_root / "episodes.csv", issues)
    compression_summary = _rows(compression_root / "summary.csv", issues)
    compression_seed_summary = _rows(compression_root / "seed_summary.csv", issues)
    eta_grid = tuple(float(value) for value in analysis["compression_ratio_grid"])
    _validate_evaluation_contract(
        compression_root,
        label="main-text compression energy",
        expected={
            **common_contract,
            "experiment": "main_text_compression_energy",
            "algorithm": "proposed",
            "compression_ratio_grid": list(eta_grid),
            "analysis_semantics": "selected_vehicle_component_counterfactual",
            "fixed_quantities": [
                "sampled_vehicle_and_channel_state",
                "learned_bandwidth",
                "selected_sensing_duration",
            ],
            "deadline_policy": "retain_and_label_infeasible_counterfactual_points",
        },
        issues=issues,
    )
    _validate_expected_row_provenance(
        compression_rows,
        "main-text compression energy",
        lambda row: headline[("proposed", int(row["train_seed"]))],
        issues,
    )
    profiles: dict[str, list[dict[str, str]]] = {}
    try:
        for row in compression_rows:
            profiles.setdefault(row["profile_id"], []).append(row)
            components = (
                float(row["sensing_energy_j"]),
                float(row["computation_energy_j"]),
                float(row["transmission_energy_j"]),
            )
            total = float(row["total_energy_j"])
            if not all(math.isfinite(value) and value >= 0.0 for value in (*components, total)):
                raise ValueError
            if not math.isclose(total, sum(components), rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError
            expected_feasible = int(
                float(row["total_latency_s"]) <= float(row["deadline_s"]) + 1e-9
            )
            if int(row["counterfactual_service_feasible"]) != expected_feasible:
                raise ValueError
        if {int(row["train_seed"]) for row in compression_rows} != set(train_seeds):
            issues.append("main-text compression lacks one or more training seeds")
        for profile_id, profile_rows in profiles.items():
            ordered = sorted(profile_rows, key=lambda row: float(row["sweep_value"]))
            if tuple(float(row["sweep_value"]) for row in ordered) != eta_grid:
                issues.append(f"main-text compression profile {profile_id} has an incomplete grid")
                continue
            computation = [float(row["computation_energy_j"]) for row in ordered]
            transmission = [float(row["transmission_energy_j"]) for row in ordered]
            if any(
                right < left
                for left, right in zip(computation, computation[1:], strict=False)
            ):
                issues.append(f"main-text compression profile {profile_id} has non-monotone computation energy")
            if any(
                right > left
                for left, right in zip(transmission, transmission[1:], strict=False)
            ):
                issues.append(f"main-text compression profile {profile_id} has non-monotone transmission energy")
        if len(compression_summary) != len(eta_grid):
            issues.append("main-text compression summary has an invalid cell count")
        if len(compression_seed_summary) != len(eta_grid) * len(train_seeds):
            issues.append("main-text compression seed summary has an invalid cell count")
    except (KeyError, TypeError, ValueError):
        issues.append("main-text compression output contains invalid values")
    _validate_seed_reduction(
        compression_summary,
        compression_seed_summary,
        metrics=(
            "sensing_energy_j",
            "computation_energy_j",
            "transmission_energy_j",
            "total_energy_j",
            "counterfactual_service_feasible",
        ),
        label="main-text compression energy",
        issues=issues,
    )


def _scalability_expectations(
    axis: str,
    cfg: dict[str, Any],
    headline: dict[tuple[str, int], ArtifactProvenance],
    cache: ProvenanceCache,
) -> dict[tuple[str, int, float], ArtifactProvenance]:
    artifacts, _ = revision_roots(cfg)
    root = artifacts / "scalability" / axis
    default_value = 20.0 if axis == "vehicles" else 3.0
    expectations: dict[tuple[str, int, float], ArtifactProvenance] = {}
    for algorithm in cfg["scalability"]["algorithms"]:
        seeds = train_seeds_for(algorithm, scope="sweep", config=cfg)
        if algorithm == "dbas":
            seeds = seeds[:1]
        for seed in seeds:
            expectations[(algorithm, int(seed), default_value)] = headline[(algorithm, int(seed))]

    validation_seeds = tuple(cfg["seeds"]["validation"])
    for spec in scalability_specs(axis, cfg):
        key = (spec.algorithm, int(spec.train_seed), float(spec.display_value))
        if spec.algorithm == "dbas":
            config = copy.deepcopy(formal_algorithm_config("dbas", cfg))
            config.setdefault("scenario", {}).update(spec.scenario_overrides)
            expectations[key] = _stateless_provenance(config)
        else:
            config = orch.materialize_config(spec.to_worker_dict(), validation_seeds)
            expectations[key] = _trained_provenance(
                spec.algorithm, spec.run_dir(root), config, cache
            )
    return expectations


def _timing_expectations(
    cfg: dict[str, Any],
    headline: dict[tuple[str, int], ArtifactProvenance],
    cache: ProvenanceCache,
) -> dict[tuple[int, int], ArtifactProvenance]:
    artifacts, _ = revision_roots(cfg)
    seed = int(cfg["seeds"]["sweep_train"][0])
    expectations = {(20, 3): headline[("proposed", seed)]}
    validation_seeds = tuple(cfg["seeds"]["validation"])
    root = artifacts / "scalability" / "timing"
    for spec in timing_scalability_specs(cfg):
        config = orch.materialize_config(spec.to_worker_dict(), validation_seeds)
        scenario = spec.scenario_overrides
        key = (int(scenario["num_vehicles"]), len(scenario["urgency_profiles"]))
        expectations[key] = _trained_provenance("proposed", spec.run_dir(root), config, cache)
    return expectations


def _rows(path: Path, issues: list[str]) -> list[dict[str, str]]:
    if not path.exists():
        issues.append(f"missing file: {path}")
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        issues.append(f"empty CSV: {path}")
    return rows


def _validate_row_provenance_fields(
    rows: list[dict[str, str]], label: str, issues: list[str]
) -> None:
    for row in rows:
        if not row.get("config_fingerprint") or not row.get("checkpoint_sha256"):
            issues.append(f"{label}: row lacks config/checkpoint provenance")
            return
        if row.get("artifact_kind") not in ("trained_checkpoint", "stateless_config"):
            issues.append(f"{label}: invalid artifact_kind")
            return


def _validate_expected_row_provenance(
    rows: list[dict[str, str]],
    label: str,
    resolver: Any,
    issues: list[str],
) -> None:
    _validate_row_provenance_fields(rows, label, issues)
    mismatch_groups: dict[
        tuple[str, str, str, tuple[str, ...]], int
    ] = {}
    resolution_groups: dict[tuple[str, str, str], int] = {}
    for row in rows:
        try:
            expected = resolver(row)
            mismatches = _provenance_mismatch_fields(row, expected)
            if not mismatches:
                continue
            key = (
                str(row.get("algorithm", "unknown")),
                str(row.get("train_seed", "unknown")),
                str(expected.config_fingerprint),
                mismatches,
            )
            mismatch_groups[key] = mismatch_groups.get(key, 0) + 1
        except (KeyError, OSError, TypeError, ValueError) as exc:
            key = (
                str(row.get("algorithm", "unknown")),
                str(row.get("train_seed", "unknown")),
                str(exc),
            )
            resolution_groups[key] = resolution_groups.get(key, 0) + 1

    for (algorithm, train_seed, fingerprint, fields), count in sorted(
        mismatch_groups.items()
    ):
        field_list = ", ".join(fields)
        issues.append(
            f"{label}: algorithm={algorithm}, train_seed={train_seed}, "
            f"expected_config={fingerprint[:12]} has stale provenance fields "
            f"[{field_list}] across {count} row(s)"
        )
    for (algorithm, train_seed, error), count in sorted(
        resolution_groups.items()
    ):
        issues.append(
            f"{label}: algorithm={algorithm}, train_seed={train_seed} cannot "
            f"resolve current provenance across {count} row(s): {error}"
        )


def _validate_priority_success_metrics(
    rows: list[dict[str, str]], label: str, issues: list[str]
) -> None:
    """Validate the paper-defined per-urgency completion decomposition.

    A task succeeds iff its effective quality is positive. Formal experiments
    contain at least one task at every configured urgency level, so each row
    must expose finite counts and rates satisfying both the per-level and
    aggregate identities.
    """

    for row_index, row in enumerate(rows):
        try:
            total_count = 0.0
            total_failures = 0.0
            for urgency in ("high", "mid", "low"):
                count = float(row[f"task_count_{urgency}"])
                failures = float(row[f"task_failure_count_{urgency}"])
                success_rate = float(row[f"task_success_rate_{urgency}"])
                if not all(math.isfinite(value) for value in (count, failures, success_rate)):
                    raise ValueError("non-finite priority completion metric")
                if count <= 0.0 or failures < 0.0 or failures > count:
                    raise ValueError("invalid priority task counts")
                expected_rate = 1.0 - failures / count
                if not 0.0 <= success_rate <= 1.0 or not math.isclose(
                    success_rate, expected_rate, rel_tol=1e-10, abs_tol=1e-12
                ):
                    raise ValueError("inconsistent priority success rate")
                total_count += count
                total_failures += failures

            aggregate_failures = float(row["task_failure_count"])
            aggregate_success_rate = float(row["task_success_rate"])
            if not math.isclose(aggregate_failures, total_failures, rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError("priority failures do not sum to aggregate")
            expected_aggregate_rate = 1.0 - total_failures / total_count
            if not math.isclose(
                aggregate_success_rate,
                expected_aggregate_rate,
                rel_tol=1e-10,
                abs_tol=1e-12,
            ):
                raise ValueError("priority rates do not match aggregate")
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(f"{label}: invalid priority success metrics at row {row_index}: {exc}")
            return


def _validate_pruning_equivalent_cells(rows: list[dict[str, str]], issues: list[str]) -> None:
    """Require rho-1/rho-2 aliases to contain identical measured results.

    Under the maintained task geometry, both settings induce candidate cap two.
    The reporting pipeline evaluates that physical mechanism once and aliases
    the result. Checking every emitted sweep metric prevents metadata-only
    equivalence claims from masking separate or stale evaluations.

    Exactness label: implementation convenience only (evidence validation).
    """

    by_rho: dict[float, dict[tuple[float, int, int], dict[str, str]]] = {
        1.0: {},
        2.0: {},
    }
    try:
        for row in rows:
            if row.get("algorithm") != "proposed":
                continue
            rho = float(row["diversity_factor"])
            if rho not in by_rho:
                continue
            key = (
                float(row["r0_m"]),
                int(row["train_seed"]),
                int(row["eval_seed"]),
            )
            by_rho[rho][key] = row
    except (KeyError, TypeError, ValueError):
        issues.append("pruning sensitivity contains invalid rho-equivalence identities")
        return

    if set(by_rho[1.0]) != set(by_rho[2.0]):
        issues.append("pruning sensitivity rho=1/rho=2 equivalence pairs are incomplete")
        return

    for key in sorted(by_rho[1.0]):
        left = by_rho[1.0][key]
        right = by_rho[2.0][key]
        for metric in SWEEP_METRIC_NAMES:
            try:
                left_value = float(left[metric])
                right_value = float(right[metric])
            except (KeyError, TypeError, ValueError):
                issues.append(f"pruning sensitivity rho-equivalence metric missing: {metric}")
                return
            equal = math.isnan(left_value) and math.isnan(right_value) or left_value == right_value
            if not equal:
                issues.append(
                    f"pruning sensitivity rho=1/rho=2 values differ for {key}, metric={metric}"
                )
                return


def _validate_hardware_payload(payload: dict[str, Any], label: str, issues: list[str]) -> None:
    """Require interpretable CPU/RAM/full-host GPU benchmark metadata."""

    if payload.get("hardware_schema_version") != 1:
        issues.append(f"{label}: unsupported hardware metadata schema")
    if not str(payload.get("hostname") or "").strip():
        issues.append(f"{label}: missing hostname")

    cpu_model = str(payload.get("cpu_model") or "").strip().lower()
    if not cpu_model or cpu_model == "unknown":
        issues.append(f"{label}: missing CPU model")
    for field in ("physical_cpu_count", "logical_cpu_count"):
        try:
            if int(payload[field]) <= 0:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            issues.append(f"{label}: invalid {field}")
    try:
        if float(payload["memory_total_gib"]) <= 0:
            raise ValueError
    except (KeyError, TypeError, ValueError):
        issues.append(f"{label}: invalid memory_total_gib")
    try:
        physical_count = int(payload["physical_gpu_count"])
        physical_gpus = list(payload["physical_gpus"])
        if physical_count <= 0 or len(physical_gpus) != physical_count:
            raise ValueError
        for gpu in physical_gpus:
            if not str(gpu.get("name") or "").strip():
                raise ValueError
            if float(gpu.get("memory_total_gib") or 0.0) <= 0:
                raise ValueError
            if not str(gpu.get("driver") or "").strip():
                raise ValueError
    except (KeyError, TypeError, ValueError):
        issues.append(f"{label}: incomplete physical GPU inventory")
    for field in ("python", "torch", "git_commit"):
        value = str(payload.get(field) or "").strip().lower()
        if not value or value == "unknown":
            issues.append(f"{label}: missing {field}")


def _validate_hardware_file(path: Path, label: str, issues: list[str]) -> None:
    if not path.exists():
        issues.append(f"missing {label}: {path}")
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        issues.append(f"{label}: invalid JSON")
        return
    _validate_hardware_payload(payload, label, issues)


def _validate_convergence(
    outputs: Path,
    cfg: dict,
    issues: list[str],
    cache: ProvenanceCache,
) -> None:
    algorithms = ("proposed", "jdls", "mappo", "ebas", "spis", "rdus")
    sources = _rows(outputs / "tables" / "convergence_sources.csv", issues)
    expected_source_ids = {
        (algorithm, int(seed))
        for algorithm in algorithms
        for seed in train_seeds_for(algorithm, config=cfg)
    }
    try:
        actual_source_ids = [(row["algorithm"], int(row["train_seed"])) for row in sources]
        if set(actual_source_ids) != expected_source_ids or len(actual_source_ids) != len(
            set(actual_source_ids)
        ):
            issues.append("convergence source identities are incomplete or duplicated")
    except (KeyError, TypeError, ValueError):
        issues.append("convergence sources contain invalid identities")

    try:
        headline = _headline_expectations(cfg, cache)
    except (FileNotFoundError, ValueError) as exc:
        issues.append(f"convergence: cannot build current provenance: {exc}")
        headline = {}

    def convergence_source_provenance(row: dict[str, str]) -> ArtifactProvenance:
        return headline[(row["algorithm"], int(row["train_seed"]))]

    _validate_expected_row_provenance(
        sources, "convergence sources", convergence_source_provenance, issues
    )
    for row in sources:
        try:
            curve_path = (
                headline_run_dir(row["algorithm"], int(row["train_seed"]), cfg)
                / "eval"
                / "evaluations.npz"
            )
            if row.get("evaluations_sha256") != sha256_file(curve_path):
                issues.append(
                    "convergence sources: evaluations_sha256 does not match current artifact"
                )
        except (KeyError, OSError, TypeError, ValueError) as exc:
            issues.append(f"convergence sources: cannot hash evaluation curve: {exc}")

    bundles = {
        algorithm: provenance_bundle_digest(
            [row for row in sources if row.get("algorithm") == algorithm]
        )
        for algorithm in algorithms
    }
    rows = _rows(outputs / "tables" / "convergence.csv", issues)
    by_algorithm = {row.get("algorithm"): row for row in rows}
    if set(by_algorithm) != set(algorithms) or len(rows) != len(algorithms):
        issues.append("convergence report algorithm identities are incomplete or duplicated")
    for algorithm in algorithms:
        row = by_algorithm.get(algorithm)
        if row is None:
            continue
        if row.get("evidence_status") != "complete_paper":
            issues.append(f"convergence {algorithm}: not complete paper evidence")
        if row.get("source_bundle_sha256") != bundles[algorithm]:
            issues.append(f"convergence {algorithm}: stale source bundle")
        try:
            if int(row["seeds"]) != len(train_seeds_for(algorithm, config=cfg)):
                issues.append(f"convergence {algorithm}: wrong train-seed count")
            isolated_count = int(row.get("isolated_wall_time_seed_count") or 0)
            expected_isolated = 1 if algorithm in ("proposed", "jdls", "mappo") else 0
            if isolated_count != expected_isolated:
                issues.append(
                    f"convergence {algorithm}: isolated timing seeds "
                    f"{isolated_count} != {expected_isolated}"
                )
        except (KeyError, TypeError, ValueError):
            issues.append(f"convergence {algorithm}: invalid count fields")

    budgets = _rows(outputs / "tables" / "equal_budget.csv", issues)
    expected_budget_ids = {
        (algorithm, budget) for algorithm in algorithms for budget in (20_000, 40_000, 60_000)
    }
    try:
        actual = [(row["algorithm"], int(row["budget_steps"])) for row in budgets]
        if set(actual) != expected_budget_ids or len(actual) != len(set(actual)):
            issues.append("equal-budget report identities are incomplete or duplicated")
        if any(row.get("evidence_status") != "complete_paper" for row in budgets):
            issues.append("equal-budget report contains non-paper partial evidence")
        for row in budgets:
            if row.get("source_bundle_sha256") != bundles[row["algorithm"]]:
                issues.append(f"equal-budget {row['algorithm']}: stale source bundle")
    except (KeyError, TypeError, ValueError):
        issues.append("equal-budget report contains invalid identities")

    reward_cfg = cfg["common_evaluation_reward"]
    reward_algorithms = tuple(reward_cfg["plotted_algorithms"])
    reward_figure_algorithms = tuple(reward_cfg["figure_algorithms"])
    reward_schema = str(reward_cfg["schema"])
    reward_summary = _rows(
        outputs / "tables" / "reward_convergence.csv", issues
    )
    reward_by_algorithm = {
        row.get("algorithm"): row for row in reward_summary
    }
    if (
        set(reward_by_algorithm) != set(reward_algorithms)
        or len(reward_summary) != len(reward_algorithms)
    ):
        issues.append("reward convergence algorithm identities are incomplete")
    for algorithm in reward_algorithms:
        row = reward_by_algorithm.get(algorithm)
        if row is None:
            continue
        try:
            if row.get("reward_schema") != reward_schema:
                raise ValueError
            if row.get("evidence_status") != "complete_paper":
                raise ValueError
            if int(row["seeds"]) != len(train_seeds_for(algorithm, config=cfg)):
                raise ValueError
            if row.get("source_bundle_sha256") != bundles[algorithm]:
                raise ValueError
            for field in (
                "final_reward_mean", "final_reward_std", "final_reward_ci95"
            ):
                if not math.isfinite(float(row[field])):
                    raise ValueError
        except (KeyError, TypeError, ValueError):
            issues.append(f"reward convergence {algorithm}: invalid summary")

    reward_source = _rows(
        outputs / "figures" / "figR_reward_convergence_source.csv", issues
    )
    source_algorithms = {row.get("algorithm") for row in reward_source}
    if source_algorithms != set(reward_figure_algorithms):
        issues.append("reward convergence figure source algorithms are incomplete")
    for row in reward_source:
        algorithm = row.get("algorithm")
        try:
            if algorithm not in reward_figure_algorithms:
                raise ValueError
            if row.get("reward_schema") != reward_schema:
                raise ValueError
            if row.get("uncertainty") != "seed_std":
                raise ValueError
            if int(row["train_seed_count"]) != len(
                train_seeds_for(str(algorithm), config=cfg)
            ):
                raise ValueError
            if row.get("source_bundle_sha256") != bundles[algorithm]:
                raise ValueError
            if int(row["timestep"]) <= 0:
                raise ValueError
            if not math.isfinite(float(row["common_evaluation_reward_mean"])):
                raise ValueError
            if not math.isfinite(
                float(row["common_evaluation_reward_seed_std"])
            ):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            issues.append("reward convergence figure source contains invalid row")


def _validate_fairness_settings(outputs: Path, cfg: dict, issues: list[str]) -> None:
    rows = _rows(outputs / "tables" / "fairness_settings.csv", issues)
    identities = [row.get("algorithm") for row in rows]
    if set(identities) != set(ALL_ALGORITHMS) or len(identities) != len(set(identities)):
        issues.append("fairness settings algorithm identities are incomplete or duplicated")
        return
    expected_hidden = list(map(int, cfg["fairness"]["hidden_sizes"]))
    for row in rows:
        algorithm = str(row["algorithm"])
        if row.get("protocol") != cfg["fairness"]["protocol"]:
            issues.append(f"fairness settings {algorithm}: wrong protocol")
        if algorithm not in STOCHASTIC_ALGORITHMS:
            continue
        try:
            if ast.literal_eval(row["network_hidden_sizes"]) != expected_hidden:
                issues.append(f"fairness settings {algorithm}: wrong hidden sizes")
            if float(row["learning_rate"]) != float(cfg["fairness"]["learning_rate"]):
                issues.append(f"fairness settings {algorithm}: wrong learning rate")
            if float(row["clip_range"]) != float(cfg["fairness"]["clip_range"]):
                issues.append(f"fairness settings {algorithm}: wrong clip range")
            if int(row["total_timesteps"]) != int(cfg["fairness"]["total_timesteps"]):
                issues.append(f"fairness settings {algorithm}: wrong training budget")
            if int(row["headline_train_seeds"]) != len(cfg["seeds"]["headline_train"]):
                issues.append(f"fairness settings {algorithm}: wrong seed count")
        except (KeyError, SyntaxError, TypeError, ValueError):
            issues.append(f"fairness settings {algorithm}: invalid values")


def validate_headline(
    path: Path,
    cfg: dict,
    issues: list[str],
    expectations: dict[tuple[str, int], ArtifactProvenance] | None = None,
) -> None:
    rows = _rows(path, issues)
    _validate_priority_success_metrics(rows, "headline", issues)
    if expectations is None:
        try:
            expectations = _headline_expectations(cfg, {})
        except (FileNotFoundError, ValueError) as exc:
            issues.append(f"headline: cannot build current provenance: {exc}")
            expectations = {}
    _validate_expected_row_provenance(
        rows,
        "headline",
        lambda row: expectations[(row["algorithm"], int(row["train_seed"]))],
        issues,
    )
    for algorithm in ALL_ALGORITHMS:
        subset = [row for row in rows if row.get("algorithm") == algorithm]
        expected_seeds = set(map(str, train_seeds_for(algorithm, config=cfg)))
        actual_seeds = {row.get("train_seed", "") for row in subset}
        if actual_seeds != expected_seeds:
            issues.append(
                f"headline {algorithm}: seeds {sorted(actual_seeds)} != {sorted(expected_seeds)}"
            )
        expected_episodes = len(expected_seeds) * len(cfg["seeds"]["test"])
        if len(subset) != expected_episodes:
            issues.append(f"headline {algorithm}: {len(subset)} rows != {expected_episodes}")
        expected_pairs = {
            (int(train_seed), int(eval_seed))
            for train_seed in expected_seeds
            for eval_seed in cfg["seeds"]["test"]
        }
        actual_pairs: list[tuple[int, int]] = []
        try:
            actual_pairs = [(int(row["train_seed"]), int(row["eval_seed"])) for row in subset]
        except (KeyError, TypeError, ValueError):
            issues.append(f"headline {algorithm}: invalid train_seed/eval_seed identity")
        if set(actual_pairs) != expected_pairs:
            issues.append(f"headline {algorithm}: seed Cartesian product is incomplete")
        if len(actual_pairs) != len(set(actual_pairs)):
            issues.append(f"headline {algorithm}: duplicate train/eval seed rows")
        for row in subset:
            for metric in ("social_welfare", "raw_rsu_reward", "perception_quality_total"):
                try:
                    if not math.isfinite(float(row[metric])):
                        issues.append(f"headline {algorithm}: non-finite {metric}")
                        break
                except (KeyError, TypeError, ValueError):
                    issues.append(f"headline {algorithm}: invalid {metric}")
                    break
            for left, right_a, right_b in (
                ("bandwidth_total", "bandwidth_admitted_sum", "bandwidth_nonadmitted_sum"),
                ("bandwidth_total", "bandwidth_selected_sum", "bandwidth_idle_sum"),
            ):
                try:
                    if not math.isclose(
                        float(row[left]),
                        float(row[right_a]) + float(row[right_b]),
                        rel_tol=1e-10,
                        abs_tol=1e-12,
                    ):
                        issues.append(f"headline {algorithm}: inconsistent bandwidth diagnostics")
                except (KeyError, TypeError, ValueError):
                    issues.append(f"headline {algorithm}: missing bandwidth diagnostics")
            if algorithm == "proposed":
                try:
                    leakage = float(row["bandwidth_nonadmitted_ratio"])
                    if not math.isfinite(leakage) or abs(leakage) > 1e-12:
                        issues.append("headline proposed: non-candidate bandwidth leakage")
                except (KeyError, TypeError, ValueError):
                    issues.append("headline proposed: missing candidate-bandwidth diagnostic")


def _report_seed_mean_map(
    rows: list[dict[str, str]], algorithm: str, metric: str
) -> dict[str, float]:
    by_seed: dict[str, list[float]] = {}
    for row in rows:
        if row.get("algorithm") != algorithm:
            continue
        try:
            value = float(row[metric])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            by_seed.setdefault(str(row["train_seed"]), []).append(value)
    return {seed: sum(values) / len(values) for seed, values in by_seed.items() if values}


def _validate_significance_report(outputs: Path, issues: list[str]) -> None:
    """Validate that paper-facing significance uses matched seed pairs."""

    from scripts.report_summary_stats import paired_signflip_p

    headline_path = outputs / "headline" / "episodes.csv"
    significance_path = outputs / "tables" / "significance.csv"
    headline = _rows(headline_path, issues)
    significance = _rows(significance_path, issues)
    if not headline or not significance:
        return

    expected_identities = {
        (metric, algorithm)
        for metric in REPORT_METRICS
        for algorithm in ALL_ALGORITHMS
        if algorithm != "proposed"
    }
    actual_identities = [(row.get("metric", ""), row.get("baseline", "")) for row in significance]
    if set(actual_identities) != expected_identities or len(actual_identities) != len(
        set(actual_identities)
    ):
        issues.append("significance report identities are incomplete or duplicated")

    source_digest = sha256_file(headline_path)
    for row in significance:
        metric = str(row.get("metric", ""))
        baseline = str(row.get("baseline", ""))
        label = f"significance {metric}/{baseline}"
        if (metric, baseline) not in expected_identities:
            continue
        proposed = _report_seed_mean_map(headline, "proposed", metric)
        comparison = _report_seed_mean_map(headline, baseline, metric)
        expected_p, expected_pairs = paired_signflip_p(proposed, comparison)
        expected_complete = len(proposed) >= 2 and set(proposed) == set(comparison)
        try:
            if row["source_csv_sha256"] != source_digest:
                issues.append(f"{label}: stale source digest")
            if int(row["paired_seed_count"]) != expected_pairs:
                issues.append(f"{label}: wrong paired seed count")
            if int(row["primary_pair_count"]) != expected_pairs:
                issues.append(f"{label}: wrong primary pair count")
            recorded_complete = row["paired_seed_complete"].lower() == "true"
            if recorded_complete != expected_complete:
                issues.append(f"{label}: wrong pairing-completeness status")
            recorded_paired = float(row["paired_signflip_p"])
            if math.isfinite(expected_p):
                if not math.isclose(recorded_paired, expected_p, rel_tol=1e-12, abs_tol=1e-12):
                    issues.append(f"{label}: wrong paired sign-flip p-value")
            elif math.isfinite(recorded_paired):
                issues.append(f"{label}: paired p-value should be unavailable")

            if expected_complete:
                if row["primary_test"] != "exact_two_sided_paired_sign_flip":
                    issues.append(f"{label}: wrong primary test")
                if row["primary_test_status"] != "complete_paper":
                    issues.append(f"{label}: primary test is not paper-complete")
                if not math.isclose(
                    float(row["primary_p_value"]),
                    expected_p,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    issues.append(f"{label}: wrong primary p-value")
            elif (
                row["primary_test"]
                or row["primary_p_value"]
                or row["primary_test_status"] != "diagnostic_only_non_final_insufficient_pairs"
            ):
                issues.append(f"{label}: insufficient pairing was presented as primary")
        except (KeyError, TypeError, ValueError):
            issues.append(f"{label}: invalid paired-significance metadata")


def _has_invalid_dbas_uncertainty(rows: list[dict[str, str]]) -> bool:
    """Return whether DBAS rows are missing or claim train-seed uncertainty."""

    dbas_rows = [row for row in rows if row.get("algorithm") == "dbas"]
    return not dbas_rows or any(
        row.get("uncertainty") != "none_stateless"
        or row.get("seed_ci95", "").strip().lower() not in {"", "nan"}
        for row in dbas_rows
    )


def _validate_manuscript_figures(
    outputs: Path, cfg: dict[str, Any], issues: list[str]
) -> None:
    """Validate the report-only, manuscript-compatible Fig. 1--9 bundle."""

    contract = cfg.get("manuscript_figures", {})
    root = outputs / str(contract.get("output_subdir", "figures/manuscript"))
    primary = tuple(f"fig{index}" for index in range(1, 10))
    full = ("fig2_full", "fig3_full", "fig4_full", "fig5_full", "fig6_full", "fig9_full")
    required = primary + full
    manifest = _rows(root / "manifest.csv", issues)
    by_figure = {row.get("figure", ""): row for row in manifest}
    if set(by_figure) != set(required) or len(manifest) != len(required):
        issues.append("manuscript figure manifest is incomplete or duplicated")
        return

    expected_inputs = {
        "fig1": "figures/figR_learning_efficiency_source.csv",
        "fig2": "scalability/vehicles/summary.csv",
        "fig2_full": "scalability/vehicles/summary.csv",
        "fig3": "dynamic_robustness/summary.csv",
        "fig3_full": "dynamic_robustness/summary.csv",
        "fig4": "cpu_robustness/formal/summary.csv",
        "fig4_full": "cpu_robustness/formal/summary.csv",
        "fig5": "scalability/vehicles/summary.csv",
        "fig5_full": "scalability/vehicles/summary.csv",
        "fig6": "dynamic_robustness/summary.csv",
        "fig6_full": "dynamic_robustness/summary.csv",
        "fig7": "main_text_analysis/compression/summary.csv",
        "fig8": "main_text_analysis/pricing/summary.csv",
        "fig9": "scalability/vehicles/summary.csv",
        "fig9_full": "scalability/vehicles/summary.csv",
    }
    paper_algorithms = set(contract.get("paper_algorithms", ()))
    full_algorithms = set(contract.get("full_algorithms", ()))
    convergence_algorithms = set(contract.get("convergence_algorithms", ()))
    vehicle_grid = {float(value) for value in cfg["scalability"]["vehicle_grid"]}
    bandwidth_grid_mhz = {
        float(value) / 1e6 for value in cfg["scalability"]["rsu_bandwidth_grid_hz"]
    }
    cpu_grid = {
        (float(low) + float(high)) / 2e9
        for low, high in cfg["main_text_analysis"]["cpu_capacity_ranges_cycles_per_s"]
    }
    compression_grid = {
        float(value) for value in cfg["main_text_analysis"]["compression_ratio_grid"]
    }
    pricing_grid = {
        float(value) for value in cfg["main_text_analysis"]["price_scale_grid"]
    }

    for stem in required:
        record = by_figure[stem]
        try:
            if int(record["contract_version"]) != int(contract["contract_version"]):
                raise ValueError("contract version mismatch")
            if record["exactness_label"] != "implementation convenience only":
                raise ValueError("invalid exactness label")
            expected_size = FIGSIZE_SINGLE_COLUMN
            if record.get("layout") != "single_column":
                raise ValueError("wrong figure layout")
            if record.get("legend_layout") != "single_row_adaptive":
                raise ValueError("wrong figure legend layout")
            recorded_size = (
                float(record["figure_width_in"]),
                float(record["figure_height_in"]),
            )
            if recorded_size != expected_size:
                raise ValueError("wrong figure canvas size")
            input_name = record["input_csv"]
            source_name = record["source_csv"]
            if input_name != expected_inputs[stem]:
                raise ValueError("unexpected formal input")
            if Path(input_name).is_absolute() or ".." in Path(input_name).parts:
                raise ValueError("unsafe input path")
            if Path(source_name).is_absolute() or ".." in Path(source_name).parts:
                raise ValueError("unsafe source path")
            input_path = outputs / input_name
            source_path = outputs / source_name
            if sha256_file(input_path) != record["input_csv_sha256"]:
                raise ValueError("input digest mismatch")
            if sha256_file(source_path) != record["source_csv_sha256"]:
                raise ValueError("plotted-source digest mismatch")
            for suffix, digest_field in (
                (".png", "png_sha256"),
                (".pdf", "pdf_sha256"),
            ):
                figure_path = root / f"{stem}{suffix}"
                if sha256_file(figure_path) != record[digest_field]:
                    raise ValueError(f"{suffix[1:]} figure digest mismatch")
            expected_pixels = tuple(round(value * FIGURE_DPI) for value in expected_size)
            if _png_dimensions(root / f"{stem}.png") != expected_pixels:
                raise ValueError("wrong PNG dimensions")
            rows = _rows(source_path, issues)
            if not rows:
                raise ValueError("empty plotted source")
            if stem == "fig1":
                if {row["algorithm"] for row in rows} != convergence_algorithms:
                    raise ValueError("wrong convergence algorithm set")
                counts = {
                    algorithm: len([row for row in rows if row["algorithm"] == algorithm])
                    for algorithm in convergence_algorithms
                }
                if min(counts.values()) < 2 or len(set(counts.values())) != 1:
                    raise ValueError("unaligned convergence trajectories")
                if {row["uncertainty"] for row in rows} != {"seed_std"}:
                    raise ValueError("wrong convergence uncertainty")
            elif stem == "fig7":
                if {float(row["compression_ratio"]) for row in rows} != compression_grid:
                    raise ValueError("wrong compression grid")
                if {row["series"] for row in rows} != {
                    "Sensing", "Computation", "Transmission", "Total"
                }:
                    raise ValueError("wrong energy components")
            elif stem == "fig8":
                if {float(row["price_scale"]) for row in rows} != pricing_grid:
                    raise ValueError("wrong pricing grid")
                if {row["algorithm"] for row in rows} != {"proposed"}:
                    raise ValueError("wrong pricing algorithm")
            else:
                expected_algorithms = (
                    full_algorithms if stem.endswith("_full") else paper_algorithms
                )
                if {row["algorithm"] for row in rows} != expected_algorithms:
                    raise ValueError("wrong comparison algorithm set")
                if _has_invalid_dbas_uncertainty(rows):
                    raise ValueError("DBAS must not carry a train-seed confidence interval")
                actual_grid = {float(row["sweep_value"]) for row in rows}
                if stem.startswith(("fig2", "fig5", "fig9")):
                    expected_grid = vehicle_grid
                elif stem.startswith(("fig3", "fig6")):
                    scale = 1e6 if max(actual_grid) > 1e5 else 1.0
                    actual_grid = {value / scale for value in actual_grid}
                    expected_grid = bandwidth_grid_mhz
                else:
                    expected_grid = cpu_grid
                if actual_grid != expected_grid:
                    raise ValueError("wrong measured grid")
        except (KeyError, OSError, TypeError, ValueError) as exc:
            issues.append(f"{stem}: invalid manuscript figure evidence: {exc}")


def _validate_evaluation_contract(
    output_dir: Path,
    *,
    label: str,
    expected: dict[str, Any],
    issues: list[str],
) -> None:
    """Validate the current evaluation contract sidecar."""

    sidecar = output_dir / CONTRACT_FILENAME
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        recorded_digest = str(payload.pop("contract_digest"))
        if payload.get("schema_version") != 1:
            raise ValueError
        if payload.get("evidence_status") != FORMAL_STATUS:
            raise ValueError
        if config_digest(payload) != recorded_digest:
            raise ValueError
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        issues.append(f"{label} has an invalid or stale evaluation contract")


def validate_final(cfg: dict) -> list[str]:
    provenance_cache: ProvenanceCache = {}
    artifacts, outputs = revision_roots(cfg)
    issues: list[str] = []
    dbas_selection = Path(cfg["baseline_calibration"]["dbas_selection_file"])
    if not dbas_selection.exists():
        issues.append(f"missing DBAS price calibration: {dbas_selection}")
    else:
        try:
            load_dbas_calibration(cfg, required=True)
        except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
            issues.append(f"invalid DBAS price calibration: {exc}")
    try:
        headline_provenance = _headline_expectations(cfg, provenance_cache)
    except (FileNotFoundError, ValueError) as exc:
        issues.append(f"headline: cannot build current provenance: {exc}")
        headline_provenance = {}
    validate_headline(
        outputs / "headline" / "episodes.csv",
        cfg,
        issues,
        headline_provenance,
    )
    _validate_significance_report(outputs, issues)
    _validate_cpu_robustness(cfg, outputs, issues)
    if headline_provenance:
        _validate_main_text_analysis(
            cfg, outputs, headline_provenance, issues
        )

    ablation = _rows(outputs / "ablation" / "summary.csv", issues)
    if {row.get("ablation_arm") for row in ablation} != set(ABLATION_ARMS):
        issues.append("ablation summary does not contain exactly five controlled arms")
    ablation_episodes = _rows(outputs / "ablation" / "episodes.csv", issues)
    _validate_priority_success_metrics(ablation_episodes, "ablation", issues)
    if ablation and any(
        field in ablation[0]
        for field in ("inference_wall_time_ms_mean", "follower_solve_time_ms_mean")
    ):
        issues.append("formal ablation summary must exclude unwarmed timing metrics")
    if any(
        row.get("runtime_evidence_status") != "diagnostic_unwarmed_non_final"
        for row in ablation_episodes
    ):
        issues.append("ablation episode timing must be labelled diagnostic and unwarmed")
    try:
        ablation_provenance = _ablation_expectations(cfg, provenance_cache)
    except (FileNotFoundError, ValueError) as exc:
        issues.append(f"ablation: cannot build current provenance: {exc}")
        ablation_provenance = {}
    _validate_expected_row_provenance(
        ablation_episodes,
        "ablation",
        lambda row: ablation_provenance[(row["ablation_arm"], int(row["train_seed"]))],
        issues,
    )
    expected_ablation = {
        (arm, int(train_seed), int(eval_seed))
        for arm in ABLATION_ARMS
        for train_seed in cfg["seeds"]["ablation_train"]
        for eval_seed in cfg["seeds"]["test"]
    }
    try:
        actual_ablation = [
            (row["ablation_arm"], int(row["train_seed"]), int(row["eval_seed"]))
            for row in ablation_episodes
        ]
        if set(actual_ablation) != expected_ablation:
            issues.append("ablation episode Cartesian product is incomplete")
        if len(actual_ablation) != len(set(actual_ablation)):
            issues.append("ablation episodes contain duplicate identities")
    except (KeyError, TypeError, ValueError):
        issues.append("ablation episodes contain invalid identities")
    ablation_train_seeds = tuple(
        int(seed) for seed in cfg["seeds"]["ablation_train"]
    )
    ablation_eval_seeds = tuple(int(seed) for seed in cfg["seeds"]["test"])
    for row in ablation:
        arm = row.get("ablation_arm", "<unknown>")
        for metric in ABLATION_UNCONDITIONAL_METRICS:
            prefix = f"{metric}_change_vs_full_pct"
            try:
                pairs = int(row[f"{prefix}_pairs"])
                expected_pairs = len(cfg["seeds"]["ablation_train"])
                if pairs != expected_pairs:
                    issues.append(f"ablation {arm}/{metric}: {pairs} pairs != {expected_pairs}")
                for field in (prefix, f"{prefix}_std", f"{prefix}_ci95"):
                    if not math.isfinite(float(row[field])):
                        raise ValueError
                p_value = float(row[f"{prefix}_paired_permutation_p"])
                if not math.isfinite(p_value) or not 0.0 <= p_value <= 1.0:
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                issues.append(f"ablation {arm}/{metric}: invalid paired statistics")

        if arm not in ABLATION_ARMS:
            continue
        metric = ABLATION_CONDITIONAL_RATIO_METRIC
        try:
            expected_ratio = conditional_ratio_summary(
                ablation_episodes,
                arm=arm,
                numerator="energy_total_sum",
                denominator="selected_vehicle_count",
                train_seeds=ablation_train_seeds,
                eval_seeds=ablation_eval_seeds,
            )
            expected_delta = paired_ablation_ratio_delta_summary(
                ablation_episodes,
                arm=arm,
                numerator="energy_total_sum",
                denominator="selected_vehicle_count",
                train_seeds=ablation_train_seeds,
                eval_seeds=ablation_eval_seeds,
            )
            if row[f"{metric}_status"] != expected_ratio["status"]:
                raise ValueError
            for field in (
                "mean",
                "std",
                "ci95",
                "seed_mean",
                "seed_std",
                "seed_ci95",
                "denominator_total",
            ):
                actual = float(row[f"{metric}_{field}"])
                expected = float(expected_ratio[field])
                if math.isfinite(expected):
                    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
                        raise ValueError
                elif math.isfinite(actual):
                    raise ValueError
            for field in ("replicates", "valid_count", "seed_replicates"):
                if int(row[f"{metric}_{field}"]) != int(expected_ratio[field]):
                    raise ValueError

            prefix = f"{metric}_change_vs_full_pct"
            if row[f"{prefix}_status"] != expected_delta["status"]:
                raise ValueError
            if int(row[f"{prefix}_pairs"]) != int(expected_delta["pairs"]):
                raise ValueError
            for field, expected_key in (
                (prefix, "mean"),
                (f"{prefix}_std", "std"),
                (f"{prefix}_ci95", "ci95"),
                (f"{prefix}_paired_permutation_p", "paired_permutation_p"),
            ):
                actual = float(row[field])
                expected = float(expected_delta[expected_key])
                if math.isfinite(expected):
                    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
                        raise ValueError
                elif math.isfinite(actual):
                    raise ValueError
        except (KeyError, TypeError, ValueError):
            issues.append(
                f"ablation {arm}/{metric}: invalid conditional statistics"
            )

    pruning_root = outputs / "pruning_sensitivity" / "matched"
    pruning = _rows(pruning_root / "summary.csv", issues)
    try:
        pruning_cells = {
            (
                row.get("sensitivity_mode"),
                float(row["r0_m"]),
                float(row["diversity_factor"]),
            )
            for row in pruning
            if row.get("algorithm") == "proposed"
        }
    except (KeyError, TypeError, ValueError):
        pruning_cells = set()
    expected_reported_cells = {
        ("pruning_only", float(radius), float(rho))
        for radius, rho in pruning_sensitivity_reported_cells(cfg)
    }
    if pruning_cells != expected_reported_cells:
        issues.append(f"pruning sensitivity has {len(pruning_cells)} proposed cells, expected 16")
    pruning_episodes = _rows(pruning_root / "episodes.csv", issues)
    _validate_priority_success_metrics(pruning_episodes, "pruning sensitivity", issues)
    validation_seeds = tuple(cfg["seeds"]["validation"])
    noncanonical_specs = {
        (
            float(spec.scenario_overrides["candidate_pruning_radius_m"]),
            float(spec.scenario_overrides["diversity_factor"]),
            int(spec.train_seed),
        ): spec
        for spec in pruning_sensitivity_specs(cfg)
    }

    def pruning_provenance(row: dict[str, str]) -> ArtifactProvenance:
        if row["algorithm"] == "dbas":
            return _stateless_provenance(_pruning_stateless_config(row, cfg))
        algorithm = row["algorithm"]
        seed = int(row["train_seed"])
        if algorithm != "proposed":
            return headline_provenance[(algorithm, seed)]
        radius = float(row["r0_m"])
        trained_rho = pruning_sensitivity_training_rho(float(row["diversity_factor"]), cfg)
        canonical = (
            float(cfg["pruning_sensitivity"]["canonical_radius_m"]),
            float(cfg["pruning_sensitivity"]["canonical_diversity_factor"]),
        )
        if (radius, trained_rho) == canonical:
            return headline_provenance[("proposed", seed)]
        spec = noncanonical_specs[(radius, trained_rho, seed)]
        expected_config = orch.materialize_config(spec.to_worker_dict(), validation_seeds)
        return _trained_provenance(
            "proposed",
            pruning_sensitivity_run_dir(radius, trained_rho, seed, cfg),
            expected_config,
            provenance_cache,
        )

    _validate_expected_row_provenance(
        pruning_episodes, "pruning sensitivity", pruning_provenance, issues
    )
    sensitivity_seeds = tuple(cfg["pruning_sensitivity"]["train_seeds"])
    pruning_eval_seeds = tuple(cfg["seeds"]["pruning_test"])
    expected_pruning = {
        (algorithm, "pruning_only", float(radius), float(rho), int(train_seed), int(eval_seed))
        for algorithm in ALL_ALGORITHMS
        for radius, rho in pruning_sensitivity_reported_cells(cfg)
        for train_seed in (sensitivity_seeds[:1] if algorithm == "dbas" else sensitivity_seeds)
        for eval_seed in pruning_eval_seeds
    }
    try:
        actual_pruning = [
            (
                row["algorithm"],
                row["sensitivity_mode"],
                float(row["r0_m"]),
                float(row["diversity_factor"]),
                int(row["train_seed"]),
                int(row["eval_seed"]),
            )
            for row in pruning_episodes
        ]
        if set(actual_pruning) != expected_pruning:
            issues.append("pruning sensitivity Cartesian product is incomplete")
        if len(actual_pruning) != len(set(actual_pruning)):
            issues.append("pruning sensitivity contains duplicate identities")
    except (KeyError, TypeError, ValueError):
        issues.append("pruning sensitivity contains invalid identities")
    for row in pruning_episodes:
        try:
            if row["sensitivity_mode"] != "pruning_only":
                raise ValueError
            if row["sensitivity_protocol"] != "matched_training":
                raise ValueError
            if row["evidence_status"] != "complete_paper":
                raise ValueError
            if row["runtime_evidence_status"] != "diagnostic_unwarmed_non_final":
                raise ValueError
            if row["algorithm"] == "proposed":
                reported_rho = float(row["diversity_factor"])
                trained_rho = float(row["trained_diversity_factor"])
                if trained_rho != pruning_sensitivity_training_rho(reported_rho, cfg):
                    raise ValueError
                expected_equivalence = reported_rho == 1.0
                if str(row["analytically_equivalent"]).lower() != str(expected_equivalence).lower():
                    raise ValueError
        except (KeyError, TypeError, ValueError):
            issues.append("pruning sensitivity contains invalid matched-training metadata")
            break
    _validate_pruning_equivalent_cells(pruning_episodes, issues)

    timing_metrics = (
        "candidate_pruning_time_ms",
        "follower_solve_time_ms",
        "aggregation_time_ms",
        "inference_wall_time_ms",
    )
    if pruning and any(f"{metric}_mean" in pruning[0] for metric in timing_metrics):
        issues.append("formal pruning summary must exclude unwarmed timing metrics")

    paired = _rows(pruning_root / "paired_deltas.csv", issues)
    if len(paired) != 16 * 4 * 2:
        issues.append(f"pruning sensitivity has {len(paired)} paired rows, expected 128")
    for row in paired:
        try:
            if int(row["pairs"]) != len(sensitivity_seeds):
                raise ValueError
            for field in ("change_pct", "change_pct_std", "change_pct_ci95"):
                if not math.isfinite(float(row[field])):
                    raise ValueError
            p_value = float(row["paired_permutation_p"])
            if not 0.0 <= p_value <= 1.0:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            issues.append("pruning sensitivity contains invalid paired statistics")
            break
    rank_rows = _rows(pruning_root / "rank_stability.csv", issues)
    expected_rank_rows = 16 * len(ALL_ALGORITHMS) * 2
    if len(rank_rows) != expected_rank_rows:
        issues.append(
            f"pruning sensitivity has {len(rank_rows)} rank rows, expected {expected_rank_rows}"
        )
    canonical_radius = float(cfg["pruning_sensitivity"]["canonical_radius_m"])
    canonical_rho = float(cfg["pruning_sensitivity"]["canonical_diversity_factor"])
    for row in rank_rows:
        try:
            if (
                float(row["canonical_r0_m"]) != canonical_radius
                or float(row["canonical_diversity_factor"]) != canonical_rho
            ):
                raise ValueError
            if (
                float(row["r0_m"]) == canonical_radius
                and float(row["diversity_factor"]) == canonical_rho
                and int(row["rank"]) != int(row["default_rank"])
            ):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            issues.append("pruning sensitivity contains invalid canonical ranks")
            break

    paper_table = _rows(pruning_root / "paper_table.csv", issues)
    try:
        paper_cells = {(float(row["r0_m"]), float(row["diversity_factor"])) for row in paper_table}
        if (
            paper_cells
            != {
                (float(radius), float(rho))
                for radius, rho in pruning_sensitivity_reported_cells(cfg)
            }
            or len(paper_table) != 16
        ):
            issues.append("pruning paper table does not contain exactly 16 cells")
        numeric_fields = (
            "social_welfare_mean",
            "social_welfare_ci95",
            "social_welfare_change_vs_canonical_pct",
            "social_welfare_change_vs_ebas_pct",
            "rsu_utility_mean",
            "rsu_utility_ci95",
            "rsu_utility_change_vs_canonical_pct",
            "rsu_utility_change_vs_ebas_pct",
            "perception_quality_mean",
            "perception_quality_ci95",
            "task_success_rate_mean",
            "task_success_rate_ci95",
        )
        for row in paper_table:
            if any(not math.isfinite(float(row[field])) for field in numeric_fields):
                raise ValueError
            if int(row["effective_candidate_cap"]) not in (2, 3, 4):
                raise ValueError
            if not 1 <= int(row["social_welfare_rank"]) <= len(ALL_ALGORITHMS):
                raise ValueError
            if not 1 <= int(row["rsu_utility_rank"]) <= len(ALL_ALGORITHMS):
                raise ValueError
    except (KeyError, TypeError, ValueError):
        issues.append("pruning paper table contains invalid values")

    for axis, grid in (
        ("vehicles", cfg["scalability"]["vehicle_grid"]),
        ("tasks", cfg["scalability"]["task_grid"]),
    ):
        scale_seed_bank = str(cfg["scalability"]["evaluation_seed_bank"])
        scale_eval_seeds = tuple(cfg["seeds"][scale_seed_bank])
        _validate_evaluation_contract(
            outputs / "scalability" / axis,
            label=f"scalability/{axis}",
            expected={
                "revision_id": cfg["revision_id"],
                "revision_schema_version": int(cfg["schema_version"]),
                "experiment": f"scalability_{axis}",
                "evaluation_seed_bank": scale_seed_bank,
                "evaluation_seeds": list(scale_eval_seeds),
                "training_seeds": list(cfg["seeds"]["sweep_train"]),
                "algorithms": list(cfg["scalability"]["algorithms"]),
                "grid": list(grid),
                "aggregation_unit": "training_seed_mean_over_evaluation_scenarios",
            },
            issues=issues,
        )
        rows = _rows(outputs / "scalability" / axis / "summary.csv", issues)
        for algorithm in cfg["scalability"]["algorithms"]:
            count = sum(row.get("algorithm") == algorithm for row in rows)
            if count != len(grid):
                issues.append(f"scalability/{axis} {algorithm}: {count} cells != {len(grid)}")
        episode_rows = _rows(outputs / "scalability" / axis / "episodes.csv", issues)
        _validate_priority_success_metrics(episode_rows, f"scalability/{axis}", issues)
        if any(
            row.get("runtime_evidence_status") != "diagnostic_unwarmed_non_final"
            for row in episode_rows
        ):
            issues.append(
                f"scalability/{axis} episode timing must be labelled diagnostic"
            )
        try:
            scale_provenance = _scalability_expectations(
                axis, cfg, headline_provenance, provenance_cache
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            issues.append(f"scalability/{axis}: cannot build current provenance: {exc}")
            scale_provenance = {}
        _validate_expected_row_provenance(
            episode_rows,
            f"scalability/{axis}",
            lambda row, expected=scale_provenance: expected[
                (row["algorithm"], int(row["train_seed"]), float(row["sweep_value"]))
            ],
            issues,
        )
        expected_scale = {
            (algorithm, float(value), int(train_seed), int(eval_seed))
            for algorithm in cfg["scalability"]["algorithms"]
            for value in grid
            for train_seed in (
                (cfg["seeds"]["sweep_train"][0],)
                if algorithm == "dbas"
                else cfg["seeds"]["sweep_train"]
            )
            for eval_seed in scale_eval_seeds
        }
        try:
            actual_scale = [
                (
                    row["algorithm"],
                    float(row["sweep_value"]),
                    int(row["train_seed"]),
                    int(row["eval_seed"]),
                )
                for row in episode_rows
            ]
            if set(actual_scale) != expected_scale:
                issues.append(f"scalability/{axis} Cartesian product is incomplete")
            if len(actual_scale) != len(set(actual_scale)):
                issues.append(f"scalability/{axis} contains duplicate identities")
        except (KeyError, TypeError, ValueError):
            issues.append(f"scalability/{axis} contains invalid identities")

    dynamic = _rows(outputs / "dynamic_robustness" / "summary.csv", issues)
    dynamic_episodes = _rows(outputs / "dynamic_robustness" / "episodes.csv", issues)
    dynamic_seed_bank = str(cfg["scalability"]["dynamic_evaluation_seed_bank"])
    dynamic_eval_seeds = tuple(cfg["seeds"][dynamic_seed_bank])
    _validate_evaluation_contract(
        outputs / "dynamic_robustness",
        label="dynamic robustness",
        expected={
            "revision_id": cfg["revision_id"],
            "revision_schema_version": int(cfg["schema_version"]),
            "experiment": "dynamic_robustness",
            "evaluation_seed_bank": dynamic_seed_bank,
            "evaluation_seeds": list(dynamic_eval_seeds),
            "training_seeds": list(cfg["seeds"]["sweep_train"]),
            "algorithms": list(cfg["scalability"]["algorithms"]),
            "aggregation_unit": "training_seed_mean_over_evaluation_scenarios",
        },
        issues=issues,
    )
    _validate_priority_success_metrics(dynamic_episodes, "dynamic robustness", issues)
    if any(
        row.get("runtime_evidence_status") != "diagnostic_unwarmed_non_final"
        for row in dynamic_episodes
    ):
        issues.append("dynamic robustness episode timing must be labelled diagnostic")
    dynamic_overrides = _dynamic_override_map(cfg)

    def dynamic_provenance(row: dict[str, str]) -> ArtifactProvenance:
        if row["algorithm"] != "dbas":
            return headline_provenance[(row["algorithm"], int(row["train_seed"]))]
        config = copy.deepcopy(formal_algorithm_config("dbas", cfg))
        config.setdefault("scenario", {}).update(
            dynamic_overrides[(row["stress_experiment"], row["stress_level"])]
        )
        return _stateless_provenance(config)

    _validate_expected_row_provenance(
        dynamic_episodes, "dynamic robustness", dynamic_provenance, issues
    )
    for algorithm in cfg["scalability"]["algorithms"]:
        count = sum(row.get("algorithm") == algorithm for row in dynamic)
        expected_cell_count = (
            len(cfg["scalability"]["occlusion_bands"])
            + len(cfg["scalability"]["path_loss_grid"])
            + len(cfg["scalability"]["rsu_bandwidth_grid_hz"])
        )
        if count != expected_cell_count:
            issues.append(
                f"dynamic robustness {algorithm}: {count} cells != {expected_cell_count}"
            )
    dynamic_levels = [("occlusion", str(level)) for level in cfg["scalability"]["occlusion_bands"]]
    dynamic_levels.extend(
        ("path_loss", f"alpha_{float(value):g}") for value in cfg["scalability"]["path_loss_grid"]
    )
    dynamic_levels.extend(
        ("rsu_bandwidth", f"{float(value) / 1e6:g}MHz")
        for value in cfg["scalability"]["rsu_bandwidth_grid_hz"]
    )
    expected_dynamic = {
        (experiment, level, algorithm, int(train_seed), int(eval_seed))
        for experiment, level in dynamic_levels
        for algorithm in cfg["scalability"]["algorithms"]
        for train_seed in (
            (cfg["seeds"]["sweep_train"][0],)
            if algorithm == "dbas"
            else cfg["seeds"]["sweep_train"]
        )
        for eval_seed in dynamic_eval_seeds
    }
    try:
        actual_dynamic = [
            (
                row["stress_experiment"],
                row["stress_level"],
                row["algorithm"],
                int(row["train_seed"]),
                int(row["eval_seed"]),
            )
            for row in dynamic_episodes
        ]
        if set(actual_dynamic) != expected_dynamic:
            issues.append("dynamic robustness Cartesian product is incomplete")
        if len(actual_dynamic) != len(set(actual_dynamic)):
            issues.append("dynamic robustness contains duplicate identities")
    except (KeyError, TypeError, ValueError):
        issues.append("dynamic robustness contains invalid identities")

    timing = _rows(outputs / "tables" / "inference_latency.csv", issues)
    expected_timing = len(cfg["scalability"]["timing_vehicle_grid"]) * len(
        cfg["scalability"]["timing_task_grid"]
    )
    if len(timing) != expected_timing:
        issues.append(f"inference timing: {len(timing)} cells != {expected_timing}")
    timing_cells: list[tuple[int, int]] = []
    try:
        timing_provenance = _timing_expectations(cfg, headline_provenance, provenance_cache)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        issues.append(f"inference timing: cannot build current provenance: {exc}")
        timing_provenance = {}
    for row in timing:
        try:
            cell = (int(row["num_vehicles"]), int(row["num_tasks"]))
            timing_cells.append(cell)
            if row.get("benchmark_kind") != "checkpoint":
                issues.append("inference timing contains non-checkpoint evidence")
            if int(row.get("warmup_repetitions", 0)) < 20:
                issues.append("inference timing warm-up is below 20 repetitions")
            if int(row.get("repetitions", 0)) < 200:
                issues.append("inference timing measurement is below 200 repetitions")
            _compare_provenance(row, timing_provenance[cell], "inference timing", issues)
        except (KeyError, TypeError, ValueError):
            issues.append("inference timing contains an invalid grid identity")
    expected_timing_cells = {
        (int(v), int(k))
        for v in cfg["scalability"]["timing_vehicle_grid"]
        for k in cfg["scalability"]["timing_task_grid"]
    }
    if set(timing_cells) != expected_timing_cells or len(timing_cells) != len(set(timing_cells)):
        issues.append("inference timing grid is incomplete or duplicated")

    ablation_timing = _rows(
        outputs / "tables" / "ablation_inference_latency.csv", issues
    )
    expected_ablation_timing = {
        (arm, int(seed))
        for arm in ABLATION_ARMS
        for seed in cfg["seeds"]["ablation_train"]
    }
    actual_ablation_timing: list[tuple[str, int]] = []
    for row in ablation_timing:
        try:
            identity = (row["ablation_arm"], int(row["train_seed"]))
            actual_ablation_timing.append(identity)
            if row.get("benchmark_kind") != "checkpoint_warmed":
                raise ValueError
            if int(row["warmup_repetitions"]) < 20 or int(row["repetitions"]) < 200:
                raise ValueError
            total = float(row["total_cycle_ms"])
            p95 = float(row["cycle_p95_ms"])
            if not math.isfinite(total) or total <= 0.0 or p95 < total:
                raise ValueError
            _compare_provenance(
                row,
                ablation_provenance[identity],
                "ablation inference timing",
                issues,
            )
        except (KeyError, TypeError, ValueError):
            issues.append("ablation inference timing contains an invalid row")
    if (
        set(actual_ablation_timing) != expected_ablation_timing
        or len(actual_ablation_timing) != len(set(actual_ablation_timing))
    ):
        issues.append("ablation inference timing grid is incomplete or duplicated")
    ablation_timing_summary = _rows(
        outputs / "tables" / "ablation_inference_latency_summary.csv", issues
    )
    if {row.get("ablation_arm") for row in ablation_timing_summary} != set(
        ABLATION_ARMS
    ):
        issues.append("ablation timing summary does not contain exactly five arms")
    for row in ablation_timing_summary:
        try:
            if row["benchmark_kind"] != "checkpoint_warmed_seed_summary":
                raise ValueError
            if int(row["seed_count"]) != len(cfg["seeds"]["ablation_train"]):
                raise ValueError
            for metric in ("total_cycle_ms", "cycle_p95_ms", "inference_wall_ms"):
                source_values = [
                    float(source[metric])
                    for source in ablation_timing
                    if source["ablation_arm"] == row["ablation_arm"]
                ]
                expected_mean = statistics.mean(source_values)
                expected_std = statistics.stdev(source_values)
                expected_ci95 = 1.96 * expected_std / math.sqrt(len(source_values))
                for suffix in ("mean", "std", "ci95"):
                    actual = float(row[f"{metric}_{suffix}"])
                    expected = {
                        "mean": expected_mean,
                        "std": expected_std,
                        "ci95": expected_ci95,
                    }[suffix]
                    if not math.isfinite(actual) or not math.isclose(
                        actual, expected, rel_tol=1e-10, abs_tol=1e-10
                    ):
                        raise ValueError
        except (KeyError, TypeError, ValueError):
            issues.append("ablation timing summary contains an invalid row")
    _validate_hardware_file(outputs / "tables" / "hardware.json", "timing hardware.json", issues)

    isolated_seed = int(cfg["seeds"]["isolated_timing_train"][0])
    for algorithm in ("proposed", "jdls", "mappo"):
        record_path = (
            artifacts
            / "headline"
            / f"{algorithm}_s{isolated_seed}_headline"
            / "training_record.json"
        )
        if not record_path.exists():
            issues.append(f"missing isolated training record: {record_path}")
            continue
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            issues.append(f"invalid isolated training record: {record_path}")
            continue
        if record.get("execution_mode") != "isolated":
            issues.append(f"{algorithm}: headline timing seed was not isolated")
        if int(record.get("train_seed", -1)) != isolated_seed:
            issues.append(f"{algorithm}: isolated training record has wrong seed")
        hardware = record.get("hardware")
        if not isinstance(hardware, dict):
            issues.append(f"{algorithm}: isolated record lacks hardware metadata")
        else:
            _validate_hardware_payload(hardware, f"{algorithm} isolated hardware", issues)
    if not (outputs / "calibration" / "dbas_price_calibration.csv").exists():
        issues.append("missing DBAS validation-price calibration table")
    _validate_convergence(outputs, cfg, issues, provenance_cache)
    _validate_fairness_settings(outputs, cfg, issues)
    for filename in (
        "summary_stats.csv",
        "action_dimensions.csv",
        "training_cost.csv",
    ):
        if not (outputs / "tables" / filename).exists():
            issues.append(f"missing report: {outputs / 'tables' / filename}")
    for stem in (
        "figR_learning_efficiency",
        "figR_reward_convergence",
        "figR_ablation",
        "figR_scalability_vehicles",
        "figR_scalability_vehicles_full",
        "figR_scalability_tasks",
        "figR_scalability_tasks_full",
        "figR_dynamic_robustness",
        "figR_dynamic_robustness_full",
        "figR_resource_sensitivity",
        "figR_resource_sensitivity_full",
        "figR_mechanism_analysis",
        "figR_latency",
        "figR_pruning_sensitivity",
    ):
        for suffix in (".png", ".pdf"):
            path = outputs / "figures" / f"{stem}{suffix}"
            if not path.exists():
                issues.append(f"missing figure: {path}")
    for suffix in (".png", ".pdf"):
        path = (
            outputs / "pruning_sensitivity" / "matched" / f"pruning_sensitivity_comparison{suffix}"
        )
        if not path.exists():
            issues.append(f"missing pruning sensitivity figure: {path}")
    pruning_source = outputs / "figures" / "figR_pruning_sensitivity_source.csv"
    if not pruning_source.exists():
        issues.append(f"missing pruning sensitivity figure source: {pruning_source}")
    for filename in (
        "figR_resource_sensitivity_source.csv",
        "figR_mechanism_analysis_source.csv",
    ):
        source = outputs / "figures" / filename
        if not source.exists():
            issues.append(f"missing main-text figure source: {source}")
    resource_source = outputs / "figures" / "figR_resource_sensitivity_source.csv"
    if resource_source.exists():
        rows = _rows(resource_source, issues)
        if _has_invalid_dbas_uncertainty(rows):
            issues.append(
                "resource sensitivity source fabricates a DBAS train-seed interval"
            )
    _validate_manuscript_figures(outputs, cfg, issues)
    return issues


def validate_preflight(cfg: dict) -> list[str]:
    issues: list[str] = []
    if int(cfg.get("training_semantics_version", 0)) <= 0:
        issues.append("training_semantics_version must be positive")
    manuscript = cfg.get("manuscript_figures", {})
    if int(manuscript.get("contract_version", 0)) != 3:
        issues.append("manuscript figure contract version must be 3")
    output_subdir = Path(str(manuscript.get("output_subdir", "")))
    if output_subdir.is_absolute() or ".." in output_subdir.parts:
        issues.append("manuscript figure output_subdir must be a safe relative path")
    expected_algorithms = {
        "proposed", "ebas", "jdls", "mappo", "dbas", "rdus", "spis"
    }
    if set(manuscript.get("full_algorithms", ())) != expected_algorithms:
        issues.append("full manuscript figures must retain all seven algorithms")
    if set(manuscript.get("paper_algorithms", ())) != {
        "proposed", "ebas", "jdls", "mappo", "dbas"
    }:
        issues.append("paper manuscript figures must use the declared five algorithms")
    if set(manuscript.get("convergence_algorithms", ())) != {
        "proposed", "ebas", "jdls", "mappo"
    }:
        issues.append("manuscript convergence must contain four trained trajectories")
    if int(cfg.get("execution", {}).get("isolated_timing_max_workers", 0)) != 1:
        issues.append("isolated timing must use exactly one worker")
    evaluation_banks = (
        "validation", "test", "pruning_test", "scalability_test", "dynamic_test",
        "main_text_test"
    )
    for index, left in enumerate(evaluation_banks):
        for right in evaluation_banks[index + 1:]:
            if set(cfg["seeds"][left]) & set(cfg["seeds"][right]):
                issues.append(f"evaluation seed banks {left} and {right} overlap")
    if len(cfg["seeds"]["headline_train"]) < 8:
        issues.append("stochastic algorithms require at least 8 common train seeds")
    if tuple(cfg["pruning_sensitivity"]["train_seeds"]) != tuple(cfg["seeds"]["headline_train"]):
        issues.append("pruning sensitivity must use all headline train seeds")
    if len(cfg["seeds"]["ablation_train"]) < 5:
        issues.append("ablation requires at least 5 train seeds")
    if len(cfg["seeds"]["sweep_train"]) < 5:
        issues.append("scalability requires at least 5 train seeds")
    if len(cfg["seeds"]["scalability_test"]) != 32:
        issues.append("scalability requires exactly 32 evaluation seeds")
    if len(cfg["seeds"]["dynamic_test"]) != 32:
        issues.append("dynamic robustness requires exactly 32 evaluation seeds")
    if len(cfg["seeds"]["main_text_test"]) != 32:
        issues.append("main-text analysis requires exactly 32 evaluation seeds")
    for algorithm in ALL_ALGORITHMS:
        config = load_algorithm_config(algorithm)
        scenario = config.get("scenario", {})
        if float(scenario.get("pricing_budget_max", 0.0)) != 0.3:
            issues.append(f"{algorithm}: pricing_budget_max must be 0.3")
        if scenario.get("roi_distance_model") != "rectangle":
            issues.append(f"{algorithm}: formal distance model must be rectangle")
        if float(scenario.get("spatial_potential_max", 0.0)) != 50.0:
            issues.append(f"{algorithm}: formal spatial-potential cap must be 50")
        if algorithm == "proposed":
            if scenario.get("enable_candidate_pruning") is not True:
                issues.append("proposed: canonical candidate pruning must be enabled")
            if float(scenario.get("diversity_factor", 0.0)) != 4.0:
                issues.append("proposed: canonical diversity_factor must be 4")
        if algorithm != "dbas":
            reward = config.get("reward", {})
            if tuple(
                float(reward.get(key, -1.0)) for key in ("omega_fail", "omega_lat", "omega_part")
            ) != (2.0, 3.0, 3.0):
                issues.append(f"{algorithm}: reward weights must be 2/3/3")
        if algorithm in STOCHASTIC_ALGORITHMS:
            formal = formal_algorithm_config(algorithm, cfg, require_dbas_calibration=False)
            algo = formal["algorithm"]
            fairness = cfg["fairness"]
            if int(algo["total_timesteps"]) != int(fairness["total_timesteps"]):
                issues.append(f"{algorithm}: total_timesteps violates fairness contract")
            if float(algo["learning_rate"]) != float(fairness["learning_rate"]):
                issues.append(f"{algorithm}: learning_rate violates fairness contract")
            if float(algo["clip_range"]) != float(fairness["clip_range"]):
                issues.append(f"{algorithm}: clip_range violates fairness contract")
            expected_hidden = list(map(int, fairness["hidden_sizes"]))
            if algorithm in ("jdls", "mappo"):
                actual_hidden = list(map(int, algo.get("hidden_sizes", ())))
                hidden_sets = (actual_hidden,)
            else:
                policy = formal.get("policy", {})
                hidden_sets = (
                    list(map(int, policy.get("actor", {}).get("hidden_sizes", ()))),
                    list(map(int, policy.get("critic", {}).get("hidden_sizes", ()))),
                )
            if any(values != expected_hidden for values in hidden_sets):
                issues.append(f"{algorithm}: hidden sizes violate fairness contract")
            if algorithm == "spis" and float(algo.get("fixed_price", -1.0)) != float(
                fairness["spis_fixed_price"]
            ):
                issues.append("spis: fixed price violates fairness contract")
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/revision_2026_v5/validation_report.json"),
    )
    args = parser.parse_args(argv)
    cfg = load_revision_config()
    issues = validate_preflight(cfg)
    if not args.preflight:
        issues.extend(validate_final(cfg))
    payload: dict[str, Any] = {
        "schema_version": int(cfg.get("schema_version", 2)),
        "revision_id": cfg["revision_id"],
        "mode": "preflight" if args.preflight else "final",
        "status": "pass" if not issues else "fail",
        "issues": issues,
    }
    if not args.preflight:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
