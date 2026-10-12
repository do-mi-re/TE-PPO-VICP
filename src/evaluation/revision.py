"""Maintained configuration loader for the configured experiments.

Exactness label: implementation convenience only.  Mathematical definitions
and statistical units are documented in ``docs/revision_experiment_protocol.md``.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from src.evaluation.provenance import config_digest, sha256_file
from src.evaluation.tasks import RunSpec
from src.sim.scenario import make_urgency_profiles

DEFAULT_REVISION_CONFIG = Path("configs/revision.yaml")
CORE_ALGORITHMS = ("proposed", "jdls", "mappo")
OTHER_STOCHASTIC_ALGORITHMS = ("ebas", "spis", "rdus")
STOCHASTIC_ALGORITHMS = CORE_ALGORITHMS + OTHER_STOCHASTIC_ALGORITHMS
ALL_ALGORITHMS = CORE_ALGORITHMS + ("dbas",) + OTHER_STOCHASTIC_ALGORITHMS


def dbas_calibration_contract_digest(
    config: dict[str, Any],
    *,
    price_grid: tuple[float, ...] | None = None,
    validation_seeds: tuple[int, ...] | None = None,
) -> str:
    """Bind headline calibration to its inputs and selection implementation."""
    from src.evaluation.sweep import load_algorithm_config

    payload = {
        "training_semantics_version": config["training_semantics_version"],
        "validation_seeds": tuple(
            config["seeds"]["validation"] if validation_seeds is None else validation_seeds
        ),
        "price_grid": tuple(
            config["baseline_calibration"]["dbas_price_grid"]
            if price_grid is None else price_grid
        ),
        "dbas_config_digest": config_digest(load_algorithm_config("dbas")),
        "calibration_source_sha256": sha256_file(
            Path(__file__).resolve().parents[2] / "scripts" / "tune_dbas_price.py"
        ),
    }
    digest = config_digest(payload)
    if digest is None:
        raise ValueError("DBAS calibration contract is not digestible")
    return digest


def cpu_robustness_contract_digest(config: dict[str, Any]) -> str:
    """Digest the CPU training, validation, and calibration contract."""
    from src.evaluation.sweep import load_algorithm_config

    payload = {
        "revision_id": config["revision_id"],
        "training_semantics_version": config["training_semantics_version"],
        "cpu_robustness": config["cpu_robustness"],
        "fairness": config["fairness"],
        "baseline_calibration": config["baseline_calibration"],
        "checkpoint_selection": config["checkpoint_selection"],
        "validation_seeds": config["seeds"][
            config["cpu_robustness"]["validation_seed_bank"]
        ],
        "dbas_config_digest": config_digest(load_algorithm_config("dbas")),
        "calibration_source_sha256": sha256_file(
            Path(__file__).resolve().parents[2] / "scripts" / "run_cpu_robustness_experiment.py"
        ),
    }
    digest = config_digest(payload)
    if digest is None:
        raise ValueError("CPU robustness contract is not digestible")
    return digest


def formal_provenance_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Dotted fields embedded into every formal experiment training config."""

    return {
        "provenance.revision_id": str(config["revision_id"]),
        "provenance.training_semantics_version": int(
            config["training_semantics_version"]
        ),
    }


def _with_formal_provenance(
    overrides: dict[str, Any], config: dict[str, Any]
) -> tuple[tuple[str, Any], ...]:
    merged = {**formal_provenance_overrides(config), **overrides}
    return tuple(sorted(merged.items()))


def load_revision_config(path: str | Path = DEFAULT_REVISION_CONFIG) -> dict[str, Any]:
    """Load and validate the experiment configuration."""
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    required = {"schema_version", "revision_id", "training_semantics_version",
                "paths", "execution", "seeds", "statistics",
                "common_evaluation_reward", "joint_learning_baselines", "ablation",
                "checkpoint_selection",
                "candidate_pruning", "pruning_sensitivity",
                "scalability", "cpu_robustness", "main_text_analysis", "fairness",
                "baseline_calibration"}
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"revision config {config_path} misses {missing}")
    if int(payload["schema_version"]) != 13:
        raise ValueError("revision config schema_version must be 13")
    if int(payload["training_semantics_version"]) != 4:
        raise ValueError("revision training_semantics_version must be 4")

    execution = payload["execution"]
    sb3_cpu_slots = int(execution.get("sb3_cpu_slots_per_task", 0))
    if not 1 <= sb3_cpu_slots <= 9:
        raise ValueError(
            "execution.sb3_cpu_slots_per_task must lie in [1, 9] for the "
            "maintained 8-environment SB3 layout"
        )

    common_reward = payload["common_evaluation_reward"]
    if common_reward.get("schema") != "common_evaluation_reward_v1":
        raise ValueError("common evaluation reward schema must be v1")
    if tuple(common_reward.get("plotted_algorithms", ())) != (
        "proposed", "jdls", "mappo", "ebas"
    ):
        raise ValueError("common evaluation reward plotted algorithms are invalid")
    if tuple(common_reward.get("figure_algorithms", ())) != (
        "proposed", "jdls", "mappo"
    ):
        raise ValueError("common evaluation reward figure algorithms are invalid")
    if common_reward.get("penalties") != {
        "omega_fail": 2.0,
        "omega_lat": 3.0,
        "omega_part": 3.0,
        "min_participants": 1,
    }:
        raise ValueError("common evaluation reward penalties are invalid")

    joint = payload["joint_learning_baselines"]
    if int(joint.get("protocol_version", 0)) != 2:
        raise ValueError("joint-learning baseline protocol version must be 2")
    if tuple(joint.get("algorithms", ())) != ("jdls", "mappo"):
        raise ValueError("joint-learning baseline algorithm set is invalid")
    if joint.get("initialization") != "cold_start_no_teacher":
        raise ValueError("joint-learning baselines must use cold start")
    if joint.get("action_parameterization") != "candidate_masked_feasible_v1":
        raise ValueError("joint-learning action parameterization is invalid")
    if int(joint.get("rollout_samples_per_update", 0)) != 1024:
        raise ValueError("joint-learning rollout batch must contain 1024 samples")
    if int(joint.get("mappo_collection_microbatch_size", 0)) != 32:
        raise ValueError("MAPPO collection microbatch size must be 32")
    expected_versions = {
        "jdls": "jdls_constraint_aware_cold_start_v1",
        "mappo": "mappo_constraint_aware_cold_start_v2",
    }
    if joint.get("implementation_versions") != expected_versions:
        raise ValueError("joint-learning implementation versions are invalid")
    expected_joint_reward = {
        "omega_vehicle": 0.001,
        "omega_feasible": 1.0,
        "omega_pre_quality": 0.01,
        "omega_tau": 0.0,
        "omega_bandwidth_use": 0.0,
        "omega_pre_lat": 0.01,
        "pre_lat_overshoot_cap": 10.0,
        "penalize_missing_latency_as_cap": True,
    }
    if joint.get("reward") != expected_joint_reward:
        raise ValueError("joint-learning reward contract is invalid")
    seeds = payload["seeds"]
    for key in ("headline_train", "sweep_train", "ablation_train",
                "isolated_timing_train", "validation", "test",
                "pruning_test", "scalability_test", "dynamic_test",
                "main_text_test"):
        values = tuple(int(v) for v in seeds.get(key, ()))
        if not values or len(values) != len(set(values)):
            raise ValueError(f"revision seed set {key!r} must be non-empty and unique")
        seeds[key] = values
    evaluation_banks = (
        "validation", "test", "pruning_test", "scalability_test", "dynamic_test",
        "main_text_test"
    )
    for index, left in enumerate(evaluation_banks):
        for right in evaluation_banks[index + 1:]:
            if set(seeds[left]) & set(seeds[right]):
                raise ValueError(
                    f"revision evaluation seed banks {left!r} and {right!r} "
                    "must be disjoint"
                )
    for key in ("sweep_train", "ablation_train", "isolated_timing_train"):
        if not set(seeds[key]).issubset(seeds["headline_train"]):
            raise ValueError(f"{key} seeds must be a subset of headline_train")
    if len(seeds["isolated_timing_train"]) != 1:
        raise ValueError("exactly one common isolated timing seed is required")
    fairness = payload["fairness"]
    if fairness.get("protocol") != "unified_core_hyperparameters":
        raise ValueError("fairness.protocol must be unified_core_hyperparameters")
    hidden = tuple(int(value) for value in fairness.get("hidden_sizes", ()))
    if not hidden or any(value <= 0 for value in hidden):
        raise ValueError("fairness.hidden_sizes must contain positive widths")
    if float(fairness.get("learning_rate", 0.0)) <= 0.0:
        raise ValueError("fairness.learning_rate must be positive")
    if float(fairness.get("clip_range", 0.0)) <= 0.0:
        raise ValueError("fairness.clip_range must be positive")
    if float(fairness.get("ent_coef", -1.0)) != 0.01:
        raise ValueError("fairness.ent_coef must be the fixed common value 0.01")
    if int(fairness.get("total_timesteps", 0)) <= 0:
        raise ValueError("fairness.total_timesteps must be positive")
    fixed_price = float(fairness.get("spis_fixed_price", -1.0))
    if not 0.0 <= fixed_price <= 0.3:
        raise ValueError("fairness.spis_fixed_price must lie in [0, 0.3]")
    if len(seeds["headline_train"]) < 5:
        raise ValueError("fairness requires at least five common headline seeds")
    if len(seeds["sweep_train"]) < 5:
        raise ValueError("scalability requires at least five common train seeds")
    if tuple(seeds["sweep_train"]) != tuple(seeds["headline_train"][:5]):
        raise ValueError("scalability must use the first five common headline seeds")
    if len(seeds["scalability_test"]) != 32:
        raise ValueError("scalability_test must contain exactly 32 seeds")
    if len(seeds["dynamic_test"]) != 32:
        raise ValueError("dynamic_test must contain exactly 32 seeds")
    if len(seeds["main_text_test"]) != 32:
        raise ValueError("main_text_test must contain exactly 32 seeds")
    if tuple(seeds["scalability_test"]) != tuple(range(60_000, 60_032)):
        raise ValueError("formal scalability_test seed bank is invalid")
    if tuple(seeds["dynamic_test"]) != tuple(range(70_000, 70_032)):
        raise ValueError("formal dynamic_test seed bank is invalid")
    if tuple(seeds["main_text_test"]) != tuple(range(80_000, 80_032)):
        raise ValueError("formal main_text_test seed bank is invalid")
    calibration = payload["baseline_calibration"]
    dbas_grid = tuple(float(value) for value in calibration.get("dbas_price_grid", ()))
    if not dbas_grid or len(dbas_grid) != len(set(dbas_grid)):
        raise ValueError("baseline_calibration.dbas_price_grid must be non-empty and unique")
    if any(not 0.0 <= value <= 0.3 for value in dbas_grid):
        raise ValueError("DBAS prices must lie in the formal [0, 0.3] budget")
    if not str(calibration.get("dbas_selection_file", "")).strip():
        raise ValueError("baseline_calibration.dbas_selection_file is required")
    interval = int(payload["ablation"].get("evaluation_interval_env_steps", 0))
    if interval <= 0:
        raise ValueError("ablation evaluation interval must be positive")
    ablation = payload["ablation"]
    if int(ablation.get("learned_follower_protocol_version", 0)) != 2:
        raise ValueError("ablation learned-follower protocol version must be 2")
    if not str(ablation.get("learned_follower_implementation_version", "")).strip():
        raise ValueError("ablation learned-follower implementation version is required")
    if (
        ablation.get("learned_follower_action_parameterization")
        != "candidate_masked_feasible_v1"
    ):
        raise ValueError(
            "ablation learned-follower action parameterization must be "
            "candidate_masked_feasible_v1"
        )
    learned_reward = ablation.get("learned_follower_reward")
    expected_learned_reward = {
        "omega_vehicle": 0.001,
        "omega_feasible": 1.0,
        "omega_pre_quality": 0.01,
        "omega_tau": 0.0,
        "omega_bandwidth_use": 0.0,
        "omega_pre_lat": 0.01,
        "pre_lat_overshoot_cap": 10.0,
    }
    if not isinstance(learned_reward, dict) or {
        key: float(learned_reward.get(key, float("nan")))
        for key in expected_learned_reward
    } != expected_learned_reward:
        raise ValueError("ablation learned-follower reward contract is invalid")
    if learned_reward.get("penalize_missing_latency_as_cap") is not True:
        raise ValueError(
            "ablation learned-follower reward must penalize missing latency as cap"
        )
    selection = payload["checkpoint_selection"]
    if selection.get("metric") != "social_welfare":
        raise ValueError("checkpoint selection metric must be social_welfare")
    if int(selection.get("evaluation_interval_env_steps", 0)) <= 0:
        raise ValueError("checkpoint selection interval must be positive")
    if selection.get("tie_break") != "earliest":
        raise ValueError("checkpoint selection tie_break must be earliest")
    if selection.get("force_terminal_evaluation") is not True:
        raise ValueError("checkpoint selection must force terminal evaluation")

    modes = tuple(payload["candidate_pruning"].get("modes", ()))
    if modes != ("pruning_only",):
        raise ValueError("candidate_pruning.modes must be pruning_only")
    if float(payload["candidate_pruning"].get("default_diversity_factor", 0.0)) != 4.0:
        raise ValueError("candidate_pruning.default_diversity_factor must be 4")
    sensitivity = payload["pruning_sensitivity"]
    if sensitivity.get("protocol") != "matched_training":
        raise ValueError("pruning_sensitivity.protocol must be matched_training")
    if sensitivity.get("mode") != "pruning_only":
        raise ValueError("formal pruning sensitivity mode must be pruning_only")
    radii = tuple(float(value) for value in sensitivity.get("radius_grid_m", ()))
    reported_rhos = tuple(
        float(value) for value in sensitivity.get("reported_diversity_grid", ())
    )
    trained_rhos = tuple(
        float(value) for value in sensitivity.get("trained_diversity_grid", ())
    )
    if radii != (60.0, 80.0, 100.0, 120.0):
        raise ValueError("formal pruning radius grid must be 60, 80, 100, 120 m")
    if reported_rhos != (1.0, 2.0, 3.0, 4.0):
        raise ValueError("reported pruning diversity grid must be 1, 2, 3, 4")
    if trained_rhos != (2.0, 3.0, 4.0):
        raise ValueError("trained pruning diversity grid must be 2, 3, 4")
    sensitivity_seeds = tuple(
        int(value) for value in sensitivity.get("train_seeds", ())
    )
    if sensitivity_seeds != tuple(seeds["headline_train"]):
        raise ValueError(
            "pruning sensitivity must use the common headline train seeds"
        )
    if sensitivity.get("evaluation_seed_bank") != "pruning_test":
        raise ValueError("pruning sensitivity must use the pruning_test seed bank")
    if int(sensitivity.get("total_timesteps", 0)) != int(
        payload["fairness"]["total_timesteps"]
    ):
        raise ValueError("pruning sensitivity must use the common training budget")
    if float(sensitivity.get("canonical_radius_m", 0.0)) != float(
        payload["candidate_pruning"]["default_radius_m"]
    ):
        raise ValueError("pruning sensitivity radius must match the canonical radius")
    if float(sensitivity.get("canonical_diversity_factor", 0.0)) != float(
        payload["candidate_pruning"]["default_diversity_factor"]
    ):
        raise ValueError("pruning sensitivity rho must match the canonical rho")
    sensitivity["radius_grid_m"] = radii
    sensitivity["reported_diversity_grid"] = reported_rhos
    sensitivity["trained_diversity_grid"] = trained_rhos
    sensitivity["train_seeds"] = sensitivity_seeds
    scalability = payload["scalability"]
    if tuple(scalability.get("algorithms", ())) != ALL_ALGORITHMS:
        raise ValueError("formal scalability algorithm set is invalid")
    if scalability.get("evaluation_seed_bank") != "scalability_test":
        raise ValueError("scalability must use the scalability_test seed bank")
    if scalability.get("dynamic_evaluation_seed_bank") != "dynamic_test":
        raise ValueError("dynamic robustness must use the dynamic_test seed bank")
    if tuple(int(value) for value in scalability.get("vehicle_grid", ())) != (
        10, 20, 30, 40, 50, 60, 80, 100
    ):
        raise ValueError("formal vehicle scalability grid is invalid")
    if tuple(int(value) for value in scalability.get("task_grid", ())) != (
        3, 6, 9, 12
    ):
        raise ValueError("formal task scalability grid is invalid")
    if tuple(float(value) for value in scalability.get("path_loss_grid", ())) != (
        2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0
    ):
        raise ValueError("formal path-loss robustness grid is invalid")
    occlusion_bands = scalability.get("occlusion_bands", {})
    if {
        str(label): tuple(float(value) for value in band)
        for label, band in occlusion_bands.items()
    } != {"default": (0.0, 1.0), "severe": (0.5, 0.9)}:
        raise ValueError("formal occlusion robustness categories are invalid")
    if tuple(
        float(value) for value in scalability.get("rsu_bandwidth_grid_hz", ())
    ) != (5e6, 10e6, 15e6, 20e6, 30e6, 40e6):
        raise ValueError("formal RSU-bandwidth robustness grid is invalid")
    cpu_robustness = payload["cpu_robustness"]
    if int(cpu_robustness.get("protocol_version", 0)) != 1:
        raise ValueError("CPU robustness protocol version must be 1")
    if cpu_robustness.get("training_seed_bank") != "headline_train":
        raise ValueError("CPU robustness must use headline_train seeds")
    if cpu_robustness.get("validation_seed_bank") != "validation":
        raise ValueError("CPU robustness must use validation seeds for selection")
    if cpu_robustness.get("evaluation_seed_bank") != "main_text_test":
        raise ValueError("CPU robustness must use main_text_test for final evaluation")
    if int(cpu_robustness.get("total_timesteps", 0)) != 600_000:
        raise ValueError("CPU robustness training budget must be 600000")
    if int(cpu_robustness["total_timesteps"]) != int(fairness["total_timesteps"]):
        raise ValueError("CPU robustness and fairness training budgets must match")
    cpu_ranges = tuple(
        tuple(float(endpoint) for endpoint in value)
        for value in cpu_robustness.get("cpu_capacity_ranges_cycles_per_s", ())
    )
    expected_cpu_ranges = (
        (2.5e9, 7.5e9), (7.5e9, 12.5e9), (12.5e9, 17.5e9),
        (17.5e9, 22.5e9), (22.5e9, 27.5e9),
    )
    if cpu_ranges != expected_cpu_ranges:
        raise ValueError("CPU robustness capacity grid is invalid")
    regime_probabilities = tuple(
        float(value) for value in cpu_robustness.get("regime_probabilities", ())
    )
    if regime_probabilities != (0.2, 0.2, 0.2, 0.2, 0.2):
        raise ValueError("CPU robustness regimes must be sampled uniformly")
    if cpu_robustness.get("checkpoint_selection") != "maximin_cpu_band_social_welfare":
        raise ValueError("CPU robustness checkpoint selection is invalid")
    if not str(cpu_robustness.get("dbas_selection_file", "")).strip():
        raise ValueError("CPU robustness DBAS selection file is required")
    cpu_robustness["cpu_capacity_ranges_cycles_per_s"] = cpu_ranges
    cpu_robustness["regime_probabilities"] = regime_probabilities

    main_text = payload["main_text_analysis"]
    if int(main_text.get("contract_version", 0)) != 2:
        raise ValueError("main-text analysis contract version must be 2")
    if main_text.get("evaluation_seed_bank") != "main_text_test":
        raise ValueError("main-text analysis must use the main_text_test seed bank")
    if main_text.get("train_seed_bank") != "headline_train":
        raise ValueError("main-text analysis must use headline training seeds")
    if tuple(main_text.get("algorithms", ())) != ALL_ALGORITHMS:
        raise ValueError("main-text analysis algorithm set is invalid")
    if tuple(main_text.get("paper_algorithms", ())) != (
        "proposed", "ebas", "jdls", "mappo", "dbas"
    ):
        raise ValueError("main-text paper algorithm subset is invalid")
    main_text_cpu_ranges = tuple(
        tuple(float(endpoint) for endpoint in value)
        for value in main_text.get("cpu_capacity_ranges_cycles_per_s", ())
    )
    if main_text_cpu_ranges != cpu_ranges:
        raise ValueError("main-text CPU-capacity grid is invalid")
    if main_text.get("cpu_policy_source") != "cpu_robustness":
        raise ValueError("main-text CPU policy source must be cpu_robustness")
    compression_grid = tuple(
        float(value) for value in main_text.get("compression_ratio_grid", ())
    )
    if compression_grid != (1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0):
        raise ValueError("main-text compression-ratio grid is invalid")
    price_grid = tuple(float(value) for value in main_text.get("price_scale_grid", ()))
    if price_grid != (0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5):
        raise ValueError("main-text price-scale grid is invalid")
    main_text["cpu_capacity_ranges_cycles_per_s"] = main_text_cpu_ranges
    main_text["compression_ratio_grid"] = compression_grid
    main_text["price_scale_grid"] = price_grid
    return deepcopy(payload)


def train_seeds_for(algorithm: str, *, scope: str = "headline",
                    config: dict[str, Any] | None = None) -> tuple[int, ...]:
    """Return the configured independent training replicates for one scope."""
    cfg = config or load_revision_config()
    if scope == "sweep":
        return tuple(cfg["seeds"]["sweep_train"])
    if scope == "ablation":
        return tuple(cfg["seeds"]["ablation_train"])
    if scope != "headline":
        raise ValueError(f"unknown revision seed scope: {scope!r}")
    if algorithm in STOCHASTIC_ALGORITHMS:
        return tuple(cfg["seeds"]["headline_train"])
    if algorithm == "dbas":
        return (int(cfg["seeds"]["headline_train"][0]),)
    raise ValueError(f"unknown revision algorithm: {algorithm!r}")


def revision_roots(config: dict[str, Any] | None = None) -> tuple[Path, Path]:
    cfg = config or load_revision_config()
    return Path(cfg["paths"]["artifacts_root"]), Path(cfg["paths"]["outputs_root"])


def cpu_robustness_specs(
    config: dict[str, Any] | None = None,
) -> list[RunSpec]:
    """Enumerate the additive CPU-domain-randomized training identities."""

    cfg = config or load_revision_config()
    contract = cfg["cpu_robustness"]
    ranges = tuple(contract["cpu_capacity_ranges_cycles_per_s"])
    probabilities = tuple(contract["regime_probabilities"])
    algorithms = ALL_ALGORITHMS
    seeds = tuple(cfg["seeds"][contract["training_seed_bank"]])
    scenario_overrides = {
        "num_vehicles": 20,
        "cpu_cycles_per_s_regimes": ranges,
        "cpu_regime_probabilities": probabilities,
    }
    specs: list[RunSpec] = []
    for algorithm in algorithms:
        algorithm_seeds = seeds[:1] if algorithm == "dbas" else seeds
        for seed in algorithm_seeds:
            overrides = {
                **fairness_overrides(algorithm, cfg),
                "evaluation.checkpoint_selection": contract["checkpoint_selection"],
                "evaluation.cpu_capacity_ranges_cycles_per_s": ranges,
                "provenance.cpu_robustness_protocol_version": int(
                    contract["protocol_version"]
                ),
            }
            if algorithm == "proposed":
                pruning = cfg["candidate_pruning"]
                scenario = {
                    **scenario_overrides,
                    "enable_candidate_pruning": bool(
                        pruning["enabled_for_full_te_ppo"]
                    ),
                    "candidate_pruning_radius_m": float(pruning["default_radius_m"]),
                    "candidate_limit_range_m": float(pruning["default_radius_m"]),
                    "diversity_factor": float(pruning["default_diversity_factor"]),
                }
            else:
                scenario = dict(scenario_overrides)
            specs.append(RunSpec(
                algorithm=algorithm,
                train_seed=int(seed),
                run_name_suffix="cpu_robust",
                scenario_overrides_items=tuple(sorted(scenario.items())),
                sweep_param=None,
                purpose="cpu_robustness",
                config_overrides_items=_with_formal_provenance(overrides, cfg),
                experiment_arm=algorithm,
            ))
    return specs


def cpu_robustness_run_dir(
    algorithm: str,
    seed: int,
    config: dict[str, Any] | None = None,
) -> Path:
    cfg = config or load_revision_config()
    artifacts, _ = revision_roots(cfg)
    return artifacts / "cpu_robustness" / f"{algorithm}_s{int(seed)}_cpu_robust"


def load_dbas_calibration(
    config: dict[str, Any] | None = None, *, required: bool = False
) -> dict[str, Any]:
    """Load DBAS's validation-only fixed-price calibration."""
    cfg = config or load_revision_config()
    path = Path(cfg["baseline_calibration"]["dbas_selection_file"])
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"DBAS evaluation requires validation-only price calibration: {path}"
            )
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError(f"invalid DBAS calibration schema version: {path}")
    if "algorithm.fixed_price_fraction" not in payload:
        raise ValueError(f"invalid DBAS calibration schema: {path}")
    fraction = float(payload["algorithm.fixed_price_fraction"])
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"DBAS calibrated price fraction lies outside [0, 1]: {path}")
    if required and int(payload.get("validation_seed_count", 0)) != len(
        cfg["seeds"]["validation"]
    ):
        raise ValueError(
            "formal DBAS calibration must use the complete validation seed bank"
        )
    if required and payload.get("calibration_contract_digest") != dbas_calibration_contract_digest(cfg):
        raise ValueError("DBAS calibration contract is missing or stale; recalibrate")
    return {
        "algorithm.fixed_price_fraction": fraction
    }


def load_cpu_robustness_dbas_calibration(
    config: dict[str, Any] | None = None, *, required: bool = False
) -> dict[str, Any]:
    """Load the one-price, full-CPU-range DBAS calibration."""

    cfg = config or load_revision_config()
    path = Path(cfg["cpu_robustness"]["dbas_selection_file"])
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"CPU robustness DBAS calibration is required: {path}"
            )
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError(f"invalid CPU robustness DBAS calibration: {path}")
    fraction = float(payload.get("algorithm.fixed_price_fraction", -1.0))
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"CPU robustness DBAS price lies outside [0, 1]: {path}")
    if required:
        if int(payload.get("validation_seed_count", 0)) != len(cfg["seeds"]["validation"]):
            raise ValueError("CPU robustness DBAS calibration seed count is invalid")
        if payload.get("validation_grid_digest") != config_digest(
            cfg["cpu_robustness"]["cpu_capacity_ranges_cycles_per_s"]
        ):
            raise ValueError("CPU robustness DBAS calibration grid is stale")
        if payload.get("calibration_contract_digest") != cpu_robustness_contract_digest(
            cfg
        ):
            raise ValueError("CPU robustness DBAS calibration contract is stale")
    return {"algorithm.fixed_price_fraction": fraction}


def fairness_overrides(
    algorithm: str, config: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return the fixed common-core settings for one stochastic method."""

    cfg = config or load_revision_config()
    if algorithm not in STOCHASTIC_ALGORITHMS:
        return {}
    fairness = cfg["fairness"]
    overrides: dict[str, Any] = {
        "algorithm.total_timesteps": int(fairness["total_timesteps"]),
        "algorithm.learning_rate": float(fairness["learning_rate"]),
        "algorithm.clip_range": float(fairness["clip_range"]),
        "algorithm.ent_coef": float(fairness["ent_coef"]),
        "evaluation.interval_env_steps": int(
            cfg["checkpoint_selection"]["evaluation_interval_env_steps"]
        ),
        "evaluation.selection_metric": str(cfg["checkpoint_selection"]["metric"]),
        "evaluation.selection_tie_break": str(
            cfg["checkpoint_selection"]["tie_break"]
        ),
        "evaluation.force_terminal_evaluation": bool(
            cfg["checkpoint_selection"]["force_terminal_evaluation"]
        ),
    }
    hidden = list(fairness["hidden_sizes"])
    if algorithm in CORE_ALGORITHMS[1:]:
        overrides["algorithm.hidden_sizes"] = hidden
    else:
        overrides["policy.actor.hidden_sizes"] = hidden
        overrides["policy.critic.hidden_sizes"] = hidden
    if algorithm == "spis":
        overrides["algorithm.fixed_price"] = float(fairness["spis_fixed_price"])
    if algorithm in tuple(cfg["joint_learning_baselines"]["algorithms"]):
        joint = cfg["joint_learning_baselines"]
        overrides.update({
            "algorithm.implementation_version": str(
                joint["implementation_versions"][algorithm]
            ),
            "algorithm.joint_action_parameterization": str(
                joint["action_parameterization"]
            ),
            "algorithm.rollout_steps": int(joint["rollout_samples_per_update"]),
        })
        if algorithm == "mappo":
            overrides["algorithm.collection_microbatch_size"] = int(
                joint["mappo_collection_microbatch_size"]
            )
        overrides.update({
            f"reward.{key}": value for key, value in joint["reward"].items()
        })
    return overrides


def headline_common_reward_overrides(
    algorithm: str, config: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Require explicit common-reward records only for joint headline runs.

    Exactness label: implementation convenience only. Require explicitly
    recorded common reward for the headline convergence report.
    """

    cfg = config or load_revision_config()
    if algorithm in ("jdls", "mappo"):
        return {
            "evaluation.common_reward_schema": str(
                cfg["common_evaluation_reward"]["schema"]
            )
        }
    return {}


def formal_algorithm_config(
    algorithm: str,
    config: dict[str, Any] | None = None,
    *,
    require_dbas_calibration: bool = True,
) -> dict[str, Any]:
    """Load an algorithm YAML and apply the fixed experiment contract."""
    from src.evaluation.sweep import load_algorithm_config

    cfg = config or load_revision_config()
    materialized = deepcopy(load_algorithm_config(algorithm))
    overrides = {
        **formal_provenance_overrides(cfg),
        **fairness_overrides(algorithm, cfg),
    }
    if algorithm == "dbas":
        overrides.update(load_dbas_calibration(
            cfg, required=require_dbas_calibration
        ))
    for dotted, value in overrides.items():
        node = materialized
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    if algorithm in cfg["common_evaluation_reward"]["plotted_algorithms"]:
        expected_reward = cfg["common_evaluation_reward"]["penalties"]
        actual_reward = materialized.get("reward", {})
        mismatches = {
            key: (actual_reward.get(key), value)
            for key, value in expected_reward.items()
            if actual_reward.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"{algorithm} common evaluation reward penalties differ: "
                f"{mismatches}"
            )
    return materialized


def headline_specs(config: dict[str, Any] | None = None) -> list[RunSpec]:
    """Enumerate the independent default-setting runs used by headline experiments.

    Exactness label: implementation convenience only.  This function freezes
    the statistical unit and artifact identity; it does not change any
    simulator mathematics.
    """
    cfg = config or load_revision_config()
    specs: list[RunSpec] = []
    for algorithm in ALL_ALGORITHMS:
        algorithm_overrides = {
            **fairness_overrides(algorithm, cfg),
            **headline_common_reward_overrides(algorithm, cfg),
        }
        for seed in train_seeds_for(algorithm, config=cfg):
            overrides: dict[str, Any] = {"num_vehicles": 20}
            if algorithm == "proposed":
                pruning = cfg["candidate_pruning"]
                overrides.update({
                    "enable_candidate_pruning": bool(
                        pruning["enabled_for_full_te_ppo"]
                    ),
                    "candidate_pruning_radius_m": float(
                        pruning["default_radius_m"]
                    ),
                    "candidate_limit_range_m": float(
                        pruning["default_radius_m"]
                    ),
                    "diversity_factor": float(
                        pruning["default_diversity_factor"]
                    ),
                })
            specs.append(RunSpec(
                algorithm=algorithm,
                train_seed=seed,
                run_name_suffix="headline",
                scenario_overrides_items=tuple(sorted(overrides.items())),
                sweep_param=None,
                purpose="baseline_line",
                config_overrides_items=_with_formal_provenance(
                    algorithm_overrides, cfg
                ),
            ))
    return specs


def headline_run_dir(algorithm: str, seed: int,
                     config: dict[str, Any] | None = None) -> Path:
    """Return the canonical revision checkpoint directory for one replicate."""
    cfg = config or load_revision_config()
    artifacts_root, _ = revision_roots(cfg)
    return artifacts_root / "headline" / f"{algorithm}_s{int(seed)}_headline"


def pruning_sensitivity_training_cells(
    config: dict[str, Any] | None = None,
) -> tuple[tuple[float, float], ...]:
    """Return distinct ``(radius, trained rho)`` identities for formal candidate-pruning.

    Exactness label: implementation convenience only.  The equivalence classes
    and matched-training contract are maintained in the revision protocol.
    """

    cfg = config or load_revision_config()
    sensitivity = cfg["pruning_sensitivity"]
    return tuple(
        (float(radius), float(rho))
        for radius in sensitivity["radius_grid_m"]
        for rho in sensitivity["trained_diversity_grid"]
    )


def pruning_sensitivity_reported_cells(
    config: dict[str, Any] | None = None,
) -> tuple[tuple[float, float], ...]:
    """Return all cells shown in the formal candidate-pruning table/heatmap."""

    cfg = config or load_revision_config()
    sensitivity = cfg["pruning_sensitivity"]
    return tuple(
        (float(radius), float(rho))
        for radius in sensitivity["radius_grid_m"]
        for rho in sensitivity["reported_diversity_grid"]
    )


def pruning_sensitivity_training_rho(
    reported_rho: float, config: dict[str, Any] | None = None
) -> float:
    """Map a reported rho to its distinct matched-training identity.

    On the maintained short-RoI grid rho 1 and 2 both induce ``N_max=2``;
    rho enters no other model term.  Reusing the rho-2 identity is therefore
    an exact equivalence, not checkpoint replay under changed semantics.
    """

    cfg = config or load_revision_config()
    rho = float(reported_rho)
    reported = tuple(cfg["pruning_sensitivity"]["reported_diversity_grid"])
    if rho not in reported:
        raise ValueError(f"rho={rho:g} is outside the reported sensitivity grid")
    return 2.0 if rho == 1.0 else rho


def pruning_sensitivity_specs(
    config: dict[str, Any] | None = None,
) -> list[RunSpec]:
    """Enumerate non-canonical matched TE-PPO training runs for formal candidate-pruning.

    The canonical ``(80 m, rho=4)`` runs retain their exact headline identity
    and are deliberately omitted here.  All other cells get dedicated
    artifacts, so evaluation cannot silently replay a policy trained under a
    different candidate-selection mechanism.
    """

    cfg = config or load_revision_config()
    sensitivity = cfg["pruning_sensitivity"]
    canonical = (
        float(sensitivity["canonical_radius_m"]),
        float(sensitivity["canonical_diversity_factor"]),
    )
    algorithm_overrides = {
        **fairness_overrides("proposed", cfg),
        "algorithm.total_timesteps": int(sensitivity["total_timesteps"]),
    }
    specs: list[RunSpec] = []
    for radius, rho in pruning_sensitivity_training_cells(cfg):
        if (radius, rho) == canonical:
            continue
        overrides = {
            "num_vehicles": 20,
            **pruning_overrides("pruning_only", radius, rho, config=cfg),
        }
        radius_label = f"{radius:g}".replace(".", "p")
        rho_label = f"{rho:g}".replace(".", "p")
        for seed in sensitivity["train_seeds"]:
            specs.append(RunSpec(
                algorithm="proposed",
                train_seed=int(seed),
                run_name_suffix=f"pruning_r{radius_label}_rho{rho_label}",
                scenario_overrides_items=tuple(sorted(overrides.items())),
                sweep_param=None,
                purpose="pruning_sensitivity",
                config_overrides_items=_with_formal_provenance(
                    algorithm_overrides, cfg
                ),
                experiment_arm=f"r{radius_label}_rho{rho_label}",
            ))
    return specs


def pruning_sensitivity_artifact_root(
    config: dict[str, Any] | None = None,
) -> Path:
    """Return the dedicated root for non-canonical matched candidate-pruning checkpoints."""

    return revision_roots(config)[0] / "pruning_sensitivity" / "matched"


def pruning_sensitivity_run_dir(
    radius_m: float,
    trained_rho: float,
    seed: int,
    config: dict[str, Any] | None = None,
) -> Path:
    """Resolve a matched candidate-pruning run, reusing only the exact canonical headline."""

    cfg = config or load_revision_config()
    sensitivity = cfg["pruning_sensitivity"]
    cell = (float(radius_m), float(trained_rho))
    canonical = (
        float(sensitivity["canonical_radius_m"]),
        float(sensitivity["canonical_diversity_factor"]),
    )
    if cell == canonical:
        return headline_run_dir("proposed", int(seed), cfg)
    matches = [
        spec for spec in pruning_sensitivity_specs(cfg)
        if int(spec.train_seed) == int(seed)
        and float(spec.scenario_overrides["candidate_pruning_radius_m"]) == cell[0]
        and float(spec.scenario_overrides["diversity_factor"]) == cell[1]
    ]
    if len(matches) != 1:
        raise ValueError(
            f"no unique pruning sensitivity identity for r0={cell[0]:g}, "
            f"rho={cell[1]:g}, seed={int(seed)}"
        )
    return matches[0].run_dir(pruning_sensitivity_artifact_root(cfg))


def scalability_specs(
    axis: str,
    config: dict[str, Any] | None = None,
) -> list[RunSpec]:
    """Enumerate shape-changing scalability runs; default cells reuse headline runs."""
    cfg = config or load_revision_config()
    if axis not in ("vehicles", "tasks"):
        raise ValueError("scalability axis must be 'vehicles' or 'tasks'")
    grid = (
        cfg["scalability"]["vehicle_grid"]
        if axis == "vehicles" else cfg["scalability"]["task_grid"]
    )
    default_value = 20 if axis == "vehicles" else 3
    specs: list[RunSpec] = []
    for value in grid:
        if int(value) == default_value:
            continue
        if axis == "vehicles":
            overrides = {"num_vehicles": int(value)}
            suffix = f"scale_v{int(value)}"
            sweep_param = "num_vehicles"
        else:
            overrides = {
                "urgency_profiles": make_urgency_profiles(int(value)),
                "task_region_span_m": 108.0 if int(value) > 4 else None,
            }
            suffix = f"k{int(value)}"
            sweep_param = "num_tasks"
        for algorithm in tuple(cfg["scalability"]["algorithms"]):
            algorithm_overrides = fairness_overrides(algorithm, cfg)
            for seed in train_seeds_for(algorithm, scope="sweep", config=cfg):
                if algorithm == "dbas" and seed != cfg["seeds"]["sweep_train"][0]:
                    continue
                proposed_overrides = dict(overrides)
                if algorithm == "proposed":
                    pruning = cfg["candidate_pruning"]
                    proposed_overrides.update({
                        "enable_candidate_pruning": True,
                        "candidate_pruning_radius_m": float(pruning["default_radius_m"]),
                        "candidate_limit_range_m": float(pruning["default_radius_m"]),
                        "diversity_factor": float(pruning["default_diversity_factor"]),
                    })
                specs.append(RunSpec(
                    algorithm=algorithm,
                    train_seed=int(seed),
                    run_name_suffix=suffix,
                    scenario_overrides_items=tuple(sorted(proposed_overrides.items())),
                    sweep_param=sweep_param,
                    purpose="sweep_metric",
                    config_overrides_items=_with_formal_provenance(
                        algorithm_overrides, cfg
                    ),
                ))
    return specs


def timing_scalability_specs(
    config: dict[str, Any] | None = None,
) -> list[RunSpec]:
    """Enumerate matched TE-PPO checkpoints for the full timing ``V x K`` grid.

    Inference timing is evaluated on one trained replicate per shape.  Random
    networks are deliberately excluded because their action branches can
    change analytical-solver cost and are not evidence about the submitted
    method.
    """

    cfg = config or load_revision_config()
    algorithm_overrides = fairness_overrides("proposed", cfg)
    pruning = cfg["candidate_pruning"]
    seed = int(cfg["seeds"]["sweep_train"][0])
    specs: list[RunSpec] = []
    for vehicles in cfg["scalability"]["timing_vehicle_grid"]:
        for tasks in cfg["scalability"]["timing_task_grid"]:
            if int(vehicles) == 20 and int(tasks) == 3:
                continue
            overrides = {
                "num_vehicles": int(vehicles),
                "urgency_profiles": make_urgency_profiles(int(tasks)),
                "task_region_span_m": 108.0 if int(tasks) > 4 else None,
                "enable_candidate_pruning": True,
                "candidate_pruning_radius_m": float(pruning["default_radius_m"]),
                "candidate_limit_range_m": float(pruning["default_radius_m"]),
                "diversity_factor": float(pruning["default_diversity_factor"]),
            }
            specs.append(RunSpec(
                algorithm="proposed",
                train_seed=seed,
                run_name_suffix=f"timing_v{int(vehicles)}_k{int(tasks)}",
                scenario_overrides_items=tuple(sorted(overrides.items())),
                # This is a two-dimensional checkpoint identity, not a
                # scalar sweep value.  V and K are already explicit in the
                # scenario overrides and suffix; leaving ``sweep_param`` unset
                # avoids pretending the composite cell has one display value.
                sweep_param=None,
                purpose="timing_scalability",
                config_overrides_items=_with_formal_provenance(
                    algorithm_overrides, cfg
                ),
            ))
    return specs


def pruning_overrides(mode: str, radius_m: float, diversity_factor: float,
                      *, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build one matched pruning-sensitivity cell."""
    cfg = config or load_revision_config()
    quality_default = float(cfg["candidate_pruning"]["default_radius_m"])
    if mode not in tuple(cfg["candidate_pruning"]["modes"]):
        raise ValueError(f"unknown pruning sensitivity mode: {mode!r}")
    out = {
        "enable_candidate_pruning": True,
        "candidate_pruning_radius_m": float(radius_m),
        "candidate_limit_range_m": float(radius_m),
        "diversity_factor": float(diversity_factor),
        "sensing_range_m": quality_default,
    }
    return out
