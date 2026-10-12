import copy
from pathlib import Path

import pytest
import yaml

import run as runmod
from src.evaluation import orchestration as orch
from src.evaluation.provenance import config_digest
from src.evaluation.revision import (
    ALL_ALGORITHMS,
    STOCHASTIC_ALGORITHMS,
    cpu_robustness_contract_digest,
    cpu_robustness_specs,
    fairness_overrides,
    formal_algorithm_config,
    headline_specs,
    load_revision_config,
    pruning_overrides,
    pruning_sensitivity_reported_cells,
    pruning_sensitivity_specs,
    pruning_sensitivity_training_cells,
    pruning_sensitivity_training_rho,
    revision_roots,
    scalability_specs,
    timing_scalability_specs,
    train_seeds_for,
)
from src.evaluation.sweep import load_algorithm_config
from src.rl.policy import price_logit_for_fraction, stable_sigmoid


def test_revision_seed_banks_and_roots_are_explicit_and_disjoint():
    config = load_revision_config()
    assert config["schema_version"] == 13
    assert config["training_semantics_version"] == 4
    assert config["common_evaluation_reward"]["schema"] == (
        "common_evaluation_reward_v1"
    )
    assert config["common_evaluation_reward"]["penalties"] == {
        "omega_fail": 2.0,
        "omega_lat": 3.0,
        "omega_part": 3.0,
        "min_participants": 1,
    }
    assert len(train_seeds_for("proposed", config=config)) == 8
    assert len(train_seeds_for("mappo", config=config)) == 8
    assert len(train_seeds_for("spis", config=config)) == 8
    assert len(train_seeds_for("proposed", scope="sweep", config=config)) == 5
    assert len(train_seeds_for("proposed", scope="ablation", config=config)) == 5
    assert config["pruning_sensitivity"]["train_seeds"] == config["seeds"][
        "headline_train"
    ]
    assert set(config["seeds"]["validation"]).isdisjoint(config["seeds"]["test"])
    assert set(config["seeds"]["pruning_test"]).isdisjoint(
        set(config["seeds"]["validation"]) | set(config["seeds"]["test"])
    )
    evaluation_banks = (
        "validation", "test", "pruning_test", "scalability_test", "dynamic_test",
        "main_text_test"
    )
    for index, left in enumerate(evaluation_banks):
        for right in evaluation_banks[index + 1:]:
            assert set(config["seeds"][left]).isdisjoint(config["seeds"][right])
    assert len(config["seeds"]["scalability_test"]) == 32
    assert len(config["seeds"]["dynamic_test"]) == 32
    assert len(config["seeds"]["main_text_test"]) == 32
    assert config["seeds"]["scalability_test"] == tuple(range(60_000, 60_032))
    assert config["seeds"]["dynamic_test"] == tuple(range(70_000, 70_032))
    assert config["seeds"]["main_text_test"] == tuple(range(80_000, 80_032))
    main_text = config["main_text_analysis"]
    assert main_text["contract_version"] == 2
    assert main_text["evaluation_seed_bank"] == "main_text_test"
    assert main_text["train_seed_bank"] == "headline_train"
    assert len(main_text["cpu_capacity_ranges_cycles_per_s"]) == 5
    assert len(main_text["compression_ratio_grid"]) == 10
    assert len(main_text["price_scale_grid"]) == 7
    cpu = config["cpu_robustness"]
    assert cpu["protocol_version"] == 1
    assert cpu["training_seed_bank"] == "headline_train"
    assert cpu["validation_seed_bank"] == "validation"
    assert cpu["evaluation_seed_bank"] == "main_text_test"
    assert cpu["cpu_capacity_ranges_cycles_per_s"] == main_text[
        "cpu_capacity_ranges_cycles_per_s"
    ]
    specs = cpu_robustness_specs(config)
    assert len(specs) == len(STOCHASTIC_ALGORITHMS) * 8 + 1
    assert {spec.algorithm for spec in specs} == set(ALL_ALGORITHMS)
    assert all(spec.purpose == "cpu_robustness" for spec in specs)
    assert len(cpu_robustness_contract_digest(config)) == 64
    artifacts, outputs = revision_roots(config)
    assert artifacts == Path("artifacts/revision_2026_v5")
    assert outputs == Path("outputs/revision_2026_v5")

def test_pruning_controls_preserve_physical_sensing_range():
    isolated = pruning_overrides("pruning_only", 60.0, 1.5)
    assert isolated["candidate_pruning_radius_m"] == 60.0
    assert isolated["candidate_limit_range_m"] == 60.0
    assert isolated["sensing_range_m"] == 80.0
    assert isolated["diversity_factor"] == 1.5


def test_formal_algorithm_configs_match_maintained_manuscript_semantics():
    for algorithm in ALL_ALGORITHMS:
        config = load_algorithm_config(algorithm)
        scenario = config["scenario"]
        assert scenario["pricing_budget_max"] == 0.3
        assert scenario["roi_distance_model"] == "rectangle"
        assert scenario["distance_floor_m"] == 1.0
        assert scenario["spatial_potential_max"] == 50.0
        if algorithm != "dbas":
            reward = config["reward"]
            assert (reward["omega_fail"], reward["omega_lat"], reward["omega_part"]) == (
                2.0, 3.0, 3.0
            )


def test_joint_headline_configs_require_explicit_common_reward_schema():
    revision = load_revision_config()
    specs = headline_specs(revision)
    for algorithm in ("jdls", "mappo"):
        matching = [spec for spec in specs if spec.algorithm == algorithm]
        assert matching
        assert all(
            spec.config_overrides["evaluation.common_reward_schema"]
            == "common_evaluation_reward_v1"
            for spec in matching
        )
        materialized = orch.materialize_config(
            matching[0].to_worker_dict(), revision["seeds"]["validation"]
        )
        assert materialized["evaluation"]["common_reward_schema"] == (
            "common_evaluation_reward_v1"
        )
    for algorithm in ("proposed", "ebas"):
        matching = [spec for spec in specs if spec.algorithm == algorithm]
        assert matching
        assert all(
            "evaluation.common_reward_schema" not in spec.config_overrides
            for spec in matching
        )
    for algorithm in ("jdls", "mappo"):
        assert "evaluation.common_reward_schema" not in fairness_overrides(
            algorithm, revision
        )
        assert "common_reward_schema" not in formal_algorithm_config(
            algorithm, revision
        )["evaluation"]


def test_common_reward_marker_does_not_invalidate_nonheadline_training():
    revision = load_revision_config()
    nonheadline = [
        *cpu_robustness_specs(revision),
        *scalability_specs("vehicles", revision),
        *scalability_specs("tasks", revision),
    ]
    joint_specs = [
        spec for spec in nonheadline if spec.algorithm in ("jdls", "mappo")
    ]
    assert joint_specs
    assert all(
        "evaluation.common_reward_schema" not in spec.config_overrides
        for spec in joint_specs
    )


def test_formal_base_config_does_not_require_headline_reward_recording():
    for algorithm in ("jdls", "mappo", "proposed", "ebas"):
        config = formal_algorithm_config(algorithm)
        assert "common_reward_schema" not in config["evaluation"]


def test_reward_curve_algorithms_share_the_common_penalty_contract():
    revision = load_revision_config()
    assert revision["common_evaluation_reward"]["figure_algorithms"] == [
        "proposed", "jdls", "mappo"
    ]
    expected = revision["common_evaluation_reward"]["penalties"]
    for algorithm in revision["common_evaluation_reward"]["plotted_algorithms"]:
        config = formal_algorithm_config(algorithm)
        assert {
            key: config["reward"][key]
            for key in expected
        } == expected


def test_sb3_server_layout_preserves_rollout_and_unifies_validation_cadence():
    revision = load_revision_config()
    effective_slots = revision["execution"]["sb3_cpu_slots_per_task"]
    assert effective_slots == 4
    for algorithm in ("proposed", "rdus", "ebas", "spis"):
        config = load_algorithm_config(algorithm)
        algo = config["algorithm"]
        n_envs = int(algo["n_envs"])
        assert (n_envs, int(algo["n_steps"])) == (8, 128)
        assert n_envs * int(algo["n_steps"]) == 1024
        assert int(config["evaluation"]["interval_env_steps"]) == 3072
        assert n_envs * int(config["artifacts"]["checkpoint_freq"]) == 65536
        assert runmod._training_task_cpu_slots(config) == 9
        assert runmod._training_task_cpu_slots(
            config, sb3_cpu_slots=effective_slots
        ) == 4

    assert runmod._training_task_cpu_slots(load_algorithm_config("jdls")) == 1
    assert runmod._training_task_cpu_slots(load_algorithm_config("mappo")) == 1


def test_common_price_initialization_maps_to_half_cap():
    fraction = 0.5
    logit = price_logit_for_fraction(fraction)
    assert stable_sigmoid(logit) == pytest.approx(fraction)
    assert 0.3 * stable_sigmoid(logit) == pytest.approx(0.15)


def test_revision_diversity_grid_exercises_nonconstant_candidate_caps():
    from src.sim.scenario import candidate_limit

    config = load_revision_config()
    caps = {
        candidate_limit(35.0, 80.0, rho)
        for rho in config["candidate_pruning"]["diversity_grid"]
    }
    assert caps == {2, 3, 4}


def test_matched_pruning_contract_deduplicates_only_exact_rho_equivalence():
    config = load_revision_config()
    assert len(pruning_sensitivity_reported_cells(config)) == 16
    assert len(pruning_sensitivity_training_cells(config)) == 12
    assert pruning_sensitivity_training_rho(1.0, config) == 2.0
    assert pruning_sensitivity_training_rho(2.0, config) == 2.0
    assert pruning_sensitivity_training_rho(3.0, config) == 3.0
    assert len(pruning_sensitivity_specs(config)) == 88


def test_revision_config_rejects_partial_pruning_training_seed_set(tmp_path):
    path = tmp_path / "revision.yaml"
    payload = yaml.safe_load(Path("configs/revision.yaml").read_text())
    payload["pruning_sensitivity"]["train_seeds"] = [123, 231, 312]
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="common headline train seeds"):
        load_revision_config(path)


def test_revision_config_rejects_overlapping_validation_and_test(tmp_path):
    path = tmp_path / "revision.yaml"
    payload = yaml.safe_load(Path("configs/revision.yaml").read_text())
    payload["seeds"]["test"][0] = payload["seeds"]["validation"][0]
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="must be disjoint"):
        load_revision_config(path)


def test_revision_config_rejects_reused_pruning_test_bank(tmp_path):
    path = tmp_path / "revision.yaml"
    payload = yaml.safe_load(Path("configs/revision.yaml").read_text())
    payload["seeds"]["pruning_test"][0] = payload["seeds"]["test"][0]
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="must be disjoint"):
        load_revision_config(path)


def test_revision_config_rejects_reused_scalability_test_bank(tmp_path):
    path = tmp_path / "revision.yaml"
    payload = yaml.safe_load(Path("configs/revision.yaml").read_text())
    payload["seeds"]["scalability_test"][0] = payload["seeds"]["pruning_test"][0]
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="must be disjoint"):
        load_revision_config(path)


def test_revision_config_rejects_unsupported_schema(tmp_path):
    path = tmp_path / "revision.yaml"
    text = Path("configs/revision.yaml").read_text(encoding="utf-8")
    path.write_text(text.replace("schema_version: 13", "schema_version: 12"), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version must be 13"):
        load_revision_config(path)


def test_formal_joint_learning_contract_is_cold_start_and_constraint_aware():
    cfg = load_revision_config()
    contract = cfg["joint_learning_baselines"]

    for algorithm in ("jdls", "mappo"):
        materialized = formal_algorithm_config(algorithm, cfg)
        assert materialized["algorithm"]["rollout_steps"] == 1024
        assert materialized["algorithm"]["joint_action_parameterization"] == (
            "candidate_masked_feasible_v1"
        )
        assert materialized["algorithm"]["implementation_version"] == (
            contract["implementation_versions"][algorithm]
        )
    assert formal_algorithm_config("mappo", cfg)["algorithm"][
        "collection_microbatch_size"
    ] == 32


@pytest.mark.parametrize("slots", (0, 10))
def test_revision_config_rejects_invalid_sb3_cpu_slots(tmp_path, slots):
    path = tmp_path / "revision.yaml"
    text = Path("configs/revision.yaml").read_text(encoding="utf-8")
    path.write_text(
        text.replace("sb3_cpu_slots_per_task: 4", f"sb3_cpu_slots_per_task: {slots}"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sb3_cpu_slots_per_task must lie"):
        load_revision_config(path)


@pytest.mark.parametrize("slots", (0, -1, 10))
def test_sb3_cpu_slot_override_rejects_invalid_reservations(slots):
    config = load_algorithm_config("proposed")

    with pytest.raises(ValueError, match="sb3_cpu_slots must lie between"):
        runmod._training_task_cpu_slots(config, sb3_cpu_slots=slots)


def test_timing_grid_specs_materialize_without_fake_scalar_display_value():
    specs = timing_scalability_specs(load_revision_config())

    assert len(specs) == 23
    for spec in specs:
        worker = spec.to_worker_dict()
        assert worker["sweep_param"] is None
        assert worker["display_value"] is None
        assert "num_vehicles" in worker["scenario_overrides"]
        assert "urgency_profiles" in worker["scenario_overrides"]


def test_scalability_seed_extension_only_adds_run_identities():
    config = load_revision_config()
    assert len(scalability_specs("vehicles", config)) == 217
    assert len(scalability_specs("tasks", config)) == 93

    prior = copy.deepcopy(config)
    prior["seeds"]["sweep_train"] = (123, 231, 312)
    validation_seeds = tuple(config["seeds"]["validation"])
    for axis in ("vehicles", "tasks"):
        before_specs = {
            (spec.algorithm, spec.train_seed, spec.run_name_suffix): spec
            for spec in scalability_specs(axis, prior)
            if spec.algorithm != "dbas"
        }
        after_specs = {
            (spec.algorithm, spec.train_seed, spec.run_name_suffix): spec
            for spec in scalability_specs(axis, config)
            if spec.algorithm != "dbas"
        }
        assert before_specs.keys() < after_specs.keys()
        for identity, before in before_specs.items():
            after = after_specs[identity]
            before_config = orch.materialize_config(
                before.to_worker_dict(), validation_seeds
            )
            after_config = orch.materialize_config(
                after.to_worker_dict(), validation_seeds
            )
            assert config_digest(before_config) == config_digest(after_config)
