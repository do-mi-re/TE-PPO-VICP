from __future__ import annotations

import pytest

from src.evaluation.ablation import (
    ABLATION_ARMS,
    build_ablation_config,
    validate_controlled_ablation,
)
from src.evaluation.sweep import load_algorithm_config


def _configs():
    base = load_algorithm_config("proposed")
    return {
        arm: build_ablation_config(
            arm,
            123,
            fixed_price=0.07 if arm == "no_dynamic_pricing" else None,
            base_config=base,
        )
        for arm in ABLATION_ARMS
    }


def test_ablation_configs_pass_shared_control_validation():
    validate_controlled_ablation(_configs())


def test_fixed_price_arm_is_calibrated_not_an_arbitrary_fallback():
    config = _configs()["no_dynamic_pricing"]
    assert config["algorithm"]["name"] == "spis"
    assert config["algorithm"]["fixed_price"] == 0.07
    assert config["scenario"]["enable_candidate_pruning"] is True


def test_equal_bandwidth_arm_preserves_other_components():
    configs = _configs()
    full = configs["full"]
    arm = configs["no_bandwidth_allocation"]
    assert arm["algorithm"]["name"] == "ebas"
    assert arm["scenario"] == full["scenario"]
    assert arm["algorithm"]["total_timesteps"] == full["algorithm"]["total_timesteps"]


def test_no_analytical_arm_uses_declared_joint_training_settings():
    configs = _configs()
    full = configs["full"]
    arm = configs["no_analytical_best_response"]
    assert arm["algorithm"]["name"] == "jdls"
    assert arm["algorithm"]["rollout_steps"] == (
        full["algorithm"]["n_envs"] * full["algorithm"]["n_steps"]
    ) == 1024
    assert arm["algorithm"]["minibatch_size"] == full["algorithm"]["batch_size"]
    assert arm["algorithm"]["n_epochs"] == full["algorithm"]["n_epochs"]
    assert arm["normalization"] == {
        "norm_obs": False,
        "norm_reward": False,
    }
    assert arm["evaluation"]["interval_env_steps"] == 3072
    assert full["evaluation"]["interval_env_steps"] == 3072
    assert arm["algorithm"]["implementation_version"] == (
        "jdls_no_br_constraint_aware_v2"
    )
    assert arm["algorithm"]["joint_action_parameterization"] == (
        "candidate_masked_feasible_v1"
    )
    assert {
        key: arm["reward"][key]
        for key in (
            "omega_vehicle", "omega_feasible", "omega_pre_quality", "omega_tau",
            "omega_bandwidth_use", "omega_pre_lat", "pre_lat_overshoot_cap",
            "penalize_missing_latency_as_cap",
        )
    } == {
        "omega_vehicle": 0.001,
        "omega_feasible": 1.0,
        "omega_pre_quality": 0.01,
        "omega_tau": 0.0,
        "omega_bandwidth_use": 0.0,
        "omega_pre_lat": 0.01,
        "pre_lat_overshoot_cap": 10.0,
        "penalize_missing_latency_as_cap": True,
    }


def test_sb3_ablation_arms_preserve_full_normalization():
    configs = _configs()
    full_normalization = configs["full"]["normalization"]
    assert full_normalization["norm_obs"] is True
    assert full_normalization["norm_reward"] is True
    for arm in (
        "no_dynamic_pricing",
        "no_bandwidth_allocation",
        "no_candidate_pruning",
    ):
        assert configs[arm]["normalization"] == full_normalization


def test_no_pruning_only_flips_the_enable_flag():
    configs = _configs()
    full_scenario = dict(configs["full"]["scenario"])
    ablated_scenario = dict(configs["no_candidate_pruning"]["scenario"])
    assert ablated_scenario.pop("enable_candidate_pruning") is False
    assert full_scenario.pop("enable_candidate_pruning") is True
    assert ablated_scenario == full_scenario


def test_control_validator_rejects_scenario_drift():
    configs = _configs()
    configs["no_bandwidth_allocation"]["scenario"]["num_vehicles"] = 21
    with pytest.raises(ValueError, match="uncontrolled scenario"):
        validate_controlled_ablation(configs)


@pytest.mark.parametrize("field", ["norm_obs", "norm_reward"])
def test_control_validator_rejects_jdls_normalization(field):
    configs = _configs()
    configs["no_analytical_best_response"]["normalization"][field] = True
    with pytest.raises(ValueError, match="custom JDLS loop"):
        validate_controlled_ablation(configs)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("rollout_steps", 128, "effective rollout"),
        ("minibatch_size", 128, "minibatch"),
        ("n_epochs", 4, "update epochs"),
    ],
)
def test_control_validator_rejects_optimizer_protocol_drift(field, value, message):
    configs = _configs()
    configs["no_analytical_best_response"]["algorithm"][field] = value
    with pytest.raises(ValueError, match=message):
        validate_controlled_ablation(configs)
