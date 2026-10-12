from __future__ import annotations

import run as runmod
from src.evaluation.revision import (
    headline_specs,
    load_revision_config,
)
from src.evaluation.tasks import RunSpec


def test_headline_specs_apply_approved_replication_policy():
    specs = headline_specs()
    counts: dict[str, int] = {}
    for spec in specs:
        counts[spec.algorithm] = counts.get(spec.algorithm, 0) + 1
    assert counts == {
        "proposed": 8,
        "jdls": 8,
        "mappo": 8,
        "dbas": 1,
        "ebas": 8,
        "spis": 8,
        "rdus": 8,
    }


def test_headline_te_ppo_is_the_canonical_pruning_enabled_full_model():
    cfg = load_revision_config()
    proposed = next(spec for spec in headline_specs(cfg) if spec.algorithm == "proposed")
    override = proposed.scenario_overrides
    assert override["enable_candidate_pruning"] is True
    assert override["candidate_pruning_radius_m"] == 80.0
    assert override["candidate_limit_range_m"] == 80.0
    assert override["diversity_factor"] == 4.0


def test_other_headline_algorithms_do_not_silently_gain_te_pruning():
    specs = [spec for spec in headline_specs() if spec.algorithm != "proposed"]
    assert all("enable_candidate_pruning" not in spec.scenario_overrides for spec in specs)


def test_every_stochastic_headline_method_has_equal_600k_step_budget():
    specs = [spec for spec in headline_specs() if spec.algorithm != "dbas"]
    assert all(
        spec.config_overrides["algorithm.total_timesteps"] == 600000
        for spec in specs
    )


def test_joint_baselines_share_the_formal_rollout_budget():
    cfg = load_revision_config()
    headline = [
        spec for spec in headline_specs(cfg) if spec.algorithm in ("jdls", "mappo")
    ]
    assert all(
        spec.config_overrides["algorithm.rollout_steps"] == 1024
        for spec in headline
    )


def test_dotted_config_overrides_are_materialized_into_training_config():
    spec = RunSpec(
        algorithm="proposed",
        train_seed=123,
        run_name_suffix="override_test",
        scenario_overrides_items=(),
        sweep_param=None,
        purpose="baseline_line",
        config_overrides_items=(("algorithm.learning_rate", 0.0003),),
    )
    config = runmod._materialize_config(spec.to_worker_dict(), (30000,))
    assert config["algorithm"]["learning_rate"] == 0.0003


def test_revision_server_worker_limits_separate_throughput_from_timing():
    cfg = load_revision_config()
    assert cfg["execution"]["max_workers"] == 32
    assert cfg["execution"]["isolated_timing_max_workers"] == 1
