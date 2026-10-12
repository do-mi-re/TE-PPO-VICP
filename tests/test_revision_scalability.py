from __future__ import annotations

from src.evaluation.revision import scalability_specs


def test_vehicle_scalability_retrains_every_nondefault_shape_with_five_seeds():
    specs = scalability_specs("vehicles")
    proposed = [spec for spec in specs if spec.algorithm == "proposed"]
    assert len(proposed) == 7 * 5
    assert {spec.display_value for spec in proposed} == {
        10.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0
    }


def test_task_scalability_includes_default_omitted_from_retraining_only():
    specs = scalability_specs("tasks")
    proposed = [spec for spec in specs if spec.algorithm == "proposed"]
    assert len(proposed) == 3 * 5
    assert {spec.display_value for spec in proposed} == {6.0, 9.0, 12.0}


def test_scalability_never_attempts_checkpoint_replay_across_shape_change():
    for axis in ("vehicles", "tasks"):
        for spec in scalability_specs(axis):
            assert (
                spec.run_name_suffix.startswith("scale_v")
                if axis == "vehicles" else spec.run_name_suffix.startswith("k")
            )
            assert spec.sweep_param in ("num_vehicles", "num_tasks")


def test_joint_scalability_baselines_do_not_inherit_candidate_pruning():
    for axis in ("vehicles", "tasks"):
        joint_specs = [
            spec
            for spec in scalability_specs(axis)
            if spec.algorithm in ("jdls", "mappo")
        ]
        assert joint_specs
        assert all(
            "enable_candidate_pruning" not in spec.scenario_overrides
            for spec in joint_specs
        )
        assert all(
            spec.config_overrides["algorithm.joint_action_parameterization"]
            == "candidate_masked_feasible_v1"
            for spec in joint_specs
        )
