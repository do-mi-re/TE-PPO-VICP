"""Regression tests for RSULeaderEnv.step_joint follower-IR enforcement."""

import unittest
from unittest.mock import patch

import numpy as np

from src.baselines.joint_learning_reward import (
    JointLearningShapingConfig,
    compute_joint_learning_shaping,
)
from src.sim.env import JointAction, RSULeaderEnv
from src.sim.scenario import ScenarioConfig


class TestStepJointIndividualRationality(unittest.TestCase):
    """``step_joint`` must drop assignments whose follower utility is non-positive."""

    def _make_env(self) -> RSULeaderEnv:
        return RSULeaderEnv(scenario_config=ScenarioConfig(num_vehicles=3))

    def _baseline_joint(self, env: RSULeaderEnv) -> JointAction:
        K = env.scenario_config.num_tasks
        V = env.scenario_config.num_vehicles
        return JointAction(
            prices=tuple([env.scenario_config.pricing_budget_max] * K),
            bandwidth=tuple([1.0 / V] * V),
            tau_s=tuple([0.05] * V),
            eta=tuple([2.0] * V),
            assignment=tuple([0] * V),
        )

    def test_negative_utility_assignment_is_dropped(self) -> None:
        env = self._make_env()
        env.reset(seed=321)
        # Force every vehicle to participate at price 0 (payment = 0, energy > 0
        # → strictly negative conditional utility), assignment to task 0.
        K = env.scenario_config.num_tasks
        V = env.scenario_config.num_vehicles
        joint = JointAction(
            prices=tuple([0.0] * K),
            bandwidth=tuple([1.0 / V] * V),
            tau_s=tuple([0.05] * V),
            eta=tuple([2.0] * V),
            assignment=tuple([0] * V),
        )
        _, _, _, _, info = env.step_joint(joint)
        # Every vehicle's IR constraint is violated -> nobody participates.
        selected_tasks = np.asarray(info["selected_tasks"], dtype=np.int64)
        self.assertTrue(np.all(selected_tasks == -1))
        self.assertEqual(float(info["total_vehicle_utility"]), 0.0)
        self.assertTrue(np.all(np.asarray(info["vehicle_utilities"]) == 0.0))

    def test_positive_utility_assignment_is_honored(self) -> None:
        env = self._make_env()
        env.reset(seed=321)
        joint = self._baseline_joint(env)
        _, _, _, _, info = env.step_joint(joint)
        selected_tasks = np.asarray(info["selected_tasks"], dtype=np.int64)
        vehicle_utilities = np.asarray(info["vehicle_utilities"], dtype=np.float64)
        # At least one vehicle should accept and all accepted utilities are > 0.
        self.assertTrue(np.any(selected_tasks == 0))
        for vehicle_idx, task_id in enumerate(selected_tasks):
            if task_id == -1:
                self.assertEqual(vehicle_utilities[vehicle_idx], 0.0)
            else:
                self.assertGreater(vehicle_utilities[vehicle_idx], 0.0)

    def test_step_joint_never_calls_analytical_best_response(self) -> None:
        env = self._make_env()
        env.reset(seed=321)
        with patch(
            "src.sim.env.solve_vehicle_best_response",
            side_effect=AssertionError("analytical solver leaked into joint path"),
        ):
            env.step_joint(self._baseline_joint(env))

    def test_training_info_mode_preserves_reward_and_shaping(self) -> None:
        full_env = self._make_env()
        training_env = self._make_env()
        full_env.reset(seed=321)
        training_env.reset(seed=321)
        joint = self._baseline_joint(full_env)

        _, full_reward, _, _, full_info = full_env.step_joint(joint)
        _, training_reward, _, _, training_info = training_env.step_joint(
            joint, info_mode="training"
        )

        self.assertEqual(full_reward, training_reward)
        for full_response, training_response in zip(
            full_info["follower_responses"],
            training_info["follower_responses"],
            strict=True,
        ):
            self.assertEqual(
                full_response.selected_task_id,
                training_response.selected_task_id,
            )
            self.assertEqual(
                full_response.selected_utility,
                training_response.selected_utility,
            )
            self.assertLessEqual(
                len(training_response.task_evaluations),
                len(full_response.task_evaluations),
            )
        self.assertEqual(full_info["social_welfare"], training_info["social_welfare"])
        np.testing.assert_array_equal(
            full_info["utility_components"]["effective_task_qualities"],
            training_info["utility_components"]["effective_task_qualities"],
        )
        cfg = JointLearningShapingConfig(
            omega_pre_quality=0.01,
            omega_tau=0.1,
            omega_bandwidth_use=1.0,
        )
        weights = tuple(task.beta for task in full_env.require_state().tasks)
        deadlines = tuple(task.tau_max_s for task in full_env.require_state().tasks)
        full_shaped = compute_joint_learning_shaping(
            full_reward,
            full_info,
            cfg=cfg,
            task_weights=weights,
            task_deadlines=deadlines,
        )
        training_shaped = compute_joint_learning_shaping(
            training_reward,
            training_info,
            cfg=cfg,
            task_weights=weights,
            task_deadlines=deadlines,
        )
        self.assertEqual(full_shaped, training_shaped)
        self.assertIn("observation_snapshot", full_info)
        self.assertNotIn("observation_snapshot", training_info)
        self.assertNotIn("violated_constraints", training_info)
        self.assertNotIn("prices", training_info["joint_action"])
        self.assertNotIn("eta", training_info["joint_action"])
        self.assertNotIn("vehicle_utilities", training_info)

    def test_step_joint_rejects_unknown_info_mode(self) -> None:
        env = self._make_env()
        env.reset(seed=321)
        with self.assertRaisesRegex(ValueError, "info_mode"):
            env.step_joint(self._baseline_joint(env), info_mode="compact")


if __name__ == "__main__":
    unittest.main()
