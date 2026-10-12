import unittest

import numpy as np

from src.sim.env import RSULeaderEnv
from src.sim.scenario import ScenarioConfig


def assert_nested_equal(test_case: unittest.TestCase, left: object, right: object) -> None:
    if isinstance(left, dict):
        test_case.assertEqual(set(left.keys()), set(right.keys()))
        for key in left:
            assert_nested_equal(test_case, left[key], right[key])
        return
    if isinstance(left, np.ndarray):
        test_case.assertTrue(np.array_equal(left, right))
        return
    test_case.assertEqual(left, right)


class TestEnvSmoke(unittest.TestCase):
    def make_env(self) -> RSULeaderEnv:
        return RSULeaderEnv(
            scenario_config=ScenarioConfig(
                num_vehicles=4,
            )
        )

    def test_reset_and_step_are_deterministic_for_fixed_seed(self) -> None:
        env_a = self.make_env()
        env_b = self.make_env()
        action = {
            "prices": np.array([0.5, 0.4, 0.3], dtype=np.float64),
            "bandwidth": np.array([0.25, 0.25, 0.25, 0.0], dtype=np.float64),
        }

        obs_a, info_a = env_a.reset(seed=123)
        obs_b, info_b = env_b.reset(seed=123)
        assert_nested_equal(self, obs_a, obs_b)
        assert_nested_equal(self, info_a, info_b)

        step_obs_a, reward_a, terminated_a, truncated_a, step_info_a = env_a.step(action)
        step_obs_b, reward_b, terminated_b, truncated_b, step_info_b = env_b.step(action)

        assert_nested_equal(self, step_obs_a, step_obs_b)
        self.assertEqual(reward_a, reward_b)
        self.assertEqual(terminated_a, terminated_b)
        self.assertEqual(truncated_a, truncated_b)
        self.assertTrue(np.array_equal(step_info_a["selected_tasks"], step_info_b["selected_tasks"]))
        self.assertTrue(
            np.array_equal(
                step_info_a["follower_sensing_durations_s"],
                step_info_b["follower_sensing_durations_s"],
            )
        )
        self.assertTrue(
            np.array_equal(
                step_info_a["follower_compression_ratios"],
                step_info_b["follower_compression_ratios"],
            )
        )

    def test_step_returns_required_diagnostics(self) -> None:
        env = self.make_env()
        env.reset(seed=7)
        _, reward, terminated, truncated, info = env.step(
            {
                "prices": np.array([0.6, 0.4, 0.2], dtype=np.float64),
                "bandwidth": np.array([0.4, 0.2, 0.2, 0.1], dtype=np.float64),
            }
        )

        self.assertTrue(np.isfinite(reward))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertIn("selected_tasks", info)
        self.assertIn("follower_sensing_durations_s", info)
        self.assertIn("follower_compression_ratios", info)
        self.assertIn("energy_components", info)
        self.assertIn("utility_components", info)
        self.assertIn("violated_constraints", info)
        self.assertEqual(info["selected_tasks"].shape, (4,))
        self.assertEqual(info["follower_sensing_durations_s"].shape, (4,))
        self.assertEqual(info["follower_compression_ratios"].shape, (4,))

    def test_invalid_bandwidth_action_raises(self) -> None:
        env = self.make_env()
        env.reset(seed=11)
        with self.assertRaises(ValueError):
            env.step(
                {
                    "prices": np.array([0.5, 0.4, 0.3], dtype=np.float64),
                    "bandwidth": np.array([0.5, 0.4, 0.3, 0.0], dtype=np.float64),
                }
            )


if __name__ == "__main__":
    unittest.main()
