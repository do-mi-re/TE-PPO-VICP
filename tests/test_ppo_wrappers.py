import unittest
from unittest.mock import patch

import numpy as np
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

from src.rl.policy import decode_bandwidth, decode_leader_action
from src.rl.ppo import (
    NormalizationConfig,
    RewardConfig,
    TEPPORewardWrapper,
    make_training_env,
    make_vectorized_training_env,
)
from src.sim.env import RSULeaderEnv
from src.sim.scenario import ScenarioConfig


class TestPPOWrappers(unittest.TestCase):
    def make_scenario_config(self) -> ScenarioConfig:
        return ScenarioConfig(num_vehicles=4)

    def test_decode_leader_action_respects_leader_constraints(self) -> None:
        action = decode_leader_action(
            np.zeros(8, dtype=np.float64),
            num_tasks=3,
            num_vehicles=4,
            pricing_budget_max=2.5,
        )

        self.assertEqual(len(action.prices), 3)
        self.assertEqual(len(action.bandwidth), 4)
        for price in action.prices:
            self.assertGreaterEqual(price, 0.0)
            self.assertLessEqual(price, 2.5)
        for weight in action.bandwidth:
            self.assertGreaterEqual(weight, 0.0)
        self.assertLessEqual(sum(action.bandwidth), 1.0 + 1e-12)

    def test_masked_bandwidth_is_zero_outside_candidate_union(self) -> None:
        logits = np.array([1000.0, -1000.0, 4.0, 3.0], dtype=np.float64)
        mask = np.array([0, 1, 1, 0], dtype=np.int8)

        bandwidth = decode_bandwidth(
            0.0, logits, eligible_vehicle_mask=mask
        )

        self.assertEqual(bandwidth[0], 0.0)
        self.assertEqual(bandwidth[3], 0.0)
        self.assertAlmostEqual(float(np.sum(bandwidth)), 0.5)
        self.assertTrue(np.all(np.isfinite(bandwidth)))

    def test_empty_candidate_union_receives_zero_bandwidth(self) -> None:
        bandwidth = decode_bandwidth(
            8.0,
            np.array([1.0, 2.0]),
            eligible_vehicle_mask=np.zeros(2, dtype=bool),
        )

        np.testing.assert_array_equal(bandwidth, np.zeros(2))

    def test_omitted_candidate_mask_preserves_original_softmax(self) -> None:
        logits = np.array([-2.0, 0.0, 1.0], dtype=np.float64)
        unmasked = decode_bandwidth(0.7, logits)
        explicit_all = decode_bandwidth(
            0.7, logits, eligible_vehicle_mask=np.ones(3, dtype=bool)
        )

        np.testing.assert_array_equal(unmasked, explicit_all)

    def test_flattened_training_env_returns_sb3_friendly_observation(self) -> None:
        env = make_training_env(
            scenario_config=self.make_scenario_config(),
            reward_config=RewardConfig(),
            seed=123,
        )

        observation, _ = env.reset(seed=123)

        self.assertIn("task_beta", observation)
        self.assertIn("vehicle_energy_weight", observation)
        self.assertIn("geometry_distance_to_roi_m", observation)
        self.assertEqual(observation["channels"].dtype, np.float32)
        self.assertEqual(observation["task_beta"].shape, (3,))
        self.assertEqual(observation["geometry_distance_to_roi_m"].shape, (12,))
        env.close()

    def test_reward_wrapper_matches_penalty_formula(self) -> None:
        base_env = RSULeaderEnv(scenario_config=self.make_scenario_config())
        reward_config = RewardConfig()
        env = TEPPORewardWrapper(base_env, reward_config=reward_config)
        env.reset(seed=321)
        action = {
            "prices": np.array([0.05, 0.04, 0.03], dtype=np.float64),
            "bandwidth": np.array([0.3, 0.2, 0.1, 0.0], dtype=np.float64),
        }

        _, wrapped_reward, _, _, info = env.step(action)

        betas = [task.beta for task in base_env.require_state().tasks]
        effective_quality = np.asarray(info["utility_components"]["effective_task_qualities"], dtype=np.float64)
        participant_counts = np.asarray(info["utility_components"]["participant_counts"], dtype=np.float64)
        expected_fail = sum(
            reward_config.omega_fail * beta
            for beta, quality in zip(betas, effective_quality, strict=True)
            if quality <= 0.0
        )
        expected_part = sum(
            reward_config.omega_part * beta * max(1.0 - participants, 0.0)
            for beta, participants in zip(betas, participant_counts, strict=True)
        )
        expected_lat = 0.0
        for response in info["follower_responses"]:
            if response.selected_task_id is None:
                continue
            evaluation = next(
                task_eval
                for task_eval in response.task_evaluations
                if task_eval.task_id == response.selected_task_id
            )
            deadline = base_env.require_state().tasks[evaluation.task_id].tau_max_s
            overshoot = max((evaluation.total_latency_s - deadline) / deadline, 0.0)
            expected_lat += reward_config.omega_lat * betas[evaluation.task_id] * overshoot

        expected_reward = (
            info["raw_rsu_reward"]
            - expected_fail
            - expected_part
            - expected_lat
        )
        self.assertAlmostEqual(info["penalty_fail"], expected_fail)
        self.assertAlmostEqual(info["penalty_part"], expected_part)
        self.assertAlmostEqual(info["penalty_lat"], expected_lat)
        self.assertAlmostEqual(info["te_ppo_reward"], expected_reward)
        self.assertAlmostEqual(wrapped_reward, expected_reward)
        env.close()

    def test_wrapped_env_passes_sb3_env_checker(self) -> None:
        env = make_training_env(
            scenario_config=self.make_scenario_config(),
            reward_config=RewardConfig(),
            seed=11,
        )
        check_env(env, warn=True)
        env.close()

    def test_vectorized_training_env_does_not_force_start_method(self) -> None:
        captured: dict[str, object] = {}

        def fake_subproc(env_fns, *args, **kwargs):
            captured["kwargs"] = kwargs
            return DummyVecEnv(env_fns)

        with patch("src.rl.ppo.SubprocVecEnv", side_effect=fake_subproc):
            env = make_vectorized_training_env(
                scenario_config=self.make_scenario_config(),
                reward_config=RewardConfig(),
                n_envs=2,
                normalization_config=NormalizationConfig(),
            )

        self.assertNotIn("start_method", captured["kwargs"])
        env.close()

    def test_vectorized_training_env_falls_back_to_dummy_on_subproc_startup_error(self) -> None:
        with patch("src.rl.ppo.SubprocVecEnv", side_effect=PermissionError("forkserver blocked")):
            env = make_vectorized_training_env(
                scenario_config=self.make_scenario_config(),
                reward_config=RewardConfig(),
                n_envs=2,
                normalization_config=NormalizationConfig(),
            )

        self.assertIsInstance(env.venv.venv, DummyVecEnv)
        env.close()


if __name__ == "__main__":
    unittest.main()
