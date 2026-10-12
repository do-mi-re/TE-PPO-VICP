import unittest

import numpy as np

from src.examples.single_step_explain import (
    DEMO_SEED,
    make_demo_action,
    make_demo_config,
    make_demo_scenario,
)
from src.models.utilities import bandwidth_cost, task_value
from src.rl.policy import decode_leader_action, latent_action_dim
from src.rl.ppo import (
    FlattenLeaderObservationWrapper,
    LeaderLatentActionWrapper,
    RewardConfig,
    TEPPORewardWrapper,
)
from src.sim.env import RSULeaderEnv


def make_demo_env() -> RSULeaderEnv:
    config = make_demo_config()
    state = make_demo_scenario()
    return RSULeaderEnv(
        scenario_config=config,
        scenario_factory=lambda rng, scenario_config, fixed_state=state: fixed_state,
    )


def assert_nested_equal(test_case: unittest.TestCase, left: object, right: object) -> None:
    if isinstance(left, dict):
        test_case.assertEqual(set(left.keys()), set(right.keys()))
        for key in left:
            assert_nested_equal(test_case, left[key], right[key])
        return
    if isinstance(left, np.ndarray):
        test_case.assertTrue(np.array_equal(left, right))
        return
    if isinstance(left, tuple):
        test_case.assertEqual(left, right)
        return
    test_case.assertEqual(left, right)


class TestEnvTrainerMechanics(unittest.TestCase):
    def test_fixed_scenario_rollout_is_reproducible(self) -> None:
        env_a = make_demo_env()
        env_b = make_demo_env()
        config = make_demo_config()
        latent = np.linspace(
            -0.4,
            0.4,
            latent_action_dim(config.num_tasks, config.num_vehicles),
            dtype=np.float32,
        )
        direct_action = decode_leader_action(
            latent.astype(np.float64),
            num_tasks=config.num_tasks,
            num_vehicles=config.num_vehicles,
            pricing_budget_max=config.pricing_budget_max,
        )
        action = {
            "prices": np.asarray(direct_action.prices, dtype=np.float64),
            "bandwidth": np.asarray(direct_action.bandwidth, dtype=np.float64),
        }

        obs_a, _ = env_a.reset(seed=DEMO_SEED)
        obs_b, _ = env_b.reset(seed=DEMO_SEED)
        assert_nested_equal(self, obs_a, obs_b)

        _, reward_a, term_a, trunc_a, info_a = env_a.step(action)
        _, reward_b, term_b, trunc_b, info_b = env_b.step(action)
        self.assertAlmostEqual(reward_a, reward_b)
        self.assertEqual(term_a, term_b)
        self.assertEqual(trunc_a, trunc_b)
        assert_nested_equal(self, info_a["leader_action"], info_b["leader_action"])
        self.assertTrue(np.array_equal(info_a["selected_tasks"], info_b["selected_tasks"]))
        self.assertTrue(
            np.array_equal(
                info_a["follower_sensing_durations_s"],
                info_b["follower_sensing_durations_s"],
            )
        )

    def test_training_wrappers_preserve_base_env_math(self) -> None:
        config = make_demo_config()
        latent = np.linspace(
            -0.4,
            0.4,
            latent_action_dim(config.num_tasks, config.num_vehicles),
            dtype=np.float32,
        )
        decoded = decode_leader_action(
            latent.astype(np.float64),
            num_tasks=config.num_tasks,
            num_vehicles=config.num_vehicles,
            pricing_budget_max=config.pricing_budget_max,
        )

        base_env = make_demo_env()
        base_env.reset(seed=DEMO_SEED)
        _, base_reward, _, _, base_info = base_env.step(
            {
                "prices": np.asarray(decoded.prices, dtype=np.float64),
                "bandwidth": np.asarray(decoded.bandwidth, dtype=np.float64),
            }
        )

        wrapped_env = make_demo_env()
        wrapped = TEPPORewardWrapper(wrapped_env, reward_config=RewardConfig())
        wrapped = LeaderLatentActionWrapper(
            wrapped,
            num_tasks=config.num_tasks,
            num_vehicles=config.num_vehicles,
            pricing_budget_max=config.pricing_budget_max,
        )
        wrapped = FlattenLeaderObservationWrapper(wrapped)
        wrapped.reset(seed=DEMO_SEED)
        _, _, _, _, wrapped_info = wrapped.step(latent)

        self.assertAlmostEqual(wrapped_info["raw_rsu_reward"], base_reward)
        self.assertTrue(np.array_equal(wrapped_info["selected_tasks"], base_info["selected_tasks"]))
        self.assertTrue(
            np.array_equal(
                wrapped_info["follower_sensing_durations_s"],
                base_info["follower_sensing_durations_s"],
            )
        )
        self.assertTrue(
            np.array_equal(
                wrapped_info["follower_compression_ratios"],
                base_info["follower_compression_ratios"],
            )
        )
        self.assertTrue(
            np.array_equal(
                wrapped_info["energy_components"]["total_j"],
                base_info["energy_components"]["total_j"],
            )
        )

    def test_reward_components_are_individually_traceable(self) -> None:
        config = make_demo_config()
        latent = np.zeros(latent_action_dim(config.num_tasks, config.num_vehicles), dtype=np.float32)
        env = make_demo_env()
        wrapped = TEPPORewardWrapper(env, reward_config=RewardConfig())
        wrapped = LeaderLatentActionWrapper(
            wrapped,
            num_tasks=config.num_tasks,
            num_vehicles=config.num_vehicles,
            pricing_budget_max=config.pricing_budget_max,
        )
        wrapped.reset(seed=DEMO_SEED)
        _, reward, _, _, info = wrapped.step(latent)

        self.assertIn("raw_rsu_reward", info)
        self.assertIn("penalty_fail", info)
        self.assertIn("penalty_lat", info)
        self.assertIn("penalty_part", info)
        self.assertIn("te_ppo_reward", info)
        expected = (
            info["raw_rsu_reward"]
            - info["penalty_fail"]
            - info["penalty_lat"]
            - info["penalty_part"]
        )
        self.assertAlmostEqual(info["te_ppo_reward"], expected)
        self.assertAlmostEqual(reward, expected)

    def test_info_dict_reconstructs_one_full_decision(self) -> None:
        env = make_demo_env()
        env.reset(seed=DEMO_SEED)
        _, reward, _, _, info = env.step(make_demo_action())

        prices = np.asarray(info["leader_action"]["prices"], dtype=np.float64)
        bandwidth = np.asarray(info["leader_action"]["bandwidth"], dtype=np.float64)
        betas = np.asarray(info["observation_snapshot"]["tasks"]["beta"], dtype=np.float64)
        follower_responses = tuple(info["follower_responses"])

        rebuilt_selected = []
        rebuilt_tau = []
        rebuilt_eta = []
        rebuilt_sensing = []
        rebuilt_computation = []
        rebuilt_transmission = []
        selected_quality_by_task = [0.0 for _ in prices]
        selected_payment = 0.0
        for response in follower_responses:
            rebuilt_selected.append(-1 if response.selected_task_id is None else response.selected_task_id)
            if response.selected_task_id is None:
                rebuilt_tau.append(0.0)
                rebuilt_eta.append(0.0)
                rebuilt_sensing.append(0.0)
                rebuilt_computation.append(0.0)
                rebuilt_transmission.append(0.0)
                continue
            selected_eval = next(
                evaluation
                for evaluation in response.task_evaluations
                if evaluation.task_id == response.selected_task_id
            )
            rebuilt_tau.append(selected_eval.tau_s_s)
            rebuilt_eta.append(selected_eval.eta)
            rebuilt_sensing.append(selected_eval.sensing_energy_j)
            rebuilt_computation.append(selected_eval.computation_energy_j)
            rebuilt_transmission.append(selected_eval.transmission_energy_j)
            selected_quality_by_task[selected_eval.task_id] += selected_eval.quality
            selected_payment += prices[selected_eval.task_id] * selected_eval.quality

        rebuilt_task_values = [
            task_value(float(beta), float(total_quality))
            for beta, total_quality in zip(betas, selected_quality_by_task, strict=True)
        ]
        rebuilt_bandwidth_cost = bandwidth_cost(
            bandwidth_cost_coeff=info["system_constants"]["bandwidth_cost_coeff_per_hz"],
            total_bandwidth_hz=info["system_constants"]["system_bandwidth_hz"],
            allocated_bandwidth_fractions=bandwidth.tolist(),
        )
        rebuilt_reward = sum(rebuilt_task_values) - selected_payment - rebuilt_bandwidth_cost

        self.assertEqual(rebuilt_selected, info["selected_tasks"].tolist())
        self.assertTrue(np.allclose(rebuilt_tau, info["follower_sensing_durations_s"]))
        self.assertTrue(np.allclose(rebuilt_eta, info["follower_compression_ratios"]))
        self.assertTrue(np.allclose(rebuilt_sensing, info["energy_components"]["sensing_j"]))
        self.assertTrue(np.allclose(rebuilt_computation, info["energy_components"]["computation_j"]))
        self.assertTrue(np.allclose(rebuilt_transmission, info["energy_components"]["transmission_j"]))
        self.assertTrue(np.allclose(rebuilt_task_values, info["utility_components"]["task_values"]))
        self.assertAlmostEqual(rebuilt_bandwidth_cost, info["utility_components"]["bandwidth_cost"])
        self.assertAlmostEqual(rebuilt_reward, reward)


if __name__ == "__main__":
    unittest.main()
