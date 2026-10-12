"""Regression test: ``episode_metrics_from_info`` extracts valid scalars from a real env step.

The import-time ``_validate_metric_schema`` guard in :mod:`src.rl.eval` only
proves the key set is complete against a synthetic probe. This test exercises
the function on the full info payload produced by :class:`RSULeaderEnv` +
:class:`TEPPORewardWrapper` with a real scenario state.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.game.follower_solver import FollowerResponse, TaskEvaluation
from src.rl.eval import episode_metrics_from_info
from src.rl.ppo import RewardConfig, TEPPORewardWrapper
from src.sim.env import RSULeaderEnv
from src.sim.scenario import ScenarioConfig


def _uniform_half_price_action(config: ScenarioConfig) -> dict[str, np.ndarray]:
    """Leader action with uniform half-budget price and uniform bandwidth split."""

    return {
        "prices": np.full(config.num_tasks, config.pricing_budget_max * 0.5, dtype=np.float64),
        "bandwidth": np.full(config.num_vehicles, 1.0 / config.num_vehicles, dtype=np.float64),
    }


class TestEpisodeMetricsFromInfo(unittest.TestCase):
    def test_extract_energy_quality_and_latency_from_real_step(self) -> None:
        # This test targets metric extraction, not the maintained centre-
        # distance model; pin the high-service fixture explicitly.
        scenario_config = ScenarioConfig(
            num_vehicles=4,
            pricing_budget_max=3e-4,
            roi_distance_model="rectangle",
        )
        env = TEPPORewardWrapper(
            RSULeaderEnv(scenario_config=scenario_config),
            reward_config=RewardConfig(),
        )
        env.reset(seed=20000)
        _, reward, _, _, info = env.step(_uniform_half_price_action(scenario_config))
        metrics = episode_metrics_from_info(dict(info), episode_reward=float(reward))
        env.close()

        self.assertIn("energy_total_sum", metrics)
        self.assertIn("quality_per_joule", metrics)
        self.assertIn("selected_latency_ratio_mean", metrics)
        self.assertIn("social_welfare", metrics)
        self.assertIn("weighted_energy_total", metrics)
        self.assertIn("candidate_pruned_pair_ratio", metrics)
        self.assertGreaterEqual(metrics["follower_solve_time_ms"], 0.0)
        self.assertGreaterEqual(metrics["energy_total_sum"], 0.0)
        self.assertGreaterEqual(metrics["selected_latency_ratio_mean"], 0.0)
        pruning = info["candidate_pruning"]
        total_pairs = (
            pruning["admitted_pair_count"] + pruning["pruned_pair_count"]
        )
        expected_ratio = pruning["pruned_pair_count"] / total_pairs
        self.assertAlmostEqual(metrics["candidate_pruned_pair_ratio"], expected_ratio)
        self.assertEqual(
            metrics["candidate_limit_pair_count"],
            float(np.sum(pruning["candidate_limits_by_task"])),
        )
        self.assertAlmostEqual(
            metrics["bandwidth_total"],
            metrics["bandwidth_admitted_sum"]
            + metrics["bandwidth_nonadmitted_sum"],
        )
        self.assertAlmostEqual(
            metrics["bandwidth_total"],
            metrics["bandwidth_selected_sum"] + metrics["bandwidth_idle_sum"],
        )
        # social_welfare = task_value_sum − weighted_energy_total − bandwidth_cost
        # (utilities.social_welfare). Must match within floating-point epsilon.
        self.assertAlmostEqual(
            metrics["social_welfare"],
            metrics["task_value_sum"]
            - metrics["weighted_energy_total"]
            - metrics["bandwidth_cost"],
            places=6,
        )

    def test_no_selected_service_is_invalid_and_has_effective_timeout_latency(self) -> None:
        info = {
            "raw_rsu_reward": 0.0,
            "te_ppo_reward": 0.0,
            "utility_components": {
                "task_values": np.zeros(3, dtype=np.float64),
                "effective_task_qualities": np.zeros(3, dtype=np.float64),
                "participant_counts": np.zeros(3, dtype=np.float64),
                "bandwidth_cost": 0.0,
                "weighted_energy_total": 0.0,
                "social_welfare": 0.0,
            },
            "energy_components": {
                "sensing_j": np.zeros(2, dtype=np.float64),
                "computation_j": np.zeros(2, dtype=np.float64),
                "transmission_j": np.zeros(2, dtype=np.float64),
                "total_j": np.zeros(2, dtype=np.float64),
            },
            "observation_snapshot": {
                "tasks": {"tau_max_s": np.array([0.1, 0.3, 0.5], dtype=np.float64)}
            },
            "follower_responses": (
                FollowerResponse(
                    vehicle_id=0,
                    bandwidth_fraction=0.5,
                    selected_task_id=None,
                    selected_utility=0.0,
                    selected_tau_s_s=None,
                    selected_eta=None,
                    selected_mu=None,
                    task_evaluations=(
                        TaskEvaluation(0, False, -math.inf, None, None, None, None, None, None, None, None, None, None, None),
                    ),
                ),
                FollowerResponse(
                    vehicle_id=1,
                    bandwidth_fraction=0.5,
                    selected_task_id=None,
                    selected_utility=0.0,
                    selected_tau_s_s=None,
                    selected_eta=None,
                    selected_mu=None,
                    task_evaluations=(
                        TaskEvaluation(0, False, -math.inf, None, None, None, None, None, None, None, None, None, None, None),
                    ),
                ),
            ),
            "leader_action": {"prices": np.zeros(3), "bandwidth": np.array([0.5, 0.5])},
            "num_tasks": 3,
            "selected_tasks": np.array([-1, -1], dtype=np.int64),
            "follower_sensing_durations_s": np.zeros(2, dtype=np.float64),
            "follower_compression_ratios": np.zeros(2, dtype=np.float64),
            "task_urgency_labels": ["high", "mid", "low"],
            "violated_constraints": {},
        }

        metrics = episode_metrics_from_info(info, episode_reward=0.0)

        self.assertTrue(math.isnan(metrics["mean_selected_latency_s"]))
        self.assertAlmostEqual(metrics["effective_mean_latency_s"], 0.5)
        self.assertEqual(metrics["selected_vehicle_count"], 0.0)
        self.assertEqual(metrics["service_valid"], 0.0)
        self.assertEqual(metrics["zero_service_indicator"], 1.0)

    def test_epsilon_service_is_invalid(self) -> None:
        info = {
            "raw_rsu_reward": 0.0,
            "te_ppo_reward": 0.0,
            "utility_components": {
                "task_values": np.array([0.1, 0.0, 0.0], dtype=np.float64),
                "effective_task_qualities": np.array([0.5, 0.0, 0.0], dtype=np.float64),
                "participant_counts": np.array([1.0, 0.0, 0.0], dtype=np.float64),
                "bandwidth_cost": 0.0,
                "weighted_energy_total": 1e-6,
                "social_welfare": 0.099999,
            },
            "energy_components": {
                "sensing_j": np.array([5e-7], dtype=np.float64),
                "computation_j": np.array([2e-7], dtype=np.float64),
                "transmission_j": np.array([3e-7], dtype=np.float64),
                "total_j": np.array([1e-6], dtype=np.float64),
            },
            "observation_snapshot": {
                "tasks": {"tau_max_s": np.array([0.1, 0.3, 0.5], dtype=np.float64)}
            },
            "follower_responses": (),
            "leader_action": {"prices": np.array([0.1, 0.1, 0.1]), "bandwidth": np.array([1.0])},
            "num_tasks": 3,
            "selected_tasks": np.array([0], dtype=np.int64),
            "follower_sensing_durations_s": np.array([1e-6], dtype=np.float64),
            "follower_compression_ratios": np.array([1.0], dtype=np.float64),
            "raw_data_rate_bps": 40_000_000.0,
            "task_urgency_labels": ["high", "mid", "low"],
            "violated_constraints": {},
        }

        metrics = episode_metrics_from_info(info, episode_reward=0.0)

        self.assertEqual(metrics["selected_vehicle_count"], 1.0)
        self.assertLess(metrics["total_raw_data_bits"], 100_000.0)
        self.assertLess(metrics["energy_total_sum"], 1e-4)
        self.assertLess(metrics["effective_task_quality_sum"], 1.0)
        self.assertEqual(metrics["service_valid"], 0.0)
        self.assertEqual(metrics["zero_service_indicator"], 1.0)


if __name__ == "__main__":
    unittest.main()


class TestUrgencyBucketingAtHigherK(unittest.TestCase):
    """Selections must stay accounted for when K exceeds three.

    episode_metrics_from_info sorts selections into task_selection_{high,mid,
    low} by matching the urgency label string, with no fallback branch: a label
    outside those three is counted nowhere, and idle+high+mid+low silently
    stops summing to the vehicle count. make_urgency_profiles therefore reuses
    the three labels rather than minting new ones, and this test pins the
    invariant that makes that safe.
    """

    def _info(self, labels, selected):
        import numpy as np

        return {
            "raw_rsu_reward": 0.0,
            "utility_components": {
                "effective_task_qualities": np.ones(
                    len(labels), dtype=np.float64
                ),
            },
            "responses": (),
            "leader_action": {
                "prices": np.zeros(len(labels)),
                "bandwidth": np.full(len(selected), 1.0 / len(selected)),
            },
            "num_tasks": len(labels),
            "selected_tasks": np.array(selected, dtype=np.int64),
            "follower_sensing_durations_s": np.zeros(len(selected)),
            "follower_compression_ratios": np.zeros(len(selected)),
            "task_urgency_labels": list(labels),
            "violated_constraints": {},
        }

    def test_twelve_tasks_keep_every_selection_accounted_for(self) -> None:
        from src.sim.scenario import make_urgency_profiles

        labels = [label for label, _, _ in make_urgency_profiles(12)]
        # One vehicle per task, plus two idle.
        selected = list(range(12)) + [-1, -1]
        metrics = episode_metrics_from_info(
            self._info(labels, selected), episode_reward=0.0
        )

        counted = (
            metrics["task_selection_idle"]
            + metrics["task_selection_high"]
            + metrics["task_selection_mid"]
            + metrics["task_selection_low"]
        )
        self.assertEqual(counted, len(selected))
        # Four tasks per urgency level at K=12.
        self.assertEqual(metrics["task_selection_high"], 4)
        self.assertEqual(metrics["task_selection_mid"], 4)
        self.assertEqual(metrics["task_selection_low"], 4)
        self.assertEqual(metrics["task_selection_idle"], 2)
        self.assertEqual(metrics["task_count_high"], 4)
        self.assertEqual(metrics["task_count_mid"], 4)
        self.assertEqual(metrics["task_count_low"], 4)
        self.assertEqual(metrics["task_success_rate_high"], 1.0)
        self.assertEqual(metrics["task_success_rate_mid"], 1.0)
        self.assertEqual(metrics["task_success_rate_low"], 1.0)

    def test_priority_success_rates_do_not_hide_low_urgency_failure(self) -> None:
        info = self._info(["high", "mid", "low"], [0, 1])
        info["utility_components"]["effective_task_qualities"] = np.array(
            [2.0, 1.0, 0.0],
            dtype=np.float64,
        )
        metrics = episode_metrics_from_info(info, episode_reward=0.0)

        self.assertAlmostEqual(metrics["task_success_rate"], 2.0 / 3.0)
        self.assertEqual(metrics["task_success_rate_high"], 1.0)
        self.assertEqual(metrics["task_success_rate_mid"], 1.0)
        self.assertEqual(metrics["task_success_rate_low"], 0.0)
        self.assertEqual(metrics["task_failure_count_low"], 1.0)

    def test_an_unknown_urgency_label_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown task urgency"):
            episode_metrics_from_info(
                self._info(["high", "urgent"], [0, 1]),
                episode_reward=0.0,
            )

    def test_absent_urgency_has_zero_counts_and_nan_rate(self) -> None:
        metrics = episode_metrics_from_info(
            self._info(["high"], [0]), episode_reward=0.0
        )

        self.assertEqual(metrics["task_count_mid"], 0.0)
        self.assertEqual(metrics["task_failure_count_mid"], 0.0)
        self.assertTrue(math.isnan(metrics["task_success_rate_mid"]))
