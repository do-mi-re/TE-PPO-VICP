import unittest

from src.examples.single_step_explain import (
    make_demo_action,
    make_demo_config,
    make_demo_scenario,
    run_demo,
)
from src.models.sensing_quality import sensing_quality, spatial_potential


class TestSingleStepExplain(unittest.TestCase):
    def test_demo_report_matches_expected_scalars(self) -> None:
        report = run_demo()

        self.assertEqual(report["selected_tasks"], [0, 1, 1])
        self.assertTrue(report["terminated"])
        self.assertFalse(report["truncated"])
        total_follower_utility = sum(
            row["tilde_U_i_k"] for row in report["per_candidate"] if row["x_i_k"] == 1
        )
        self.assertAlmostEqual(
            report["social_welfare"], report["rsu_utility"] + total_follower_utility, places=6
        )

    def test_demo_report_contains_all_candidates_and_constraints_hold(self) -> None:
        report = run_demo()

        self.assertEqual(len(report["per_candidate"]), 6)
        self.assertEqual(report["constraints"]["violated_constraints"]["selected_latency"], [])
        for item in report["constraints"]["selected_latency"]:
            self.assertTrue(item["satisfied"])
        for item in report["constraints"]["at_most_one_task_per_vehicle"]:
            self.assertTrue(item["satisfied"])
            self.assertLessEqual(item["selected_count"], 1)

    def test_report_quantities_match_direct_formula_calls(self) -> None:
        report = run_demo()
        config = make_demo_config()
        scenario = make_demo_scenario()
        action = make_demo_action()

        self.assertEqual(report["leader_action"]["prices"], [float(value) for value in action["prices"]])
        self.assertEqual(report["leader_action"]["bandwidth"], [float(value) for value in action["bandwidth"]])

        candidate = next(
            row for row in report["per_candidate"] if row["vehicle_id"] == 0 and row["task_id"] == 0
        )
        vehicle = scenario.vehicles[0]
        expected_a = spatial_potential(
            resolution_coeff=vehicle.resolution_coeff,
            occlusion_factor=scenario.occlusion_factors[0][0],
            distance_to_roi_m=scenario.distance_to_roi_m[0][0],
            sensing_range_m=config.sensing_range_m,
            sensing_decay_exponent=config.sensing_decay_exponent,
            max_value=config.spatial_potential_max,
        )
        expected_q = sensing_quality(
            resolution_coeff=vehicle.resolution_coeff,
            occlusion_factor=scenario.occlusion_factors[0][0],
            distance_to_roi_m=scenario.distance_to_roi_m[0][0],
            sensing_range_m=config.sensing_range_m,
            sensing_decay_exponent=config.sensing_decay_exponent,
            sensing_duration_s=candidate["tau_s_s"],
            data_efficiency_factor=config.data_efficiency_factor,
            raw_data_rate_bps=vehicle.raw_data_rate_bps,
            spatial_potential_max=config.spatial_potential_max,
        )
        self.assertAlmostEqual(candidate["A_i_k"], expected_a, places=6)
        self.assertAlmostEqual(candidate["q_i_k"], expected_q, places=6)
        # Vehicle 0 lies 2 m beyond the rectangle's left edge, rather than
        # 12 m from its centre. The resulting raw potential reaches the cap.
        self.assertEqual(scenario.distance_to_roi_m[0][0], 2.0)
        self.assertEqual(candidate["A_i_k"], 50.0)

    def test_demo_uses_formal_model_bounds(self) -> None:
        config = make_demo_config()
        scenario = make_demo_scenario()
        action = make_demo_action()
        self.assertEqual(config.pricing_budget_max, 0.3)
        self.assertEqual(config.spatial_potential_max, 50.0)
        self.assertTrue(config.enable_candidate_pruning)
        self.assertEqual([task.candidate_limit for task in scenario.tasks], [4, 4])
        self.assertTrue(all(0.0 <= price <= config.pricing_budget_max for price in action["prices"]))


if __name__ == "__main__":
    unittest.main()
