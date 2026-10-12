import unittest

from src.examples.sanity_experiments import run_all_sanity_experiments


def experiment_map() -> dict[str, dict[str, object]]:
    payload = run_all_sanity_experiments()
    return {experiment["experiment_id"]: experiment for experiment in payload["experiments"]}


class TestSanityExperiments(unittest.TestCase):
    def test_price_monotonicity_experiment_matches_current_demo_regime(self) -> None:
        experiment = experiment_map()["A_price_monotonicity"]
        trend = experiment["trend_checks"]
        rows = experiment["rows"]

        self.assertTrue(trend["participants_non_decreasing"])
        self.assertTrue(trend["task_tau_non_decreasing"])
        self.assertTrue(trend["payment_non_decreasing"])
        self.assertEqual(trend["social_welfare_shape"], "interior_peak")
        self.assertTrue(all(0.0 <= row["price"] <= 0.3 for row in rows))
        self.assertEqual(rows[0]["participant_count"], 0)
        self.assertEqual(rows[-1]["participant_count"], 2)

    def test_bandwidth_monotonicity_experiment_has_expected_direction(self) -> None:
        experiment = experiment_map()["B_bandwidth_monotonicity"]
        trend = experiment["trend_checks"]
        rows = experiment["rows"]

        self.assertTrue(trend["tau_tr_strictly_decreasing"])
        self.assertTrue(trend["E_tr_strictly_decreasing"])
        self.assertTrue(trend["selected_utility_non_decreasing_if_task_unchanged"])
        self.assertFalse(trend["selection_regime_changes"])
        self.assertTrue(all(row["selected_task_id"] == 0 for row in rows))

    def test_occlusion_sweep_reports_saturation_plateau(self) -> None:
        experiment = experiment_map()["C_occlusion_sensitivity"]
        trend = experiment["trend_checks"]
        rows = experiment["rows"]

        self.assertTrue(trend["A_non_increasing"])
        self.assertTrue(trend["q_non_increasing"])
        self.assertTrue(trend["tilde_utility_non_increasing"])
        self.assertFalse(trend["selection_becomes_less_favorable"])
        self.assertTrue(all(row["A_i_k"] == 50.0 for row in rows))
        self.assertTrue(all(row["q_i_k"] == rows[0]["q_i_k"] for row in rows))
        self.assertIn("selection threshold was not crossed on this occlusion grid", experiment["notes"])
        self.assertEqual(rows[0]["selected_task_id"], 0)
        self.assertEqual(rows[-1]["selected_task_id"], 0)

    def test_lambda_sensitivity_experiment_captures_dropout_threshold(self) -> None:
        experiment = experiment_map()["D_lambda_sensitivity"]
        trend = experiment["trend_checks"]
        rows = experiment["rows"]

        self.assertTrue(trend["non_participation_ratio_non_decreasing"])
        self.assertTrue(trend["vehicle_2_tau_non_increasing_before_dropout"])
        self.assertTrue(trend["vehicle_2_eventually_drops_out"])
        self.assertEqual(rows[0]["vehicle_2_selected_task_id"], 1)
        self.assertEqual(rows[-1]["vehicle_2_selected_task_id"], -1)
        self.assertEqual(rows[-1]["non_participation_ratio"], 1.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
