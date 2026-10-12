import unittest
from dataclasses import replace

from src.game.follower_solver import (
    ChannelProfile,
    SolverParams,
    TaskCandidate,
    VehicleProfile,
    feasible_sensing_duration_upper_bound,
    optimal_eta_given_mu,
    optimal_tau_given_mu,
    solve_latency_multiplier,
    solve_task_response,
    solve_vehicle_best_response,
    unit_latency_intensity,
    uplink_rate_for_vehicle,
)
from src.models.sensing_quality import spatial_potential


class TestFollowerSolver(unittest.TestCase):
    def make_vehicle(self, *, energy_weight: float = 0.01) -> VehicleProfile:
        return VehicleProfile(
            vehicle_id=3,
            raw_data_rate_bps=4.0,
            resolution_coeff=5.0,
            cpu_cycles_per_s=1_000.0,
            sensor_power_w=0.1,
            tx_power_w=0.1,
            energy_weight=energy_weight,
        )

    def make_channel(self) -> ChannelProfile:
        return ChannelProfile(
            distance_to_rsu_m=10.0,
            fading_power=10.0,
            path_loss_exponent=2.0,
        )

    def make_params(self) -> SolverParams:
        return SolverParams(
            sensing_range_m=80.0,
            sensing_decay_exponent=2.0,
            data_efficiency_factor=0.5,
            baseline_complexity_cycles_per_bit=1.0,
            scaling_complexity_cycles_per_bit=0.2,
            complexity_exponent=2.0,
            switched_capacitance_coeff=1e-9,
            total_bandwidth_hz=1_000.0,
            noise_power_w=1e-3,
            bisection_tol=1e-9,
            selection_tie_tol=1e-12,
        )

    def make_task(
        self,
        *,
        task_id: int,
        price: float,
        deadline_s: float = 0.5,
        distance_to_roi_m: float = 20.0,
        occlusion_factor: float = 0.1,
    ) -> TaskCandidate:
        return TaskCandidate(
            task_id=task_id,
            price_per_quality_unit=price,
            deadline_s=deadline_s,
            distance_to_roi_m=distance_to_roi_m,
            occlusion_factor=occlusion_factor,
        )

    def test_task_response_satisfies_latency_constraint(self) -> None:
        evaluation = solve_task_response(
            task=self.make_task(task_id=1, price=0.5),
            vehicle=self.make_vehicle(),
            channel=self.make_channel(),
            bandwidth_fraction=0.2,
            params=self.make_params(),
        )

        self.assertTrue(evaluation.feasible)
        self.assertIsNotNone(evaluation.total_latency_s)
        self.assertLessEqual(evaluation.total_latency_s, 0.5 + 1e-9)

    def test_feasible_sensing_upper_bound_matches_latency_identity(self) -> None:
        vehicle = self.make_vehicle()
        params = self.make_params()
        rate = 600.0
        eta = 2.0
        deadline = 0.5
        upper = feasible_sensing_duration_upper_bound(
            deadline_s=deadline,
            vehicle=vehicle,
            params=params,
            uplink_rate_bps=rate,
            compression_ratio=eta,
        )
        omega = unit_latency_intensity(
            vehicle=vehicle,
            params=params,
            uplink_rate_bps=rate,
            compression_ratio=eta,
        )
        self.assertAlmostEqual(upper * omega, deadline, places=12)

    def test_zero_bandwidth_is_infeasible(self) -> None:
        evaluation = solve_task_response(
            task=self.make_task(task_id=1, price=0.5),
            vehicle=self.make_vehicle(),
            channel=self.make_channel(),
            bandwidth_fraction=0.0,
            params=self.make_params(),
        )

        self.assertFalse(evaluation.feasible)
        self.assertEqual(evaluation.utility, float("-inf"))

    def test_optimal_tau_decreases_with_mu(self) -> None:
        vehicle = self.make_vehicle()
        params = self.make_params()
        task = self.make_task(task_id=1, price=0.05, deadline_s=2.0)
        spatial = spatial_potential(
            resolution_coeff=vehicle.resolution_coeff,
            occlusion_factor=task.occlusion_factor,
            distance_to_roi_m=task.distance_to_roi_m,
            sensing_range_m=params.sensing_range_m,
            sensing_decay_exponent=params.sensing_decay_exponent,
        )
        uplink_rate_bps = 600.0
        eta_low = optimal_eta_given_mu(
            vehicle=vehicle,
            params=params,
            uplink_rate_bps=uplink_rate_bps,
            mu=0.0,
        )
        eta_high = optimal_eta_given_mu(
            vehicle=vehicle,
            params=params,
            uplink_rate_bps=uplink_rate_bps,
            mu=5.0,
        )
        tau_low = optimal_tau_given_mu(
            task=task,
            vehicle=vehicle,
            params=params,
            spatial_potential_value=spatial,
            uplink_rate_bps=uplink_rate_bps,
            compression_ratio=eta_low,
            mu=0.0,
        )
        tau_high = optimal_tau_given_mu(
            task=task,
            vehicle=vehicle,
            params=params,
            spatial_potential_value=spatial,
            uplink_rate_bps=uplink_rate_bps,
            compression_ratio=eta_high,
            mu=5.0,
        )

        self.assertLessEqual(tau_high, tau_low)

    def test_latency_intensity_implies_total_latency_scales_with_tau(self) -> None:
        vehicle = self.make_vehicle()
        params = self.make_params()
        eta = optimal_eta_given_mu(
            vehicle=vehicle,
            params=params,
            uplink_rate_bps=600.0,
            mu=0.0,
        )
        omega = unit_latency_intensity(
            vehicle=vehicle,
            params=params,
            uplink_rate_bps=600.0,
            compression_ratio=eta,
        )

        self.assertGreater(omega, 1.0)

    def test_higher_price_increases_conditional_utility(self) -> None:
        low_eval = solve_task_response(
            task=self.make_task(task_id=1, price=0.01),
            vehicle=self.make_vehicle(),
            channel=self.make_channel(),
            bandwidth_fraction=0.2,
            params=self.make_params(),
        )
        high_eval = solve_task_response(
            task=self.make_task(task_id=1, price=0.5),
            vehicle=self.make_vehicle(),
            channel=self.make_channel(),
            bandwidth_fraction=0.2,
            params=self.make_params(),
        )

        self.assertTrue(low_eval.feasible)
        self.assertTrue(high_eval.feasible)
        self.assertGreaterEqual(high_eval.utility, low_eval.utility)

    def test_higher_price_can_change_selected_task(self) -> None:
        baseline = solve_vehicle_best_response(
            task_candidates=[
                self.make_task(task_id=1, price=0.2, distance_to_roi_m=30.0),
                self.make_task(task_id=2, price=0.2, distance_to_roi_m=20.0),
            ],
            vehicle=self.make_vehicle(),
            channel=self.make_channel(),
            bandwidth_fraction=0.2,
            params=self.make_params(),
        )
        repriced = solve_vehicle_best_response(
            task_candidates=[
                self.make_task(task_id=1, price=0.5, distance_to_roi_m=30.0),
                self.make_task(task_id=2, price=0.2, distance_to_roi_m=20.0),
            ],
            vehicle=self.make_vehicle(),
            channel=self.make_channel(),
            bandwidth_fraction=0.2,
            params=self.make_params(),
        )

        self.assertEqual(baseline.selected_task_id, 2)
        self.assertEqual(repriced.selected_task_id, 1)

    def test_higher_bandwidth_reduces_transmission_latency_and_energy(self) -> None:
        low_bandwidth = solve_task_response(
            task=self.make_task(task_id=1, price=0.5),
            vehicle=self.make_vehicle(),
            channel=self.make_channel(),
            bandwidth_fraction=0.1,
            params=self.make_params(),
        )
        high_bandwidth = solve_task_response(
            task=self.make_task(task_id=1, price=0.5),
            vehicle=self.make_vehicle(),
            channel=self.make_channel(),
            bandwidth_fraction=0.3,
            params=self.make_params(),
        )

        self.assertTrue(low_bandwidth.feasible)
        self.assertTrue(high_bandwidth.feasible)
        self.assertGreater(low_bandwidth.transmission_latency_s, high_bandwidth.transmission_latency_s)
        self.assertGreater(low_bandwidth.transmission_energy_j, high_bandwidth.transmission_energy_j)

    def test_no_task_selected_when_all_utilities_are_non_positive(self) -> None:
        response = solve_vehicle_best_response(
            task_candidates=[
                self.make_task(task_id=1, price=0.0),
                self.make_task(task_id=2, price=0.0, distance_to_roi_m=30.0),
            ],
            vehicle=self.make_vehicle(energy_weight=1.0),
            channel=self.make_channel(),
            bandwidth_fraction=0.2,
            params=self.make_params(),
        )

        self.assertIsNone(response.selected_task_id)
        self.assertEqual(response.selected_utility, 0.0)

    def test_exactly_one_task_selected(self) -> None:
        response = solve_vehicle_best_response(
            task_candidates=[
                self.make_task(task_id=1, price=0.6, distance_to_roi_m=20.0),
                self.make_task(task_id=2, price=0.2, distance_to_roi_m=45.0),
            ],
            vehicle=self.make_vehicle(),
            channel=self.make_channel(),
            bandwidth_fraction=0.2,
            params=self.make_params(),
        )

        self.assertEqual(response.selected_task_id, 1)
        positive_tasks = [evaluation for evaluation in response.task_evaluations if evaluation.utility > 0.0]
        self.assertGreaterEqual(len(positive_tasks), 1)

    def test_smallest_task_id_breaks_utility_tie(self) -> None:
        response = solve_vehicle_best_response(
            task_candidates=[
                self.make_task(task_id=2, price=0.5, distance_to_roi_m=20.0),
                self.make_task(task_id=1, price=0.5, distance_to_roi_m=20.0),
            ],
            vehicle=self.make_vehicle(),
            channel=self.make_channel(),
            bandwidth_fraction=0.2,
            params=self.make_params(),
        )

        self.assertEqual(response.selected_task_id, 1)

    def test_bracket_failure_returns_none(self) -> None:
        """``solve_latency_multiplier`` returns ``None`` when it cannot bracket a root.

        Simulated by setting ``max_bracket_expansions=0`` so only the initial
        ``upper=1`` point is tested, paired with a pathologically tight
        deadline (``1e-9 s``) which keeps the residual positive at both
        endpoints. The solver returns ``None``, which the caller maps to
        ``_infeasible_evaluation``.
        """

        vehicle = self.make_vehicle()
        channel = self.make_channel()
        params = replace(self.make_params(), max_bracket_expansions=0)
        task = self.make_task(task_id=1, price=0.5, deadline_s=1e-9)

        rate = uplink_rate_for_vehicle(
            vehicle=vehicle,
            channel=channel,
            bandwidth_fraction=0.2,
            params=params,
        )
        spatial = spatial_potential(
            resolution_coeff=vehicle.resolution_coeff,
            occlusion_factor=task.occlusion_factor,
            distance_to_roi_m=task.distance_to_roi_m,
            sensing_range_m=params.sensing_range_m,
            sensing_decay_exponent=params.sensing_decay_exponent,
        )

        mu = solve_latency_multiplier(
            task=task,
            vehicle=vehicle,
            params=params,
            spatial_potential_value=spatial,
            uplink_rate_bps=rate,
        )

        self.assertIsNone(mu)

    def test_bracket_failure_propagates_to_infeasible_evaluation(self) -> None:
        """``solve_task_response`` returns an infeasible evaluation on bracket failure."""

        params = replace(self.make_params(), max_bracket_expansions=0)
        evaluation = solve_task_response(
            task=self.make_task(task_id=1, price=0.5, deadline_s=1e-9),
            vehicle=self.make_vehicle(),
            channel=self.make_channel(),
            bandwidth_fraction=0.2,
            params=params,
        )

        self.assertFalse(evaluation.feasible)
        self.assertEqual(evaluation.utility, float("-inf"))
        self.assertIsNone(evaluation.total_latency_s)

    def test_invalid_channel_inputs_return_infeasible_evaluations(self) -> None:
        response = solve_vehicle_best_response(
            task_candidates=[
                self.make_task(task_id=1, price=0.5),
                self.make_task(task_id=2, price=0.4, distance_to_roi_m=30.0),
            ],
            vehicle=self.make_vehicle(),
            channel=ChannelProfile(
                distance_to_rsu_m=0.0,
                fading_power=10.0,
                path_loss_exponent=2.0,
            ),
            bandwidth_fraction=0.2,
            params=self.make_params(),
        )

        self.assertIsNone(response.selected_task_id)
        self.assertEqual(response.selected_utility, 0.0)
        self.assertEqual(len(response.task_evaluations), 2)
        self.assertTrue(all(not evaluation.feasible for evaluation in response.task_evaluations))
        self.assertTrue(
            all(evaluation.utility == float("-inf") for evaluation in response.task_evaluations)
        )

    def test_raw_upload_service_floor_bounds_selected_sensing_duration(self) -> None:
        vehicle = self.make_vehicle()
        params = replace(
            self.make_params(),
            compression_override=1.0,
            min_service_raw_bits=1.0,
        )

        evaluation = solve_task_response(
            task=self.make_task(task_id=1, price=0.5, deadline_s=1.0),
            vehicle=vehicle,
            channel=self.make_channel(),
            bandwidth_fraction=0.2,
            params=params,
        )

        self.assertTrue(evaluation.feasible)
        self.assertEqual(evaluation.eta, 1.0)
        self.assertIsNotNone(evaluation.tau_s_s)
        self.assertGreaterEqual(
            vehicle.raw_data_rate_bps * evaluation.tau_s_s,
            1.0 - 1e-9,
        )
        self.assertAlmostEqual(
            evaluation.transmission_latency_s,
            vehicle.raw_data_rate_bps * evaluation.tau_s_s / uplink_rate_for_vehicle(
                vehicle=vehicle,
                channel=self.make_channel(),
                bandwidth_fraction=0.2,
                params=params,
            ),
        )

    def test_raw_upload_service_floor_can_make_task_infeasible(self) -> None:
        params = replace(
            self.make_params(),
            compression_override=1.0,
            min_service_raw_bits=10.0,
        )

        evaluation = solve_task_response(
            task=self.make_task(task_id=1, price=0.5, deadline_s=0.01),
            vehicle=self.make_vehicle(),
            channel=self.make_channel(),
            bandwidth_fraction=0.2,
            params=params,
        )

        self.assertFalse(evaluation.feasible)
        self.assertIsNone(evaluation.tau_s_s)


if __name__ == "__main__":
    unittest.main()
