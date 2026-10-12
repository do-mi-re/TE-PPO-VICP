import math
import unittest

from src.models.utilities import (
    bandwidth_cost,
    rsu_utility,
    social_welfare,
    task_value,
    total_energy,
    vehicle_task_utility,
    vehicle_utility,
)


class TestUtilities(unittest.TestCase):
    def test_total_energy_adds_components(self) -> None:
        self.assertEqual(total_energy(1.0, 2.0, 3.0), 6.0)

    def test_vehicle_task_utility_matches_formula(self) -> None:
        utility = vehicle_task_utility(2.0, 5.0, 0.5, 1.0, 2.0, 3.0)
        self.assertEqual(utility, 2.0 * 5.0 - 0.5 * (1.0 + 2.0 + 3.0))

    def test_vehicle_utility_ignores_unselected_tasks(self) -> None:
        utility = vehicle_utility(
            task_selection_flags=[1, 0],
            prices_per_task=[2.0, 10.0],
            qualities_per_task=[5.0, 5.0],
            energy_weight=0.5,
            sensing_energies_j=[1.0, 1.0],
            computation_energies_j=[2.0, 2.0],
            transmission_energies_j=[3.0, 3.0],
        )
        self.assertEqual(utility, 2.0 * 5.0 - 0.5 * 6.0)

    def test_task_value_is_zero_when_quality_is_zero(self) -> None:
        self.assertEqual(task_value(3.0, 0.0), 0.0)

    def test_bandwidth_cost_is_linear(self) -> None:
        self.assertAlmostEqual(bandwidth_cost(2.0, 10.0, [0.1, 0.2]), 6.0)

    def test_rsu_utility_matches_formula(self) -> None:
        utility = rsu_utility(
            task_weights_beta=[3.0],
            qualities_by_task=[[2.0, 1.0]],
            selections_by_task=[[1, 0]],
            prices_by_task=[4.0],
            total_bandwidth_hz=10.0,
            bandwidth_cost_coeff=2.0,
            allocated_bandwidth_fractions=[0.1, 0.2],
        )
        expected = 3.0 * math.log1p(2.0) - (4.0 * 2.0) - (2.0 * 10.0 * 0.3)
        self.assertAlmostEqual(utility, expected)

    def test_social_welfare_equals_rsu_plus_vehicle_utilities(self) -> None:
        rsu = rsu_utility(
            task_weights_beta=[3.0],
            qualities_by_task=[[2.0, 1.0]],
            selections_by_task=[[1, 0]],
            prices_by_task=[4.0],
            total_bandwidth_hz=10.0,
            bandwidth_cost_coeff=2.0,
            allocated_bandwidth_fractions=[0.1, 0.2],
        )
        vehicle_a = vehicle_utility(
            task_selection_flags=[1],
            prices_per_task=[4.0],
            qualities_per_task=[2.0],
            energy_weight=0.5,
            sensing_energies_j=[1.0],
            computation_energies_j=[2.0],
            transmission_energies_j=[3.0],
        )
        vehicle_b = vehicle_utility(
            task_selection_flags=[0],
            prices_per_task=[4.0],
            qualities_per_task=[1.0],
            energy_weight=0.25,
            sensing_energies_j=[0.5],
            computation_energies_j=[0.5],
            transmission_energies_j=[0.5],
        )
        welfare = social_welfare(
            task_weights_beta=[3.0],
            qualities_by_task=[[2.0, 1.0]],
            selections_by_task=[[1, 0]],
            energy_weights_by_vehicle=[0.5, 0.25],
            sensing_energies_by_task=[[1.0, 0.5]],
            computation_energies_by_task=[[2.0, 0.5]],
            transmission_energies_by_task=[[3.0, 0.5]],
            total_bandwidth_hz=10.0,
            bandwidth_cost_coeff=2.0,
            allocated_bandwidth_fractions=[0.1, 0.2],
        )
        self.assertAlmostEqual(welfare, rsu + vehicle_a + vehicle_b)

    def test_invalid_selection_flag_raises(self) -> None:
        with self.assertRaises(ValueError):
            vehicle_utility(
                task_selection_flags=[2],
                prices_per_task=[1.0],
                qualities_per_task=[1.0],
                energy_weight=1.0,
                sensing_energies_j=[1.0],
                computation_energies_j=[1.0],
                transmission_energies_j=[1.0],
            )


if __name__ == "__main__":
    unittest.main()
