import unittest

from src.models.communication import transmission_latency
from src.models.computation import (
    compression_complexity,
    computation_energy,
    computation_latency,
    processed_data_size,
)


class TestComputation(unittest.TestCase):
    def test_processed_data_size_is_unchanged_at_eta_one(self) -> None:
        self.assertEqual(processed_data_size(120.0, 1.0), 120.0)

    def test_processed_data_size_decreases_with_larger_eta(self) -> None:
        self.assertLess(processed_data_size(120.0, 4.0), processed_data_size(120.0, 2.0))

    def test_compression_complexity_matches_formula(self) -> None:
        self.assertEqual(compression_complexity(2.0, 1.0, 3.0, 2.0), 13.0)

    def test_computation_latency_increases_with_eta(self) -> None:
        lower = computation_latency(100.0, 0.5, 1.0, 1.0, 0.2, 2.0, 1000.0)
        higher = computation_latency(100.0, 0.5, 4.0, 1.0, 0.2, 2.0, 1000.0)
        self.assertGreater(higher, lower)

    def test_computation_energy_increases_with_eta(self) -> None:
        lower = computation_energy(100.0, 0.5, 1.0, 1.0, 0.2, 2.0, 1e-9, 1000.0)
        higher = computation_energy(100.0, 0.5, 4.0, 1.0, 0.2, 2.0, 1e-9, 1000.0)
        self.assertGreater(higher, lower)

    def test_larger_eta_reduces_transmission_burden_but_raises_compute_cost(self) -> None:
        raw_bits = 100.0 * 0.5
        rate_bps = 20.0
        smaller_eta_bits = processed_data_size(raw_bits, 2.0)
        larger_eta_bits = processed_data_size(raw_bits, 4.0)
        self.assertLess(larger_eta_bits, smaller_eta_bits)
        self.assertLess(
            transmission_latency(larger_eta_bits, rate_bps),
            transmission_latency(smaller_eta_bits, rate_bps),
        )

        lower_latency = computation_latency(100.0, 0.5, 2.0, 1.0, 0.2, 2.0, 1000.0)
        higher_latency = computation_latency(100.0, 0.5, 4.0, 1.0, 0.2, 2.0, 1000.0)
        lower_energy = computation_energy(100.0, 0.5, 2.0, 1.0, 0.2, 2.0, 1e-9, 1000.0)
        higher_energy = computation_energy(100.0, 0.5, 4.0, 1.0, 0.2, 2.0, 1e-9, 1000.0)
        self.assertGreater(higher_latency, lower_latency)
        self.assertGreater(higher_energy, lower_energy)

    def test_invalid_compression_ratio_raises(self) -> None:
        with self.assertRaises(ValueError):
            processed_data_size(100.0, 0.5)

    def test_nonpositive_cpu_raises(self) -> None:
        with self.assertRaises(ValueError):
            computation_latency(100.0, 0.5, 2.0, 1.0, 0.2, 2.0, 0.0)


if __name__ == "__main__":
    unittest.main()
