import unittest

from src.models.communication import (
    channel_gain,
    transmission_energy,
    transmission_latency,
    uplink_rate,
)


class TestCommunication(unittest.TestCase):
    def test_channel_gain_decreases_with_distance(self) -> None:
        near_gain = channel_gain(1.0, 10.0, 2.0)
        far_gain = channel_gain(1.0, 20.0, 2.0)
        self.assertGreater(near_gain, far_gain)

    def test_uplink_rate_increases_with_bandwidth_fraction(self) -> None:
        channel = channel_gain(1.0, 10.0, 2.0)
        low = uplink_rate(0.1, 1_000_000.0, 0.5, channel, 1e-9)
        high = uplink_rate(0.2, 1_000_000.0, 0.5, channel, 1e-9)
        self.assertGreater(high, low)

    def test_transmission_latency_matches_rate_formula(self) -> None:
        self.assertEqual(transmission_latency(120.0, 40.0), 3.0)

    def test_transmission_latency_decreases_with_bandwidth_fraction(self) -> None:
        gain = channel_gain(1.0, 10.0, 2.0)
        low_rate = uplink_rate(0.1, 1_000_000.0, 0.5, gain, 1e-9)
        high_rate = uplink_rate(0.3, 1_000_000.0, 0.5, gain, 1e-9)
        self.assertGreater(
            transmission_latency(120.0, low_rate),
            transmission_latency(120.0, high_rate),
        )

    def test_transmission_energy_is_power_times_latency(self) -> None:
        self.assertEqual(transmission_energy(2.0, 1.5), 3.0)

    def test_zero_bandwidth_fraction_raises(self) -> None:
        with self.assertRaises(ValueError):
            uplink_rate(0.0, 1_000_000.0, 0.5, 0.1, 1e-9)

    def test_zero_rate_raises_for_latency(self) -> None:
        with self.assertRaises(ValueError):
            transmission_latency(100.0, 0.0)

    def test_nonpositive_distance_raises(self) -> None:
        with self.assertRaises(ValueError):
            channel_gain(1.0, 0.0, 2.0)


if __name__ == "__main__":
    unittest.main()
