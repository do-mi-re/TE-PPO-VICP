import math
import unittest

from src.models.sensing_quality import (
    raw_data_volume,
    sensing_energy,
    sensing_quality,
    spatial_potential,
    temporal_gain,
)


class TestSensingQuality(unittest.TestCase):
    def test_raw_data_volume_is_linear_in_duration(self) -> None:
        self.assertEqual(raw_data_volume(8.0, 0.5), 4.0)

    def test_sensing_energy_is_linear_in_duration(self) -> None:
        self.assertEqual(sensing_energy(20.0, 0.25), 5.0)

    def test_spatial_potential_is_zero_at_full_occlusion(self) -> None:
        self.assertEqual(spatial_potential(2.0, 1.0, 10.0, 80.0, 2.0), 0.0)

    def test_temporal_gain_is_zero_at_zero_duration(self) -> None:
        self.assertEqual(temporal_gain(0.0, 0.1, 5.0), 0.0)

    def test_sensing_quality_increases_with_longer_duration(self) -> None:
        short_quality = sensing_quality(2.0, 0.1, 20.0, 80.0, 2.0, 0.1, 0.05, 10.0)
        long_quality = sensing_quality(2.0, 0.1, 20.0, 80.0, 2.0, 0.5, 0.05, 10.0)
        self.assertGreater(long_quality, short_quality)

    def test_sensing_quality_decreases_with_distance(self) -> None:
        near_quality = sensing_quality(2.0, 0.1, 20.0, 80.0, 2.0, 0.5, 0.05, 10.0)
        far_quality = sensing_quality(2.0, 0.1, 40.0, 80.0, 2.0, 0.5, 0.05, 10.0)
        self.assertGreater(near_quality, far_quality)

    def test_sensing_quality_decreases_with_occlusion(self) -> None:
        low_occlusion = sensing_quality(2.0, 0.1, 20.0, 80.0, 2.0, 0.5, 0.05, 10.0)
        high_occlusion = sensing_quality(2.0, 0.6, 20.0, 80.0, 2.0, 0.5, 0.05, 10.0)
        self.assertGreater(low_occlusion, high_occlusion)

    def test_sensing_quality_has_diminishing_returns_in_duration(self) -> None:
        quality_t1 = sensing_quality(2.0, 0.1, 20.0, 80.0, 2.0, 0.1, 0.05, 10.0)
        quality_t2 = sensing_quality(2.0, 0.1, 20.0, 80.0, 2.0, 0.3, 0.05, 10.0)
        quality_t3 = sensing_quality(2.0, 0.1, 20.0, 80.0, 2.0, 0.5, 0.05, 10.0)
        self.assertGreater(quality_t2 - quality_t1, quality_t3 - quality_t2)

    def test_spatial_potential_matches_formula(self) -> None:
        expected = 3.0 * (1.0 - 0.2) * math.pow(40.0 / 80.0, -2.0)
        self.assertAlmostEqual(spatial_potential(3.0, 0.2, 40.0, 80.0, 2.0), expected)

    def test_spatial_potential_clamps_to_max_value(self) -> None:
        # Near-RoI input: uncapped A = 10·(1−0)·(1/80)^-2 = 64000.
        uncapped = spatial_potential(10.0, 0.0, 1.0, 80.0, 2.0)
        self.assertGreater(uncapped, 50.0)
        self.assertEqual(spatial_potential(10.0, 0.0, 1.0, 80.0, 2.0, max_value=50.0), 50.0)

    def test_spatial_potential_max_value_inactive_below_cap(self) -> None:
        # A below the cap is returned unchanged.
        below = spatial_potential(3.0, 0.2, 40.0, 80.0, 2.0)
        self.assertLess(below, 50.0)
        self.assertAlmostEqual(spatial_potential(3.0, 0.2, 40.0, 80.0, 2.0, max_value=50.0), below)

    def test_sensing_quality_forwards_spatial_potential_max(self) -> None:
        capped = sensing_quality(10.0, 0.0, 1.0, 80.0, 2.0, 0.1, 0.05, 10.0, spatial_potential_max=50.0)
        uncapped = sensing_quality(10.0, 0.0, 1.0, 80.0, 2.0, 0.1, 0.05, 10.0)
        self.assertLess(capped, uncapped)
        self.assertAlmostEqual(capped, 50.0 * temporal_gain(0.1, 0.05, 10.0))

    def test_invalid_max_value_raises(self) -> None:
        with self.assertRaises(ValueError):
            spatial_potential(1.0, 0.1, 10.0, 80.0, 2.0, max_value=0.0)

    def test_invalid_occlusion_raises(self) -> None:
        with self.assertRaises(ValueError):
            spatial_potential(1.0, 1.1, 10.0, 80.0, 2.0)

    def test_nonpositive_distance_raises(self) -> None:
        with self.assertRaises(ValueError):
            spatial_potential(1.0, 0.1, 0.0, 80.0, 2.0)


if __name__ == "__main__":
    unittest.main()
