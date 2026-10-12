import unittest

import pytest

from src.sim.scenario import ScenarioConfig, sample_scenario, task_distance_for_vehicle


class TestScenarioGeometry(unittest.TestCase):
    def test_default_distance_is_to_rectangle_with_floor(self) -> None:
        config = ScenarioConfig()
        self.assertEqual(config.roi_distance_model, "rectangle")
        self.assertEqual(
            task_distance_for_vehicle(
                vehicle_position_m=(13.0, 14.0),
                task_center_m=(10.0, 10.0),
                task_length_m=20.0,
                task_width_m=6.0,
                roi_distance_model=config.roi_distance_model,
                distance_floor_m=config.distance_floor_m,
            ),
            1.0,
        )

    def test_center_distance_remains_available_for_diagnostics(self) -> None:
        self.assertEqual(
            task_distance_for_vehicle(
                vehicle_position_m=(13.0, 14.0),
                task_center_m=(10.0, 10.0),
                task_length_m=20.0,
                task_width_m=6.0,
                roi_distance_model="center",
                distance_floor_m=1.0,
            ),
            5.0,
        )

    def test_rectangle_distance_uses_floor_inside_roi(self) -> None:
        distance = task_distance_for_vehicle(
            vehicle_position_m=(10.0, 10.0),
            task_center_m=(10.0, 10.0),
            task_length_m=20.0,
            task_width_m=6.0,
            roi_distance_model="rectangle",
            distance_floor_m=1.0,
        )

        self.assertEqual(distance, 1.0)

    def test_rectangle_distance_matches_edge_offset_outside_roi(self) -> None:
        distance = task_distance_for_vehicle(
            vehicle_position_m=(25.0, 10.0),
            task_center_m=(10.0, 10.0),
            task_length_m=20.0,
            task_width_m=6.0,
            roi_distance_model="rectangle",
            distance_floor_m=1.0,
        )

        self.assertEqual(distance, 5.0)

    def test_sampled_scenario_respects_distance_floor(self) -> None:
        config = ScenarioConfig(num_vehicles=6, distance_floor_m=1.0, roi_distance_model="rectangle")
        rng = __import__("numpy").random.default_rng(123)

        state = sample_scenario(rng, config)

        min_distance = min(min(row) for row in state.distance_to_roi_m)
        self.assertGreaterEqual(min_distance, 1.0)

    def test_sampled_scenario_occlusion_stays_within_narrowed_range(self) -> None:
        config = ScenarioConfig(num_vehicles=8, occlusion_range=(0.3, 0.5))
        rng = __import__("numpy").random.default_rng(123)

        state = sample_scenario(rng, config)

        all_values = [value for row in state.occlusion_factors for value in row]
        self.assertGreaterEqual(min(all_values), 0.3)
        self.assertLessEqual(max(all_values), 0.5)

    def test_default_occlusion_range_matches_configured_uniform(self) -> None:
        """Omitted and explicit default occlusion ranges yield identical seeded draws."""
        config_default = ScenarioConfig(num_vehicles=6)
        config_explicit = ScenarioConfig(num_vehicles=6, occlusion_range=(0.0, 1.0))
        np = __import__("numpy")

        state_default = sample_scenario(np.random.default_rng(123), config_default)
        state_explicit = sample_scenario(np.random.default_rng(123), config_explicit)

        self.assertEqual(state_default.occlusion_factors, state_explicit.occlusion_factors)


if __name__ == "__main__":
    unittest.main()


# ── Task-count expansion (K > 4) ──────────────────────────────────────

def test_default_task_geometry_is_unchanged_by_the_subdivision_path():
    """K <= 4 must reproduce the published geometry bit for bit.

    Values captured before _subdivided_task_regions existed. Task centres are
    drawn before vehicle positions, so a change in the number of RNG draws
    would shift the vehicles too -- hence the vehicle assertion.
    """
    import numpy as np

    from src.sim.scenario import ScenarioConfig, sample_scenario

    state = sample_scenario(np.random.default_rng(0), ScenarioConfig(num_vehicles=20))

    centres = [t.roi_center_m for t in state.tasks]
    assert centres[0][0] == pytest.approx(114.198, abs=1e-3)
    assert centres[1][1] == pytest.approx(153.232, abs=1e-3)
    assert centres[2][0] == pytest.approx(157.532, abs=1e-3)
    # Vehicle sampling consumes the RNG after task sampling.
    assert state.vehicles[0].position_m[0] == pytest.approx(93.223, abs=1e-3)


@pytest.mark.parametrize("num_tasks", [3, 4, 6, 9, 12])
def test_task_regions_host_every_task_without_overlap(num_tasks):
    """RoI rectangles must stay disjoint as K grows.

    A cell is inset by half the longest task before sampling, so a long RoI
    cannot spill into its neighbour. Without that inset K=9 and K=12 produced
    overlapping regions.
    """
    import numpy as np

    from src.sim.scenario import (
        ScenarioConfig,
        make_urgency_profiles,
        sample_scenario,
    )

    config = ScenarioConfig(
        num_vehicles=20,
        urgency_profiles=make_urgency_profiles(num_tasks),
        task_region_span_m=None if num_tasks <= 4 else 108.0,
    )
    regression_seeds = tuple(range(10)) + (6467, 24900)
    for seed in regression_seeds:
        state = sample_scenario(np.random.default_rng(seed), config)
        assert len(state.tasks) == num_tasks
        boxes = [
            (
                t.roi_center_m[0] - t.length_m / 2,
                t.roi_center_m[0] + t.length_m / 2,
                t.roi_center_m[1] - t.width_m / 2,
                t.roi_center_m[1] + t.width_m / 2,
            )
            for t in state.tasks
        ]
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                overlaps = a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]
                assert not overlaps, f"tasks {i} and {j} overlap at seed {seed}"


def test_subdivision_rejects_corridor_cells_shorter_than_maximum_roi():
    import numpy as np

    from src.sim.scenario import ScenarioConfig, make_urgency_profiles, sample_scenario

    config = ScenarioConfig(
        num_vehicles=1,
        urgency_profiles=make_urgency_profiles(16),
        task_region_span_m=108.0,
    )
    with pytest.raises(ValueError, match="shorter than the maximum RoI length"):
        sample_scenario(np.random.default_rng(0), config)


def test_make_urgency_profiles_reuses_only_the_three_known_labels():
    """A fourth label would be dropped silently by episode_metrics_from_info.

    That function buckets selections by matching "high"/"mid"/"low" and has no
    fallback branch, so an unrecognised label is counted in none of the
    task_selection_* columns -- no error, just a quiet undercount.
    """
    from collections import Counter

    from src.rl.eval import EPISODE_METRIC_NAMES
    from src.sim.scenario import make_urgency_profiles

    for num_tasks in (3, 6, 9, 12):
        labels = [label for label, _, _ in make_urgency_profiles(num_tasks)]
        assert set(labels) == {"high", "mid", "low"}
        # Multiples of three keep the urgency mix balanced.
        assert set(Counter(labels).values()) == {num_tasks // 3}

    # The metric schema still has exactly the three fixed buckets plus idle.
    selection_columns = {n for n in EPISODE_METRIC_NAMES if n.startswith("task_selection_")}
    assert selection_columns == {
        "task_selection_idle",
        "task_selection_high",
        "task_selection_mid",
        "task_selection_low",
    }


def test_make_urgency_profiles_rejects_a_non_positive_count():
    from src.sim.scenario import make_urgency_profiles

    with pytest.raises(ValueError, match="positive"):
        make_urgency_profiles(0)
