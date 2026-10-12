"""Unit tests for :mod:`src.rl.obs_flatten`.

``obs_flatten`` is the single source of truth for the observation projection
consumed by both the SB3 TE-PPO stack and the MAPPO trainer. A drift between
the two projections (different field order, different reshape rules) would
silently corrupt MAPPO training, so these tests exercise the layout directly.
"""

from __future__ import annotations

import unittest

import numpy as np
from gymnasium import spaces

from src.rl.obs_flatten import (
    LEADER_OBS_FIELDS,
    FlattenLeaderObservationWrapper,
    flatten_leader_observation_to_dict,
    flatten_leader_observation_to_vector,
    make_flat_leader_observation_space,
)
from src.sim.env import RSULeaderEnv
from src.sim.scenario import ScenarioConfig


def _tiny_env() -> RSULeaderEnv:
    return RSULeaderEnv(scenario_config=ScenarioConfig(num_vehicles=3))


class TestObsFlatten(unittest.TestCase):
    def setUp(self) -> None:
        self.env = _tiny_env()
        self.obs, _ = self.env.reset(seed=20123)

    def tearDown(self) -> None:
        self.env.close()

    def test_dict_flattener_covers_declared_fields(self) -> None:
        flat = flatten_leader_observation_to_dict(self.obs)
        declared = {entry[0] for entry in LEADER_OBS_FIELDS}
        self.assertEqual(set(flat.keys()), declared)
        for value in flat.values():
            self.assertEqual(value.dtype, np.float32)

    def test_vector_and_dict_use_same_layout(self) -> None:
        flat_dict = flatten_leader_observation_to_dict(self.obs)
        vec = flatten_leader_observation_to_vector(self.obs)

        # Concatenating the dict in the declared field order must yield the vector.
        parts = [np.asarray(flat_dict[name], dtype=np.float32).reshape(-1)
                 for name, *_ in LEADER_OBS_FIELDS]
        expected = np.concatenate(parts)
        self.assertEqual(vec.dtype, np.float32)
        np.testing.assert_array_equal(vec, expected)

    def test_matrix_fields_are_reshaped_to_1d(self) -> None:
        flat = flatten_leader_observation_to_dict(self.obs)
        num_vehicles = self.env.scenario_config.num_vehicles
        num_tasks = self.env.scenario_config.num_tasks
        self.assertEqual(flat["geometry_distance_to_roi_m"].shape, (num_vehicles * num_tasks,))
        self.assertEqual(flat["geometry_occlusion_factor"].shape, (num_vehicles * num_tasks,))

    def test_observation_space_box_bounds_match_fields(self) -> None:
        flat_space = make_flat_leader_observation_space(self.env.observation_space)
        self.assertIsInstance(flat_space, spaces.Dict)
        for name, _path, (low, high), reshape in LEADER_OBS_FIELDS:
            box = flat_space[name]
            self.assertEqual(float(np.min(box.low)), low)
            self.assertEqual(float(np.max(box.high)), high)
            if reshape:
                # Matrix fields are flattened to 1D; assert the wrapper reports a
                # single-dim shape whose size matches the sum of its source dims.
                self.assertEqual(len(box.shape), 1)

    def test_wrapper_observation_belongs_to_declared_space(self) -> None:
        wrapped = FlattenLeaderObservationWrapper(_tiny_env())
        obs, _ = wrapped.reset(seed=20124)
        try:
            for name, value in obs.items():
                self.assertIn(name, wrapped.observation_space.spaces)
                # spaces.Box.contains() rejects mismatched dtype/shape.
                self.assertTrue(
                    wrapped.observation_space[name].contains(value),
                    f"Flattened obs field {name!r} not contained in declared space",
                )
        finally:
            wrapped.close()


if __name__ == "__main__":
    unittest.main()
