"""Shared schema for flattening the leader environment observation.

Exactness label: implementation convenience only.

LEADER_OBS_FIELDS declares the ordering, reshape rules, and bounds used by
both SB3 dictionary observations and custom actor/critic vectors.
"""

from __future__ import annotations

from typing import Any, Final

import gymnasium
import numpy as np
from gymnasium import spaces

__all__ = [
    "FlattenLeaderObservationWrapper",
    "LEADER_OBS_FIELDS",
    "flatten_leader_observation_to_dict",
    "flatten_leader_observation_to_vector",
    "make_flat_leader_observation_space",
]


# Each entry: (flat_key, nested_path, (low, high), reshape_to_1d)
#   flat_key      — the key used in the flattened Dict / order in the vector
#   nested_path   — tuple of keys to walk in the source observation
#   (low, high)   — Box bounds used by the flattened observation space
#   reshape_to_1d — whether to ``.reshape(-1)`` (collapses per-vehicle-per-task
#                   matrices into a single 1D vector)
LEADER_OBS_FIELDS: Final[tuple[tuple[str, tuple[str, ...], tuple[float, float], bool], ...]] = (
    ("channels", ("channels",), (0.0, np.inf), False),
    ("task_beta", ("tasks", "beta"), (0.0, np.inf), False),
    ("task_tau_max_s", ("tasks", "tau_max_s"), (0.0, np.inf), False),
    ("task_length_m", ("tasks", "length_m"), (0.0, np.inf), False),
    ("task_width_m", ("tasks", "width_m"), (0.0, np.inf), False),
    ("task_candidate_limit", ("tasks", "candidate_limit"), (0.0, np.inf), False),
    ("vehicle_energy_weight", ("vehicles", "energy_weight"), (0.0, np.inf), False),
    ("vehicle_cpu_cycles_per_s", ("vehicles", "cpu_cycles_per_s"), (0.0, np.inf), False),
    ("vehicle_resolution_coeff", ("vehicles", "resolution_coeff"), (0.0, np.inf), False),
    ("vehicle_sensor_power_w", ("vehicles", "sensor_power_w"), (0.0, np.inf), False),
    ("geometry_distance_to_rsu_m", ("geometry", "distance_to_rsu_m"), (0.0, np.inf), False),
    ("geometry_distance_to_roi_m", ("geometry", "distance_to_roi_m"), (0.0, np.inf), True),
    ("geometry_occlusion_factor", ("geometry", "occlusion_factor"), (0.0, 1.0), True),
)


def _walk(source: object, path: tuple[str, ...]) -> object:
    cursor = source
    for key in path:
        cursor = cursor[key]  # type: ignore[index]
    return cursor


def flatten_leader_observation_to_dict(
    observation: dict[str, object],
) -> dict[str, np.ndarray]:
    """Flatten a nested env observation into the SB3-friendly one-level dict.

    Each value is cast to ``float32``; matrices listed with ``reshape_to_1d``
    are collapsed with ``.reshape(-1)``.
    """

    out: dict[str, np.ndarray] = {}
    for flat_key, path, _bounds, reshape in LEADER_OBS_FIELDS:
        value = np.asarray(_walk(observation, path), dtype=np.float32)
        if reshape:
            value = value.reshape(-1)
        out[flat_key] = value
    return out


def flatten_leader_observation_to_vector(
    observation: dict[str, object],
) -> np.ndarray:
    """Flatten a nested env observation into a single 1D ``float32`` vector.

    Uses the same field order as :func:`flatten_leader_observation_to_dict`,
    so MAPPO's actor/critic consume the exact projection the TE-PPO stack
    would see.
    """

    parts: list[np.ndarray] = []
    for _flat_key, path, _bounds, _reshape in LEADER_OBS_FIELDS:
        parts.append(np.asarray(_walk(observation, path), dtype=np.float32).reshape(-1))
    return np.concatenate(parts, axis=0)


def make_flat_leader_observation_space(base_space: spaces.Dict) -> spaces.Dict:
    """Build the Dict observation space matching :func:`flatten_leader_observation_to_dict`."""

    flat: dict[str, spaces.Space] = {}
    for flat_key, path, (low, high), reshape in LEADER_OBS_FIELDS:
        source: spaces.Space = base_space
        for key in path:
            source = source[key]  # type: ignore[index]
        shape = tuple(source.shape) if not reshape else (int(np.prod(source.shape)),)
        flat[flat_key] = spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)
    return spaces.Dict(flat)


class FlattenLeaderObservationWrapper(gymnasium.ObservationWrapper):
    """Flatten the nested RSU state into a one-level dict for SB3 ``MultiInputPolicy``.

    Defers to :func:`flatten_leader_observation_to_dict` / :func:`make_flat_leader_observation_space`
    so the MAPPO trainer (which consumes a flat 1D vector) stays in lock-step with
    this dict layout.
    """

    def __init__(self, env: gymnasium.Env[dict[str, object], Any]) -> None:
        super().__init__(env)
        base_obs_space = env.observation_space
        if not isinstance(base_obs_space, spaces.Dict):
            raise TypeError("FlattenLeaderObservationWrapper requires a Dict observation space")
        self.observation_space = make_flat_leader_observation_space(base_obs_space)

    def observation(self, observation: dict[str, object]) -> dict[str, np.ndarray]:
        return flatten_leader_observation_to_dict(observation)
