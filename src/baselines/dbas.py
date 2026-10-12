"""DBAS baseline: distance-based heuristic resource allocation.

Exactness label: paper-consistent approximation.

There is no learning component. The leader uses a fixed price for every task
and assigns bandwidth shares inversely proportional to the vehicle-to-RoI
distance (vehicles that are closer to any task get more bandwidth).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.sim.env import RSULeaderEnv
from src.sim.state import LeaderAction, ScenarioState

DEFAULT_FIXED_PRICE_FRACTION = 0.13333333333333333


@dataclass(frozen=True)
class DistanceBasedPolicy:
    """Distance-based heuristic policy returning a leader action per scenario."""

    fixed_price_fraction: float = DEFAULT_FIXED_PRICE_FRACTION  # of pricing_budget_max
    distance_floor_m: float = 1.0

    def act(self, state: ScenarioState, *, pricing_budget_max: float) -> LeaderAction:
        bandwidth = _distance_based_bandwidth(state, distance_floor_m=self.distance_floor_m)
        prices = np.full(
            len(state.tasks),
            float(self.fixed_price_fraction) * float(pricing_budget_max),
            dtype=np.float64,
        )
        return LeaderAction(
            prices=tuple(float(p) for p in prices),
            bandwidth=tuple(float(b) for b in bandwidth),
        )


def _distance_based_bandwidth(
    state: ScenarioState,
    *,
    distance_floor_m: float = 1.0,
) -> np.ndarray:
    """Allocate bandwidth shares inversely to each vehicle's nearest-task distance."""

    distances = np.asarray(state.distance_to_roi_m, dtype=np.float64)
    nearest = np.min(distances, axis=1)
    nearest = np.maximum(nearest, distance_floor_m)
    weights = 1.0 / nearest
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.full(len(state.vehicles), 1.0 / len(state.vehicles), dtype=np.float64)
    return weights / total


def evaluate_distance_policy(
    env: RSULeaderEnv,
    *,
    seeds: list[int],
    fixed_price_fraction: float = DEFAULT_FIXED_PRICE_FRACTION,
) -> list[dict[str, object]]:
    """Run :class:`DistanceBasedPolicy` over a fixed seed bank and return per-episode info."""

    policy = DistanceBasedPolicy(fixed_price_fraction=fixed_price_fraction)
    rows: list[dict[str, object]] = []
    for episode_idx, seed in enumerate(seeds):
        observation, _ = env.reset(seed=int(seed))
        action = policy.act(
            env.require_state(),
            pricing_budget_max=env.scenario_config.pricing_budget_max,
        )
        _, reward, terminated, truncated, info = env.step(action)
        if not (terminated or truncated):
            raise RuntimeError("DBAS expects single-step episodes")
        rows.append({
            "episode_index": episode_idx,
            "eval_seed": int(seed),
            "reward": float(reward),
            "info": dict(info),
        })
    return rows
