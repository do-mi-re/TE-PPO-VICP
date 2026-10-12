"""EBAS baseline: dynamic pricing + equal bandwidth allocation.

Exactness label: paper-consistent approximation.

The leader still learns task pricing through PPO, but the bandwidth share is
forced to a uniform allocation across all vehicles (``w_i = 1 / V``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gymnasium
import numpy as np
from gymnasium import spaces

from src.rl.obs_flatten import FlattenLeaderObservationWrapper
from src.rl.policy import decode_prices

if TYPE_CHECKING:
    from src.rl.ppo import RewardConfig
    from src.sim.scenario import ScenarioConfig


__all__ = ["EqualBandwidthActionWrapper", "build_env"]


class EqualBandwidthActionWrapper(gymnasium.ActionWrapper):
    """Expose only the ``K`` price logits; bandwidth is hard-coded to ``1/V``."""

    def __init__(
        self,
        env: gymnasium.Env[dict[str, object], Any],
        *,
        num_tasks: int,
        num_vehicles: int,
        pricing_budget_max: float,
        action_logit_bound: float = 8.0,
    ) -> None:
        super().__init__(env)
        if num_tasks <= 0 or num_vehicles <= 0:
            raise ValueError("num_tasks and num_vehicles must be positive")
        if action_logit_bound <= 0.0:
            raise ValueError("action_logit_bound must be positive")
        self.num_tasks = num_tasks
        self.num_vehicles = num_vehicles
        self.pricing_budget_max = pricing_budget_max
        self.action_logit_bound = action_logit_bound
        self.action_space = spaces.Box(
            low=-action_logit_bound,
            high=action_logit_bound,
            shape=(num_tasks,),
            dtype=np.float32,
        )
        self._equal_bandwidth = np.full(num_vehicles, 1.0 / num_vehicles, dtype=np.float64)

    def action(self, action: np.ndarray) -> dict[str, np.ndarray]:
        prices = decode_prices(
            np.asarray(action, dtype=np.float64),
            pricing_budget_max=self.pricing_budget_max,
        )
        return {
            "prices": np.asarray(prices, dtype=np.float64),
            "bandwidth": self._equal_bandwidth.copy(),
        }


def build_env(
    base_env: gymnasium.Env[dict[str, object], Any],
    *,
    scenario_config: "ScenarioConfig",
    reward_config: "RewardConfig",
    action_logit_bound: float,
    options: dict[str, Any],
    lean_info: bool = False,
) -> gymnasium.Env[dict[str, object], Any]:
    """Assemble the EBAS wrapper stack: TE-PPO reward + fixed equal bandwidth."""

    del options  # EBAS has no extra knobs
    # Lazy import so :mod:`src.baselines.ebas` does not form a circular import
    # with :mod:`src.rl.ppo` (which itself dispatches to this module).
    from src.rl.ppo import TEPPORewardWrapper

    env = TEPPORewardWrapper(base_env, reward_config=reward_config, lean_info=lean_info)
    env = EqualBandwidthActionWrapper(
        env,
        num_tasks=scenario_config.num_tasks,
        num_vehicles=scenario_config.num_vehicles,
        pricing_budget_max=scenario_config.pricing_budget_max,
        action_logit_bound=action_logit_bound,
    )
    return FlattenLeaderObservationWrapper(env)
