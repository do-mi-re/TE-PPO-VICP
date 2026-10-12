"""SPIS baseline: static pricing + dynamic bandwidth allocation.

Exactness label: paper-consistent approximation.

The leader keeps learning bandwidth shares via PPO, but the per-task price
vector is frozen to a configured constant (typically the average price of a
reference TE-PPO run).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gymnasium
import numpy as np
from gymnasium import spaces

from src.rl.obs_flatten import FlattenLeaderObservationWrapper
from src.rl.policy import decode_bandwidth

if TYPE_CHECKING:
    from src.rl.ppo import RewardConfig
    from src.sim.scenario import ScenarioConfig


__all__ = ["FixedPriceActionWrapper", "build_env", "resolve_fixed_average_price"]


class FixedPriceActionWrapper(gymnasium.ActionWrapper):
    """Expose ``1 + V`` bandwidth logits; prices come from a fixed vector."""

    def __init__(
        self,
        env: gymnasium.Env[dict[str, object], Any],
        *,
        num_tasks: int,
        num_vehicles: int,
        fixed_prices: np.ndarray,
        action_logit_bound: float = 8.0,
    ) -> None:
        super().__init__(env)
        if num_tasks <= 0 or num_vehicles <= 0:
            raise ValueError("num_tasks and num_vehicles must be positive")
        if action_logit_bound <= 0.0:
            raise ValueError("action_logit_bound must be positive")
        prices = np.asarray(fixed_prices, dtype=np.float64).reshape(-1)
        if prices.shape != (num_tasks,):
            raise ValueError(f"fixed_prices must have shape ({num_tasks},)")
        self.num_tasks = num_tasks
        self.num_vehicles = num_vehicles
        self._fixed_prices = prices
        self.action_logit_bound = action_logit_bound
        self.action_space = spaces.Box(
            low=-action_logit_bound,
            high=action_logit_bound,
            shape=(1 + num_vehicles,),
            dtype=np.float32,
        )

    def action(self, action: np.ndarray) -> dict[str, np.ndarray]:
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        utilization_logit = float(action[0])
        bandwidth_logits = action[1:]
        bandwidth = decode_bandwidth(utilization_logit, bandwidth_logits)
        return {
            "prices": self._fixed_prices.copy(),
            "bandwidth": np.asarray(bandwidth, dtype=np.float64),
        }


def resolve_fixed_average_price(
    *,
    num_tasks: int,
    explicit_value: float | list[float] | tuple[float, ...] | None = None,
    reference_run_dir: str | Path | None = None,
    pricing_budget_max: float | None = None,
) -> np.ndarray:
    """Resolve the fixed price vector for SPIS.

    Resolution order:
    1. ``explicit_value`` (scalar broadcast or per-task list).
    2. ``reference_run_dir/eval/selected_validation_rows.csv`` average price columns.
    3. Fallback: ``pricing_budget_max / 2`` for every task.
    """

    if explicit_value is not None:
        if isinstance(explicit_value, (list, tuple)):
            arr = np.asarray(explicit_value, dtype=np.float64)
            if arr.shape != (num_tasks,):
                raise ValueError(f"explicit price list must have shape ({num_tasks},)")
            return arr
        return np.full(num_tasks, float(explicit_value), dtype=np.float64)

    if reference_run_dir is not None:
        rows_path = Path(reference_run_dir) / "eval" / "selected_validation_rows.csv"
        if rows_path.exists():
            total = 0.0
            count = 0
            with rows_path.open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if "average_price" in (reader.fieldnames or []):
                    for row in reader:
                        total += float(row["average_price"])
                        count += 1
            if count > 0:
                # average_price is a scalar mean across all tasks; broadcast it
                # uniformly as the reference price for each task.
                return np.full(num_tasks, total / float(count), dtype=np.float64)

    if pricing_budget_max is None:
        raise ValueError(
            "SPIS price could not be resolved: provide explicit_value, reference_run_dir, or pricing_budget_max"
        )
    return np.full(num_tasks, 0.5 * float(pricing_budget_max), dtype=np.float64)


def build_env(
    base_env: gymnasium.Env[dict[str, object], Any],
    *,
    scenario_config: "ScenarioConfig",
    reward_config: "RewardConfig",
    action_logit_bound: float,
    options: dict[str, Any],
    lean_info: bool = False,
) -> gymnasium.Env[dict[str, object], Any]:
    """Assemble the SPIS wrapper stack: TE-PPO reward + fixed prices, dynamic bandwidth."""

    from src.rl.ppo import TEPPORewardWrapper

    fixed_prices = resolve_fixed_average_price(
        num_tasks=scenario_config.num_tasks,
        explicit_value=options.get("fixed_price"),
        reference_run_dir=options.get("reference_run_dir"),
        pricing_budget_max=scenario_config.pricing_budget_max,
    )
    env = TEPPORewardWrapper(base_env, reward_config=reward_config, lean_info=lean_info)
    env = FixedPriceActionWrapper(
        env,
        num_tasks=scenario_config.num_tasks,
        num_vehicles=scenario_config.num_vehicles,
        fixed_prices=fixed_prices,
        action_logit_bound=action_logit_bound,
    )
    return FlattenLeaderObservationWrapper(env)
