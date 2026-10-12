"""Unit tests for the baseline env factories.

Each SB3-PPO-compatible baseline exports ``build_env(base_env, ...)``.
These tests verify each builder produces the right action-space shape and
reward layout so wrappers don't silently drift apart.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.baselines import dbas as _dbas
from src.baselines import ebas as _ebas
from src.baselines import spis as _spis
from src.rl.obs_flatten import FlattenLeaderObservationWrapper
from src.rl.ppo import RewardConfig, build_ppo_env
from src.sim.env import RSULeaderEnv
from src.sim.scenario import ScenarioConfig


def _base_env() -> tuple[RSULeaderEnv, ScenarioConfig]:
    config = ScenarioConfig(num_vehicles=3, pricing_budget_max=2.5)
    return RSULeaderEnv(scenario_config=config), config


class TestBaselineBuilders(unittest.TestCase):
    def test_proposed_builder_uses_latent_action_dim(self) -> None:
        env, cfg = _base_env()
        wrapped = build_ppo_env(
            env,
            scenario_config=cfg,
            reward_config=RewardConfig(),
            action_logit_bound=8.0,
            options={},
        )
        try:
            # latent dim = K (prices) + 1 (utilization) + V (bandwidth share).
            expected_dim = cfg.num_tasks + 1 + cfg.num_vehicles
            self.assertEqual(wrapped.action_space.shape, (expected_dim,))
            self.assertIsInstance(wrapped, FlattenLeaderObservationWrapper)
        finally:
            wrapped.close()

    def test_ebas_builder_exposes_only_price_logits(self) -> None:
        env, cfg = _base_env()
        wrapped = _ebas.build_env(
            env,
            scenario_config=cfg,
            reward_config=RewardConfig(),
            action_logit_bound=8.0,
            options={},
        )
        try:
            self.assertEqual(wrapped.action_space.shape, (cfg.num_tasks,))
            # Zero-logit action → uniform half-budget prices + uniform 1/V bandwidth.
            obs, _ = wrapped.reset(seed=20200)
            _, _, _, _, info = wrapped.step(np.zeros(cfg.num_tasks, dtype=np.float32))
            bw = info["leader_action"]["bandwidth"]
            np.testing.assert_allclose(bw, np.full(cfg.num_vehicles, 1.0 / cfg.num_vehicles), rtol=1e-6)
        finally:
            wrapped.close()

    def test_spis_builder_freezes_price_vector(self) -> None:
        env, cfg = _base_env()
        fixed_price = 0.4
        wrapped = _spis.build_env(
            env,
            scenario_config=cfg,
            reward_config=RewardConfig(),
            action_logit_bound=8.0,
            options={"fixed_price": fixed_price},
        )
        try:
            # Action exposes utilization logit + V bandwidth logits only.
            self.assertEqual(wrapped.action_space.shape, (1 + cfg.num_vehicles,))
            obs, _ = wrapped.reset(seed=20201)
            action = np.zeros(1 + cfg.num_vehicles, dtype=np.float32)
            _, _, _, _, info = wrapped.step(action)
            prices = info["leader_action"]["prices"]
            np.testing.assert_allclose(prices, np.full(cfg.num_tasks, fixed_price), rtol=1e-6)
        finally:
            wrapped.close()

    def test_dbas_default_price_fraction_matches_tuned_config(self) -> None:
        env, cfg = _base_env()
        try:
            env.reset(seed=20202)
            policy = _dbas.DistanceBasedPolicy()
            action = policy.act(env.require_state(), pricing_budget_max=cfg.pricing_budget_max)
            expected_price = cfg.pricing_budget_max * 0.13333333333333333
            np.testing.assert_allclose(
                action.prices,
                np.full(cfg.num_tasks, expected_price),
                rtol=1e-12,
            )
        finally:
            env.close()

if __name__ == "__main__":
    unittest.main()
