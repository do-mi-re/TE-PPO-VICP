"""Unit tests for JDLS's centralized Gaussian+MultiCategorical actor.

The assignment log-probability must carry a nonzero policy gradient to the
Categorical assignment head. Tests also check action shapes, candidate masks,
and consistency between sampled and recomputed log-probabilities.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from src.baselines.jdls import (
    CONSTRAINT_AWARE_JOINT_ACTION_PARAMETERIZATION,
    JDLSCentralizedActor,
    _build_joint_action,
    train_from_config,
)
from src.sim.env import RSULeaderEnv
from src.sim.scenario import ScenarioConfig


class TestJDLSCentralizedActor(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.obs_dim = 32
        self.num_tasks = 3
        self.num_vehicles = 4
        self.hidden_sizes = (16, 16)
        self.actor = JDLSCentralizedActor(
            obs_dim=self.obs_dim,
            num_tasks=self.num_tasks,
            num_vehicles=self.num_vehicles,
            hidden_sizes=self.hidden_sizes,
        )

    def test_sample_shapes(self) -> None:
        batch = 5
        obs = torch.randn(batch, self.obs_dim)
        cont, assign, logp = self.actor.sample(obs)
        cont_dim = self.num_tasks + 1 + self.num_vehicles + self.num_vehicles * 2
        self.assertEqual(cont.shape, (batch, cont_dim))
        self.assertEqual(assign.shape, (batch, self.num_vehicles))
        self.assertEqual(logp.shape, (batch,))
        # Assignments must all be valid task IDs.
        self.assertTrue(torch.all(assign >= 0).item())
        self.assertTrue(torch.all(assign < self.num_tasks).item())

    def test_gradient_flows_through_assign_logp(self) -> None:
        # Verify that the policy gradient can update the assignment head.
        obs = torch.randn(3, self.obs_dim)
        _, assign, logp = self.actor.sample(obs)
        # Detach the continuous action from the computation graph so we only
        # see whether the Categorical branch carries a gradient to
        # assign_logits.weight.
        loss = -logp.mean()
        loss.backward()
        grad = self.actor.assign_logits.weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_log_prob_entropy_matches_sample(self) -> None:
        obs = torch.randn(4, self.obs_dim)
        cont, assign, logp_sample = self.actor.sample(obs)
        logp_recomputed, entropy = self.actor.log_prob_entropy(obs, cont, assign)
        # Same sample → same log-prob (within float tolerance).
        self.assertTrue(torch.allclose(logp_sample.detach(), logp_recomputed.detach(), atol=1e-5))
        # Entropy must be positive for the diagonal Gaussian + Categorical mix.
        self.assertTrue(torch.all(entropy > 0).item())

    def test_candidate_mask_applies_to_sampling_and_log_prob(self) -> None:
        obs = torch.randn(2, self.obs_dim)
        mask = torch.zeros(
            (2, self.num_vehicles, self.num_tasks), dtype=torch.bool
        )
        mask[:, :, 1] = True
        # Empty candidate rows use the deterministic task-0 placeholder; the
        # decoder later gives those vehicles zero sensing/bandwidth.
        mask[1, 2, :] = False
        cont, assign, sampled_logp = self.actor.sample(obs, assignment_mask=mask)
        self.assertTrue(torch.all(assign[0] == 1).item())
        self.assertEqual(int(assign[1, 2]), 0)
        recomputed_logp, _ = self.actor.log_prob_entropy(
            obs, cont, assign, assignment_mask=mask
        )
        self.assertTrue(
            torch.allclose(sampled_logp, recomputed_logp, atol=1e-5)
        )

    def test_constraint_aware_decoder_stays_inside_latency_domain(self) -> None:
        scenario = ScenarioConfig(
            num_vehicles=self.num_vehicles,
            enable_candidate_pruning=False,
        )
        env = RSULeaderEnv(scenario_config=scenario)
        env.reset(seed=123)
        K = scenario.num_tasks
        V = scenario.num_vehicles
        cont_dim = K + 1 + V + 2 * V
        joint = _build_joint_action(
            cont_np=np.zeros(cont_dim, dtype=np.float64),
            assign_np=np.zeros(V, dtype=np.int64),
            num_tasks=K,
            num_vehicles=V,
            env=env,
            parameterization=CONSTRAINT_AWARE_JOINT_ACTION_PARAMETERIZATION,
        )
        _, _, _, _, info = env.step_joint(joint)
        expected_eta = np.sqrt(env.solver_params.compression_ratio_max)
        for response in info["follower_responses"]:
            assigned = next(
                evaluation
                for evaluation in response.task_evaluations
                if evaluation.task_id == 0
            )
            self.assertTrue(assigned.feasible)
            self.assertAlmostEqual(float(assigned.eta), expected_eta, places=10)
            self.assertLessEqual(
                float(assigned.total_latency_s),
                float(env.require_state().tasks[0].tau_max_s) + 1e-12,
            )

    def test_training_rejects_unsupported_normalization(self) -> None:
        for field in ("norm_obs", "norm_reward"):
            with self.subTest(field=field):
                config = {
                    "normalization": {
                        "norm_obs": field == "norm_obs",
                        "norm_reward": field == "norm_reward",
                    },
                }
                with self.assertRaisesRegex(ValueError, "does not support"):
                    train_from_config(config)


if __name__ == "__main__":
    unittest.main()
