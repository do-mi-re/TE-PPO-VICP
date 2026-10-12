"""Unit tests for the joint-learning reward shaping helper.

We construct synthetic ``info`` dicts (no env) to verify the three core
behaviors the shaping is meant to deliver:

1. Negative gradient when the rationality filter zeroes everything.
2. Approximate ``shaped = raw + omega·utility`` when the episode is healthy
   (no penalties triggered).
3. ``penalty_part`` matches the ``omega_part · beta · deficit`` formula.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.baselines.joint_learning_reward import (
    JointLearningShapingConfig,
    compute_joint_learning_shaping,
)
from src.game.follower_solver import FollowerResponse, TaskEvaluation


def _make_cfg(**overrides: float | int) -> JointLearningShapingConfig:
    """Zero-weight baseline with explicit per-test overrides, so every test
    isolates exactly the terms it's asserting on."""

    defaults: dict[str, float | int] = {
        "omega_vehicle": 0.0,
        "omega_feasible": 0.0,
        "omega_pre_quality": 0.0,
        "omega_tau": 0.0,
        "omega_bandwidth_use": 0.0,
        "omega_pre_lat": 0.0,
        "omega_fail": 0.0,
        "omega_lat": 0.0,
        "omega_part": 0.0,
        "min_participants": 1,
    }
    defaults.update(overrides)
    return JointLearningShapingConfig(**defaults)


def _task_eval(
    *,
    task_id: int,
    feasible: bool,
    utility: float,
    total_latency_s: float | None = None,
) -> TaskEvaluation:
    return TaskEvaluation(
        task_id=task_id,
        feasible=feasible,
        utility=utility,
        tau_s_s=0.1,
        eta=1.0,
        mu=None,
        quality=1.0 if feasible else 0.0,
        sensing_latency_s=0.0,
        computation_latency_s=0.0,
        transmission_latency_s=0.0,
        total_latency_s=total_latency_s,
        sensing_energy_j=0.0,
        computation_energy_j=0.0,
        transmission_energy_j=0.0,
    )


def _make_info(
    *,
    num_vehicles: int,
    num_tasks: int,
    assignment: list[int],
    per_vehicle_utility: list[float],
    feasible: list[bool],
    participant_counts: list[float] | None = None,
    effective_task_qualities: list[float] | None = None,
    selected_task_ids: list[int | None] | None = None,
    total_latencies: list[float | None] | None = None,
) -> dict[str, object]:
    responses = []
    for v in range(num_vehicles):
        evaluations = [
            _task_eval(
                task_id=k,
                feasible=(k == assignment[v] and feasible[v]),
                utility=(per_vehicle_utility[v] if k == assignment[v] else -999.0),
                total_latency_s=(
                    total_latencies[v]
                    if total_latencies is not None and k == (selected_task_ids or [None])[v]
                    else None
                ),
            )
            for k in range(num_tasks)
        ]
        responses.append(
            FollowerResponse(
                vehicle_id=v,
                bandwidth_fraction=1.0 / num_vehicles,
                selected_task_id=(selected_task_ids or [None] * num_vehicles)[v],
                selected_utility=per_vehicle_utility[v] if feasible[v] else 0.0,
                selected_tau_s_s=0.1,
                selected_eta=1.0,
                selected_mu=None,
                task_evaluations=tuple(evaluations),
            )
        )
    if participant_counts is None:
        participant_counts = [0.0] * num_tasks
    if effective_task_qualities is None:
        effective_task_qualities = [0.0] * num_tasks
    return {
        "joint_action": {
            "assignment": np.asarray(assignment, dtype=np.int64),
            # tau_s is read by the omega_tau term; default zeros so tests
            # that don't set omega_tau explicitly are unaffected.
            "tau_s": np.zeros(num_vehicles, dtype=np.float64),
        },
        "follower_responses": tuple(responses),
        "utility_components": {
            "effective_task_qualities": np.asarray(effective_task_qualities, dtype=np.float64),
            "participant_counts": np.asarray(participant_counts, dtype=np.float64),
        },
        "vehicle_utilities": np.zeros(num_vehicles, dtype=np.float64),
        "total_vehicle_utility": 0.0,
    }


class TestJointLearningShaping(unittest.TestCase):
    def test_missing_latency_sentinel_can_receive_capped_overshoot(self) -> None:
        info = _make_info(
            num_vehicles=1,
            num_tasks=1,
            assignment=[0],
            per_vehicle_utility=[-float("inf")],
            feasible=[False],
        )
        cfg = _make_cfg(
            omega_pre_lat=0.5,
            pre_lat_overshoot_cap=10.0,
            penalize_missing_latency_as_cap=True,
        )
        shaped, diag = compute_joint_learning_shaping(
            0.0,
            info,
            cfg=cfg,
            task_weights=(1.0,),
            task_deadlines=(1.0,),
        )
        self.assertEqual(diag["pre_filter_latency_overshoot"], 10.0)
        self.assertEqual(shaped, -5.0)

    def test_negative_gradient_when_all_rejected(self) -> None:
        # All 3 vehicles would have been rationality-filtered (utility < 0,
        # feasible=True). post-filter total_vehicle_utility = 0, but the
        # pre-filter assigned-task utilities sum to -30.
        info = _make_info(
            num_vehicles=3,
            num_tasks=2,
            assignment=[0, 1, 0],
            per_vehicle_utility=[-10.0, -5.0, -15.0],
            feasible=[True, True, True],
        )
        cfg = _make_cfg(
            omega_vehicle=0.1, omega_feasible=1.0,
            omega_fail=2.0, omega_lat=3.0, omega_part=3.0,
        )
        shaped, diag = compute_joint_learning_shaping(
            0.0, info, cfg=cfg, task_weights=(1.0, 1.0), task_deadlines=(1.0, 1.0)
        )
        self.assertAlmostEqual(diag["raw_assigned_utility_sum"], -30.0)
        self.assertEqual(diag["num_feasible_assignments"], 3.0)
        # Net pull from utility: 0.1 * -30 = -3. Feasible bonus: +3. They
        # cancel to zero, but penalty_part (3.0 * 1.0 * 1 * 2 tasks = 6)
        # still makes the shaped reward strictly negative.
        self.assertLess(shaped, 0.0)

    def test_healthy_case_approximates_raw_plus_utility_pull(self) -> None:
        # All tasks satisfied, positive per-vehicle utility, no penalties.
        info = _make_info(
            num_vehicles=2,
            num_tasks=2,
            assignment=[0, 1],
            per_vehicle_utility=[4.0, 6.0],
            feasible=[True, True],
            participant_counts=[1.0, 1.0],
            effective_task_qualities=[3.0, 2.5],
            selected_task_ids=[0, 1],
            total_latencies=[0.1, 0.2],
        )
        cfg = _make_cfg(
            omega_vehicle=0.1, omega_feasible=1.0,
            omega_fail=2.0, omega_lat=3.0, omega_part=3.0,
        )
        raw_reward = 20.0
        shaped, diag = compute_joint_learning_shaping(
            raw_reward, info, cfg=cfg, task_weights=(1.0, 1.0), task_deadlines=(1.0, 1.0)
        )
        self.assertEqual(diag["penalty_fail"], 0.0)
        self.assertEqual(diag["penalty_part"], 0.0)
        self.assertEqual(diag["penalty_lat"], 0.0)
        # shaped = raw + 0.1 * (4 + 6) + 1.0 * 2 = 20 + 1 + 2 = 23
        self.assertAlmostEqual(shaped, 23.0, places=6)
        self.assertAlmostEqual(diag["raw_assigned_utility_sum"], 10.0)
        self.assertEqual(diag["num_feasible_assignments"], 2.0)

    def test_penalty_part_matches_formula(self) -> None:
        # Two tasks, both with zero participants and nonzero beta. Deficit per
        # task = min_participants - 0 = 2, so penalty_part = omega_part *
        # (beta_0 + beta_1) * 2.
        info = _make_info(
            num_vehicles=1,
            num_tasks=2,
            assignment=[0],
            per_vehicle_utility=[0.0],
            feasible=[False],
            participant_counts=[0.0, 0.0],
            effective_task_qualities=[0.0, 0.0],
        )
        cfg = _make_cfg(
            omega_vehicle=0.1, omega_feasible=1.0,
            omega_part=3.0, min_participants=2,
        )
        betas = (1.5, 2.5)
        shaped, diag = compute_joint_learning_shaping(
            0.0, info, cfg=cfg, task_weights=betas, task_deadlines=(1.0, 1.0)
        )
        expected = cfg.omega_part * (betas[0] + betas[1]) * 2.0
        self.assertAlmostEqual(diag["penalty_part"], expected, places=6)
        self.assertAlmostEqual(diag["participant_deficit_total"], 4.0)
        # infeasible + utility=0 still contributes 0 to raw_assigned_utility_sum
        self.assertEqual(diag["raw_assigned_utility_sum"], 0.0)

    def test_utility_sum_includes_infeasible_assignments(self) -> None:
        """Utility shaping includes assigned-task utility independently of feasibility."""

        info = _make_info(
            num_vehicles=1,
            num_tasks=2,
            assignment=[0],
            per_vehicle_utility=[5.0],
            feasible=[False],
            participant_counts=[0.0, 0.0],
            effective_task_qualities=[0.0, 0.0],
        )
        cfg = _make_cfg(omega_vehicle=1.0)
        shaped, diag = compute_joint_learning_shaping(
            0.0, info, cfg=cfg, task_weights=(1.0, 1.0), task_deadlines=(1.0, 1.0)
        )
        self.assertAlmostEqual(diag["raw_assigned_utility_sum"], 5.0, places=6)
        self.assertEqual(diag["num_feasible_assignments"], 0.0)
        # With everything else muted, shaped should be exactly raw + 1.0*5.0.
        self.assertAlmostEqual(shaped, 5.0, places=6)

    def test_pre_filter_quality_rewards_all_assignments(self) -> None:
        """Quality shaping sums log1p(quality) independently of feasibility."""

        # Manually construct responses so we can control .quality per-vehicle
        # independently from the feasible flag (the default _task_eval couples
        # them). Vehicle 0: infeasible but quality=3.0. Vehicle 1: feasible
        # and quality=5.0.
        resp0_eval = TaskEvaluation(
            task_id=0, feasible=False, utility=-1.0,
            tau_s_s=0.1, eta=1.0, mu=None, quality=3.0,
            sensing_latency_s=0.0, computation_latency_s=0.0,
            transmission_latency_s=0.0, total_latency_s=0.5,
            sensing_energy_j=0.0, computation_energy_j=0.0,
            transmission_energy_j=0.0,
        )
        resp1_eval = TaskEvaluation(
            task_id=0, feasible=True, utility=2.0,
            tau_s_s=0.1, eta=1.0, mu=None, quality=5.0,
            sensing_latency_s=0.0, computation_latency_s=0.0,
            transmission_latency_s=0.0, total_latency_s=0.3,
            sensing_energy_j=0.0, computation_energy_j=0.0,
            transmission_energy_j=0.0,
        )
        responses = (
            FollowerResponse(
                vehicle_id=0, bandwidth_fraction=0.5,
                selected_task_id=None, selected_utility=0.0,
                selected_tau_s_s=0.1, selected_eta=1.0, selected_mu=None,
                task_evaluations=(resp0_eval,),
            ),
            FollowerResponse(
                vehicle_id=1, bandwidth_fraction=0.5,
                selected_task_id=0, selected_utility=2.0,
                selected_tau_s_s=0.1, selected_eta=1.0, selected_mu=None,
                task_evaluations=(resp1_eval,),
            ),
        )
        info = {
            "joint_action": {
                "assignment": np.asarray([0, 0], dtype=np.int64),
                "tau_s": np.zeros(2, dtype=np.float64),
            },
            "follower_responses": responses,
            "utility_components": {
                "effective_task_qualities": np.asarray([5.0], dtype=np.float64),
                "participant_counts": np.asarray([1.0], dtype=np.float64),
            },
            "vehicle_utilities": np.zeros(2, dtype=np.float64),
            "total_vehicle_utility": 0.0,
        }
        cfg = _make_cfg(omega_pre_quality=1.0)
        shaped, diag = compute_joint_learning_shaping(
            0.0, info, cfg=cfg, task_weights=(1.0,), task_deadlines=(1.0,)
        )
        # ln(1+3) + ln(1+5) — both vehicles contribute, infeasible included.
        expected = float(np.log1p(3.0) + np.log1p(5.0))
        self.assertAlmostEqual(diag["pre_filter_quality_log"], expected, places=6)
        self.assertAlmostEqual(shaped, expected, places=6)

    def test_sensing_shaping_handles_infeasible_responses(self) -> None:
        """Sensing-action shaping remains defined for infeasible sentinel responses."""

        # Construct two vehicles and one task with infeasible responses:
        # every task_evaluation is a "sentinel infeasible" (feasible=False,
        # utility=-inf, quality=None, total_latency_s=None) — mirrors what
        # follower_solver._infeasible_evaluation produces when tau_s<=0.
        sentinel_eval = TaskEvaluation(
            task_id=0, feasible=False, utility=-float("inf"),
            tau_s_s=None, eta=None, mu=None, quality=None,
            sensing_latency_s=None, computation_latency_s=None,
            transmission_latency_s=None, total_latency_s=None,
            sensing_energy_j=None, computation_energy_j=None,
            transmission_energy_j=None,
        )
        responses = (
            FollowerResponse(
                vehicle_id=0, bandwidth_fraction=0.5,
                selected_task_id=None, selected_utility=0.0,
                selected_tau_s_s=None, selected_eta=None, selected_mu=None,
                task_evaluations=(sentinel_eval,),
            ),
            FollowerResponse(
                vehicle_id=1, bandwidth_fraction=0.5,
                selected_task_id=None, selected_utility=0.0,
                selected_tau_s_s=None, selected_eta=None, selected_mu=None,
                task_evaluations=(sentinel_eval,),
            ),
        )
        # Policy action: tau_s = [0.25, 0.5] (expressed in seconds). With
        # task_deadlines=(1.0,), normalized contributions per vehicle are
        # clip(tau_s / 1.0, 0, 1) = [0.25, 0.5] → sum 0.75. With omega_tau=1
        # the reward is exactly 0.75 even though feasible=False for both.
        info = {
            "joint_action": {
                "assignment": np.asarray([0, 0], dtype=np.int64),
                "tau_s": np.asarray([0.25, 0.5], dtype=np.float64),
            },
            "follower_responses": responses,
            "utility_components": {
                "effective_task_qualities": np.asarray([0.0], dtype=np.float64),
                "participant_counts": np.asarray([0.0], dtype=np.float64),
            },
            "vehicle_utilities": np.zeros(2, dtype=np.float64),
            "total_vehicle_utility": 0.0,
        }
        cfg = _make_cfg(
            omega_vehicle=1.0,       # silenced by isfinite(-inf)
            omega_feasible=1.0,      # silenced (feasible=False everywhere)
            omega_pre_quality=1.0,   # silenced (quality=None)
            omega_tau=1.0,           # the one we expect to fire
            omega_pre_lat=1.0,       # silenced (total_latency_s=None)
        )
        shaped, diag = compute_joint_learning_shaping(
            0.0, info, cfg=cfg, task_weights=(1.0,), task_deadlines=(1.0,)
        )
        # Every env-derived shaping term is 0; only omega_tau fires.
        self.assertEqual(diag["raw_assigned_utility_sum"], 0.0)
        self.assertEqual(diag["num_feasible_assignments"], 0.0)
        self.assertEqual(diag["pre_filter_quality_log"], 0.0)
        self.assertEqual(diag["pre_filter_latency_overshoot"], 0.0)
        self.assertAlmostEqual(diag["tau_nonzero_reward"], 0.75, places=6)
        self.assertAlmostEqual(shaped, 0.75, places=6)

    def test_bandwidth_shaping_handles_infeasible_responses(self) -> None:
        """Bandwidth-action shaping remains defined for infeasible sentinel responses."""

        # Vary bandwidth actions while keeping infeasible sentinel responses.
        sentinel_eval = TaskEvaluation(
            task_id=0, feasible=False, utility=-float("inf"),
            tau_s_s=None, eta=None, mu=None, quality=None,
            sensing_latency_s=None, computation_latency_s=None,
            transmission_latency_s=None, total_latency_s=None,
            sensing_energy_j=None, computation_energy_j=None,
            transmission_energy_j=None,
        )
        responses = (
            FollowerResponse(
                vehicle_id=0, bandwidth_fraction=0.0,
                selected_task_id=None, selected_utility=0.0,
                selected_tau_s_s=None, selected_eta=None, selected_mu=None,
                task_evaluations=(sentinel_eval,),
            ),
            FollowerResponse(
                vehicle_id=1, bandwidth_fraction=0.0,
                selected_task_id=None, selected_utility=0.0,
                selected_tau_s_s=None, selected_eta=None, selected_mu=None,
                task_evaluations=(sentinel_eval,),
            ),
        )
        info = {
            "joint_action": {
                "assignment": np.asarray([0, 0], dtype=np.int64),
                "tau_s": np.zeros(2, dtype=np.float64),
                # The leader's bandwidth policy: vehicle 0 gets 0.3, vehicle
                # 1 gets 0.1. Sum = 0.4, within the [0, 1] budget.
                "bandwidth": np.asarray([0.3, 0.1], dtype=np.float64),
            },
            "follower_responses": responses,
            "utility_components": {
                "effective_task_qualities": np.asarray([0.0], dtype=np.float64),
                "participant_counts": np.asarray([0.0], dtype=np.float64),
            },
            "vehicle_utilities": np.zeros(2, dtype=np.float64),
            "total_vehicle_utility": 0.0,
        }
        cfg = _make_cfg(
            omega_vehicle=1.0,            # silenced by isfinite(-inf)
            omega_feasible=1.0,           # silenced (feasible=False everywhere)
            omega_pre_quality=1.0,        # silenced (quality=None)
            omega_bandwidth_use=1.0,      # the one we expect to fire
            omega_pre_lat=1.0,            # silenced (total_latency_s=None)
        )
        shaped, diag = compute_joint_learning_shaping(
            0.0, info, cfg=cfg, task_weights=(1.0,), task_deadlines=(1.0,)
        )
        self.assertAlmostEqual(diag["bandwidth_use_reward"], 0.4, places=6)
        # Every other term is 0; shaped = 0 + 1.0 * 0.4 = 0.4.
        self.assertAlmostEqual(shaped, 0.4, places=6)

    def test_shaped_formula_all_terms_exact(self) -> None:
        """Check every term of the joint shaping formula documented in baseline.md."""

        # 2 vehicles, 1 task. Vehicle 0 is feasible (utility=4, quality=2);
        # vehicle 1 is infeasible but its utility (-3) and quality (1) are
        # finite and non-zero, so they DO contribute to the pre-filter
        # accumulators regardless of participation-filter acceptance.
        evals = (
            TaskEvaluation(
                task_id=0, feasible=True, utility=4.0,
                tau_s_s=0.2, eta=1.0, mu=None, quality=2.0,
                sensing_latency_s=0.0, computation_latency_s=0.0,
                transmission_latency_s=0.0, total_latency_s=0.3,
                sensing_energy_j=0.0, computation_energy_j=0.0,
                transmission_energy_j=0.0,
            ),
            TaskEvaluation(
                task_id=0, feasible=False, utility=-3.0,
                tau_s_s=0.4, eta=1.0, mu=None, quality=1.0,
                sensing_latency_s=0.0, computation_latency_s=0.0,
                transmission_latency_s=0.0, total_latency_s=1.5,
                sensing_energy_j=0.0, computation_energy_j=0.0,
                transmission_energy_j=0.0,
            ),
        )
        responses = (
            FollowerResponse(
                vehicle_id=0, bandwidth_fraction=0.3,
                selected_task_id=0, selected_utility=4.0,
                selected_tau_s_s=0.2, selected_eta=1.0, selected_mu=None,
                task_evaluations=(evals[0],),
            ),
            FollowerResponse(
                vehicle_id=1, bandwidth_fraction=0.2,
                selected_task_id=None, selected_utility=0.0,
                selected_tau_s_s=0.4, selected_eta=1.0, selected_mu=None,
                task_evaluations=(evals[1],),
            ),
        )
        info = {
            "joint_action": {
                "assignment": np.asarray([0, 0], dtype=np.int64),
                "tau_s": np.asarray([0.2, 0.4], dtype=np.float64),
                "bandwidth": np.asarray([0.3, 0.2], dtype=np.float64),
            },
            "follower_responses": responses,
            "utility_components": {
                # 1 task, effective quality 0 to fire penalty_fail
                "effective_task_qualities": np.asarray([0.0], dtype=np.float64),
                # 0 participants to fire penalty_part
                "participant_counts": np.asarray([0.0], dtype=np.float64),
            },
            "vehicle_utilities": np.zeros(2, dtype=np.float64),
            "total_vehicle_utility": 0.0,
        }
        cfg = JointLearningShapingConfig(
            omega_vehicle=0.5,
            omega_feasible=2.0,
            omega_pre_quality=0.7,
            omega_tau=3.0,
            omega_bandwidth_use=4.0,
            omega_pre_lat=0.1,
            pre_lat_overshoot_cap=10.0,
            omega_fail=2.0,
            omega_lat=3.0,
            omega_part=3.0,
            min_participants=1,
        )
        task_weights = (1.5,)
        task_deadlines = (1.0,)
        raw_reward = 1.0
        shaped, diag = compute_joint_learning_shaping(
            raw_reward, info, cfg=cfg,
            task_weights=task_weights, task_deadlines=task_deadlines,
        )
        # Expected breakdown:
        # raw = 1.0
        # utility: 4.0 + (-3.0) = 1.0;  0.5 * 1.0 = +0.5
        # feasible: 1 (vehicle 0 only);  2.0 * 1 = +2.0
        # pre_quality: ln(1+2) + ln(1+1) = ln(3) + ln(2);  0.7 * (ln3+ln2)
        # tau: clip(0.2/1.0)+clip(0.4/1.0) = 0.6;  3.0 * 0.6 = +1.8
        # bandwidth: 0.3+0.2 = 0.5;  4.0 * 0.5 = +2.0
        # pre_lat overshoot: v0 (0.3-1.0)/1.0 → clip to 0, v1 (1.5-1.0)/1.0=0.5
        #                   → clipped by cap 10 to 0.5;  -0.1 * 0.5 = -0.05
        # penalty_fail: quality 0.0 ≤ 0 → 2.0 * 1.5 = 3.0
        # penalty_lat:  selected_task_id for v1 is None; v0 selected=0 with
        #               total_latency_s=0.3 < deadline 1.0 → overshoot 0 → 0
        # penalty_part: deficit = 1 − 0 = 1 → 3.0 * 1.5 * 1 = 4.5
        expected = (
            1.0
            + 0.5 * 1.0
            + 2.0 * 1.0
            + 0.7 * (float(np.log1p(2.0)) + float(np.log1p(1.0)))
            + 3.0 * 0.6
            + 4.0 * 0.5
            - 0.1 * 0.5
            - 3.0
            - 0.0
            - 4.5
        )
        # Cross-check individual diagnostics against the formula.
        self.assertAlmostEqual(diag["raw_assigned_utility_sum"], 1.0, places=6)
        self.assertEqual(diag["num_feasible_assignments"], 1.0)
        self.assertAlmostEqual(
            diag["pre_filter_quality_log"],
            float(np.log1p(2.0) + np.log1p(1.0)),
            places=6,
        )
        self.assertAlmostEqual(diag["tau_nonzero_reward"], 0.6, places=6)
        self.assertAlmostEqual(diag["bandwidth_use_reward"], 0.5, places=6)
        self.assertAlmostEqual(diag["pre_filter_latency_overshoot"], 0.5, places=6)
        self.assertAlmostEqual(diag["penalty_fail"], 3.0, places=6)
        self.assertAlmostEqual(diag["penalty_lat"], 0.0, places=6)
        self.assertAlmostEqual(diag["penalty_part"], 4.5, places=6)
        self.assertAlmostEqual(shaped, expected, places=6)


if __name__ == "__main__":
    unittest.main()
