"""Gym-style RSU leader environment without PPO-specific logic."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import gymnasium
import numpy as np
from gymnasium import spaces

from src.game.follower_solver import (
    ChannelProfile,
    FollowerResponse,
    TaskCandidate,
    VehicleProfile,
    evaluate_assigned_task,
    infeasible_task_evaluation,
    solve_vehicle_best_response,
    uplink_rate_for_vehicle,
)
from src.models.communication import channel_gain
from src.models.utilities import bandwidth_cost, rsu_utility, task_value
from src.sim.pruning import candidate_masks
from src.sim.scenario import ScenarioConfig, make_solver_params, sample_scenario
from src.sim.state import LeaderAction, ScenarioState

# Optional instrumentation for per-cycle follower-solving time.
# Left at None the hot path pays one `is not
# None` test per step; nothing accumulates and no timing calls are made.
# Set to a list to have `_run_followers` append its own duration in seconds.
FOLLOWER_SOLVE_TIMINGS: list[float] | None = None


@dataclass(frozen=True)
class JointAction:
    """Joint leader+vehicle action used by JDLS / MAPPO baselines."""

    prices: tuple[float, ...]
    bandwidth: tuple[float, ...]
    tau_s: tuple[float, ...]
    eta: tuple[float, ...]
    assignment: tuple[int, ...]  # selected task id per vehicle, -1 if abstain


ScenarioFactory = Callable[[np.random.Generator, ScenarioConfig], ScenarioState]


def _clone_nested(value: object) -> object:
    """Return a shallowly immutable copy of nested observation-style payloads."""

    if isinstance(value, dict):
        return {key: _clone_nested(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    if isinstance(value, tuple):
        return tuple(_clone_nested(item) for item in value)
    if isinstance(value, list):
        return [_clone_nested(item) for item in value]
    return value


class RSULeaderEnv(gymnasium.Env):
    """Single-step RSU leader environment faithful to one scheduling cycle."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        scenario_config: ScenarioConfig | None = None,
        scenario_factory: ScenarioFactory | None = None,
        action_tol: float = 1e-12,
    ) -> None:
        super().__init__()
        self.scenario_config = scenario_config or ScenarioConfig()
        self.solver_params = make_solver_params(self.scenario_config)
        self.scenario_factory = scenario_factory or sample_scenario
        self.action_tol = action_tol
        self._state: ScenarioState | None = None
        self._observation: dict[str, object] | None = None
        # Per-vehicle admitted task ids, recomputed on every reset. None when
        # candidate pruning is disabled (the default).
        self._candidate_masks: tuple[frozenset[int], ...] | None = None
        self._candidate_pruning_time_ms = 0.0
        self._terminated = True

        num_tasks = self.scenario_config.num_tasks
        num_vehicles = self.scenario_config.num_vehicles
        self.action_space = spaces.Dict(
            {
                "prices": spaces.Box(
                    low=0.0,
                    high=self.scenario_config.pricing_budget_max,
                    shape=(num_tasks,),
                    dtype=np.float64,
                ),
                "bandwidth": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(num_vehicles,),
                    dtype=np.float64,
                ),
            }
        )
        self.observation_space = spaces.Dict(
            {
                "channels": spaces.Box(
                    low=0.0,
                    high=np.inf,
                    shape=(num_vehicles,),
                    dtype=np.float64,
                ),
                "tasks": spaces.Dict(
                    {
                        "beta": spaces.Box(0.0, np.inf, shape=(num_tasks,), dtype=np.float64),
                        "tau_max_s": spaces.Box(0.0, np.inf, shape=(num_tasks,), dtype=np.float64),
                        "length_m": spaces.Box(0.0, np.inf, shape=(num_tasks,), dtype=np.float64),
                        "width_m": spaces.Box(0.0, np.inf, shape=(num_tasks,), dtype=np.float64),
                        "candidate_limit": spaces.Box(0, np.iinfo(np.int64).max, shape=(num_tasks,), dtype=np.int64),
                    }
                ),
                "vehicles": spaces.Dict(
                    {
                        "energy_weight": spaces.Box(0.0, np.inf, shape=(num_vehicles,), dtype=np.float64),
                        "cpu_cycles_per_s": spaces.Box(0.0, np.inf, shape=(num_vehicles,), dtype=np.float64),
                        "resolution_coeff": spaces.Box(0.0, np.inf, shape=(num_vehicles,), dtype=np.float64),
                        "sensor_power_w": spaces.Box(0.0, np.inf, shape=(num_vehicles,), dtype=np.float64),
                    }
                ),
                "geometry": spaces.Dict(
                    {
                        "distance_to_rsu_m": spaces.Box(0.0, np.inf, shape=(num_vehicles,), dtype=np.float64),
                        "distance_to_roi_m": spaces.Box(
                            0.0,
                            np.inf,
                            shape=(num_vehicles, num_tasks),
                            dtype=np.float64,
                        ),
                        "occlusion_factor": spaces.Box(
                            0.0,
                            1.0,
                            shape=(num_vehicles, num_tasks),
                            dtype=np.float64,
                        ),
                    }
                ),
            }
        )

    def require_state(self) -> ScenarioState:
        """Return the current scenario state, raising if not yet reset."""
        if self._state is None:
            raise RuntimeError("reset() must be called before accessing state")
        return self._state

    def candidate_pair_mask(self) -> np.ndarray:
        """Return the current vehicle-task admission mask.

        Paper-exact: ``mask[i, k]`` is the candidate indicator
        :math:`c_{i,k}` used by the maintained pruning rule.  With pruning
        disabled, every vehicle-task pair is admissible.  A fresh array is
        returned so callers cannot mutate environment state.
        """

        state = self.require_state()
        shape = (len(state.vehicles), len(state.tasks))
        if self._candidate_masks is None:
            return np.ones(shape, dtype=bool)
        mask = np.zeros(shape, dtype=bool)
        for vehicle_idx, allowed_tasks in enumerate(self._candidate_masks):
            for task_id in allowed_tasks:
                mask[vehicle_idx, int(task_id)] = True
        return mask

    def candidate_vehicle_mask(self) -> np.ndarray:
        """Return :math:`m_i=max_k c_{i,k}` for the current scenario."""

        return np.any(self.candidate_pair_mask(), axis=1)

    def vehicle_channel_profiles(
        self,
    ) -> tuple[tuple[VehicleProfile, ChannelProfile], ...]:
        """Return deterministic model profiles for the current scenario.

        Exactness label: implementation convenience only.  Joint-learning
        action decoders use the same profiles as :meth:`step_joint`; exposing
        them prevents a second, potentially divergent state-to-model mapping.
        """

        return tuple(_build_vehicle_channel_profiles(self.require_state()))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[dict[str, object], dict[str, object]]:
        super().reset(seed=seed)
        if options is not None:
            raise ValueError("options are not supported in this environment")

        self._state = self.scenario_factory(self.np_random, self.scenario_config)
        # The admission set depends only on episode-static geometry, so it is
        # computed before exposing the observation/action boundary and only
        # once per scheduling cycle. None = pruning disabled.
        pruning_started = time.perf_counter()
        self._candidate_masks = (
            candidate_masks(self._state, self.scenario_config)
            if self.scenario_config.enable_candidate_pruning
            else None
        )
        self._candidate_pruning_time_ms = (
            (time.perf_counter() - pruning_started) * 1000.0
            if self.scenario_config.enable_candidate_pruning
            else 0.0
        )
        self._observation = self._build_observation(self._state)
        self._terminated = False
        return self._observation, {"seed": seed}

    def step(
        self,
        action: dict[str, np.ndarray] | LeaderAction,
    ) -> tuple[dict[str, object], float, bool, bool, dict[str, object]]:
        if self._terminated or self._state is None or self._observation is None:
            raise RuntimeError("reset() must be called before step()")

        leader_action = self._coerce_action(action)
        follower_started = time.perf_counter()
        follower_responses = self._run_followers(self._state, leader_action)
        follower_solve_time_ms = (time.perf_counter() - follower_started) * 1000.0
        aggregation_started = time.perf_counter()
        reward, utility_components = self._compute_reward(self._state, leader_action, follower_responses)
        violated_constraints = self._collect_constraint_violations(self._state, follower_responses)
        aggregation_time_ms = (time.perf_counter() - aggregation_started) * 1000.0
        info = self._build_info(
            leader_action,
            follower_responses,
            utility_components,
            violated_constraints,
            follower_solve_time_ms=follower_solve_time_ms,
            aggregation_time_ms=aggregation_time_ms,
        )

        self._terminated = True
        return self._observation, reward, True, False, info

    def _coerce_action(self, action: dict[str, np.ndarray] | LeaderAction) -> LeaderAction:
        if isinstance(action, LeaderAction):
            prices = np.asarray(action.prices, dtype=np.float64)
            bandwidth = np.asarray(action.bandwidth, dtype=np.float64)
        else:
            if not isinstance(action, dict):
                raise ValueError("action must be a LeaderAction or dict")
            if "prices" not in action or "bandwidth" not in action:
                raise ValueError("action dict must contain 'prices' and 'bandwidth'")
            prices = np.asarray(action["prices"], dtype=np.float64)
            bandwidth = np.asarray(action["bandwidth"], dtype=np.float64)

        if prices.shape != (self.scenario_config.num_tasks,):
            raise ValueError("prices action has the wrong shape")
        if bandwidth.shape != (self.scenario_config.num_vehicles,):
            raise ValueError("bandwidth action has the wrong shape")
        if np.any(prices < 0.0) or np.any(prices > self.scenario_config.pricing_budget_max + self.action_tol):
            raise ValueError("prices must stay within the leader feasible set")
        if np.any(bandwidth < 0.0) or np.any(bandwidth > 1.0 + self.action_tol):
            raise ValueError("bandwidth shares must stay within [0, 1]")
        if float(np.sum(bandwidth)) > 1.0 + self.action_tol:
            raise ValueError("bandwidth shares must sum to at most 1")

        return LeaderAction(
            prices=tuple(float(np.clip(price, 0.0, self.scenario_config.pricing_budget_max)) for price in prices),
            bandwidth=tuple(float(np.clip(weight, 0.0, 1.0)) for weight in bandwidth),
        )

    def _build_observation(self, state: ScenarioState) -> dict[str, object]:
        gains = np.array(
            [
                channel_gain(
                    fading_power=channel.fading_power,
                    distance_to_rsu_m=channel.distance_to_rsu_m,
                    path_loss_exponent=channel.path_loss_exponent,
                )
                for channel in state.channels
            ],
            dtype=np.float64,
        )
        return {
            "channels": gains,
            "tasks": {
                "beta": np.array([task.beta for task in state.tasks], dtype=np.float64),
                "tau_max_s": np.array([task.tau_max_s for task in state.tasks], dtype=np.float64),
                "length_m": np.array([task.length_m for task in state.tasks], dtype=np.float64),
                "width_m": np.array([task.width_m for task in state.tasks], dtype=np.float64),
                "candidate_limit": np.array([task.candidate_limit for task in state.tasks], dtype=np.int64),
            },
            "vehicles": {
                "energy_weight": np.array([vehicle.energy_weight for vehicle in state.vehicles], dtype=np.float64),
                "cpu_cycles_per_s": np.array([vehicle.cpu_cycles_per_s for vehicle in state.vehicles], dtype=np.float64),
                "resolution_coeff": np.array([vehicle.resolution_coeff for vehicle in state.vehicles], dtype=np.float64),
                "sensor_power_w": np.array([vehicle.sensor_power_w for vehicle in state.vehicles], dtype=np.float64),
            },
            "geometry": {
                "distance_to_rsu_m": np.array(
                    [channel.distance_to_rsu_m for channel in state.channels],
                    dtype=np.float64,
                ),
                "distance_to_roi_m": np.array(state.distance_to_roi_m, dtype=np.float64),
                "occlusion_factor": np.array(state.occlusion_factors, dtype=np.float64),
            },
        }

    def _run_followers(
        self,
        state: ScenarioState,
        action: LeaderAction,
    ) -> tuple[FollowerResponse, ...]:
        if FOLLOWER_SOLVE_TIMINGS is not None:
            started = time.perf_counter()
            try:
                return self._run_followers_impl(state, action)
            finally:
                FOLLOWER_SOLVE_TIMINGS.append(time.perf_counter() - started)
        return self._run_followers_impl(state, action)

    def _run_followers_impl(
        self,
        state: ScenarioState,
        action: LeaderAction,
    ) -> tuple[FollowerResponse, ...]:
        responses: list[FollowerResponse] = []
        for vehicle_idx, (vehicle, channel) in enumerate(
            _build_vehicle_channel_profiles(state)
        ):
            allowed = (
                None if self._candidate_masks is None
                else self._candidate_masks[vehicle_idx]
            )
            task_candidates = [
                TaskCandidate(
                    task_id=task.task_id,
                    price_per_quality_unit=action.prices[task.task_id],
                    deadline_s=task.tau_max_s,
                    distance_to_roi_m=state.distance_to_roi_m[vehicle_idx][task.task_id],
                    occlusion_factor=state.occlusion_factors[vehicle_idx][task.task_id],
                )
                for task in state.tasks
                if allowed is None or task.task_id in allowed
            ]
            responses.append(
                solve_vehicle_best_response(
                    task_candidates=task_candidates,
                    vehicle=vehicle,
                    channel=channel,
                    bandwidth_fraction=action.bandwidth[vehicle_idx],
                    params=self.solver_params,
                )
            )
        return tuple(responses)

    def _compute_reward(
        self,
        state: ScenarioState,
        action: LeaderAction,
        responses: tuple[FollowerResponse, ...],
    ) -> tuple[float, dict[str, object]]:
        qualities_by_task: list[list[float]] = [[] for _ in state.tasks]
        selections_by_task: list[list[int]] = [[] for _ in state.tasks]
        task_values_list: list[float] = []
        selected_vehicle_utilities: list[float] = []
        effective_task_qualities: list[float] = []
        participant_counts: list[int] = []
        weighted_energy_total = 0.0
        beta_weighted_quality_total = 0.0

        for task in state.tasks:
            total_quality = 0.0
            count = 0
            for v_idx, response in enumerate(responses):
                evaluation = next(
                    (
                        task_evaluation
                        for task_evaluation in response.task_evaluations
                        if task_evaluation.task_id == task.task_id
                    ),
                    None,
                )
                selected = int(
                    evaluation is not None
                    and response.selected_task_id == task.task_id
                    and evaluation.feasible
                )
                quality = (
                    0.0
                    if evaluation is None or evaluation.quality is None
                    else evaluation.quality
                )
                qualities_by_task[task.task_id].append(quality)
                selections_by_task[task.task_id].append(selected)
                if selected:
                    total_quality += quality
                    count += 1
                    weighted_energy_total += state.vehicles[v_idx].energy_weight * (
                        (0.0 if evaluation.sensing_energy_j is None else evaluation.sensing_energy_j)
                        + (0.0 if evaluation.computation_energy_j is None else evaluation.computation_energy_j)
                        + (0.0 if evaluation.transmission_energy_j is None else evaluation.transmission_energy_j)
                    )
            effective_task_qualities.append(total_quality)
            participant_counts.append(count)
            task_values_list.append(task_value(task.beta, total_quality))
            beta_weighted_quality_total += task.beta * total_quality

        for response in responses:
            selected_vehicle_utilities.append(response.selected_utility)

        reward = rsu_utility(
            task_weights_beta=[task.beta for task in state.tasks],
            qualities_by_task=qualities_by_task,
            selections_by_task=selections_by_task,
            prices_by_task=list(action.prices),
            total_bandwidth_hz=self.scenario_config.system_bandwidth_hz,
            bandwidth_cost_coeff=self.scenario_config.bandwidth_cost_coeff_per_hz,
            allocated_bandwidth_fractions=list(action.bandwidth),
        )
        bandwidth_cost_value = bandwidth_cost(
            bandwidth_cost_coeff=self.scenario_config.bandwidth_cost_coeff_per_hz,
            total_bandwidth_hz=self.scenario_config.system_bandwidth_hz,
            allocated_bandwidth_fractions=list(action.bandwidth),
        )
        # Social welfare = Σβ·ln(1+q) − Σ_v ω_v·E_v − bandwidth_cost
        # (the maintained manuscript equations, src/models/utilities.py:186). Differs from
        # rsu_utility by replacing the internal payment transfer with weighted
        # vehicle energy cost, so it's the metric to compare across algorithms
        # that pay vehicles differently.
        social_welfare_value = float(sum(task_values_list)) - weighted_energy_total - bandwidth_cost_value
        return reward, {
            "task_values": np.array(task_values_list, dtype=np.float64),
            "vehicle_selected_utilities": np.array(selected_vehicle_utilities, dtype=np.float64),
            "bandwidth_cost": bandwidth_cost_value,
            "effective_task_qualities": np.array(effective_task_qualities, dtype=np.float64),
            "participant_counts": np.array(participant_counts, dtype=np.int64),
            "weighted_energy_total": float(weighted_energy_total),
            "social_welfare": float(social_welfare_value),
            # Q_total = Σ_k β_k Σ_i x*_{i,k} q_{i,k}(τ^s*_{i,k})·I(τ^tot_{i,k}<=τ_k^max);
            # the inner sum reuses effective_task_qualities' per-task total_quality,
            # which is already gated on evaluation.feasible/selection above.
            "beta_weighted_quality_total": float(beta_weighted_quality_total),
        }

    def _collect_constraint_violations(
        self,
        state: ScenarioState,
        responses: tuple[FollowerResponse, ...],
    ) -> dict[str, object]:
        infeasible_candidates: list[dict[str, int | str]] = []
        selected_latency_violations: list[dict[str, int | float]] = []

        for response in responses:
            for evaluation in response.task_evaluations:
                if not evaluation.feasible:
                    infeasible_candidates.append(
                        {
                            "vehicle_id": response.vehicle_id,
                            "task_id": evaluation.task_id,
                            "reason": "follower_infeasible",
                        }
                    )
                if (
                    response.selected_task_id == evaluation.task_id
                    and evaluation.total_latency_s is not None
                    and evaluation.total_latency_s
                    > state.tasks[evaluation.task_id].tau_max_s + self.solver_params.bisection_tol
                ):
                    selected_latency_violations.append(
                        {
                            "vehicle_id": response.vehicle_id,
                            "task_id": evaluation.task_id,
                            "total_latency_s": evaluation.total_latency_s,
                            "deadline_s": state.tasks[evaluation.task_id].tau_max_s,
                        }
                    )

        return {
            "follower_infeasible_candidates": infeasible_candidates,
            "selected_latency": selected_latency_violations,
        }

    def _build_info(
        self,
        action: LeaderAction,
        responses: tuple[FollowerResponse, ...],
        utility_components: dict[str, object],
        violated_constraints: dict[str, object],
        *,
        follower_solve_time_ms: float = 0.0,
        aggregation_time_ms: float = 0.0,
    ) -> dict[str, object]:
        if self._observation is None:
            raise RuntimeError("observation must be initialized before building step info")

        selected_tasks = np.array(
            [-1 if response.selected_task_id is None else response.selected_task_id for response in responses],
            dtype=np.int64,
        )
        selected_tau = np.array(
            [0.0 if response.selected_tau_s_s is None else response.selected_tau_s_s for response in responses],
            dtype=np.float64,
        )
        selected_eta = np.array(
            [0.0 if response.selected_eta is None else response.selected_eta for response in responses],
            dtype=np.float64,
        )
        sensing_energy = np.zeros(len(responses), dtype=np.float64)
        computation_energy = np.zeros(len(responses), dtype=np.float64)
        transmission_energy = np.zeros(len(responses), dtype=np.float64)
        total_energy = np.zeros(len(responses), dtype=np.float64)
        for response_idx, response in enumerate(responses):
            if response.selected_task_id is None:
                continue
            selected_eval = next(
                evaluation
                for evaluation in response.task_evaluations
                if evaluation.task_id == response.selected_task_id
            )
            sensing_energy[response_idx] = 0.0 if selected_eval.sensing_energy_j is None else selected_eval.sensing_energy_j
            computation_energy[response_idx] = (
                0.0 if selected_eval.computation_energy_j is None else selected_eval.computation_energy_j
            )
            transmission_energy[response_idx] = (
                0.0 if selected_eval.transmission_energy_j is None else selected_eval.transmission_energy_j
            )
            total_energy[response_idx] = (
                sensing_energy[response_idx]
                + computation_energy[response_idx]
                + transmission_energy[response_idx]
            )

        return {
            "system_constants": {
                "system_bandwidth_hz": float(self.scenario_config.system_bandwidth_hz),
                "bandwidth_cost_coeff_per_hz": float(self.scenario_config.bandwidth_cost_coeff_per_hz),
                "pricing_budget_max": float(self.scenario_config.pricing_budget_max),
                "bisection_tol": float(self.solver_params.bisection_tol),
            },
            "leader_action": {
                "prices": np.array(action.prices, dtype=np.float64),
                "bandwidth": np.array(action.bandwidth, dtype=np.float64),
            },
            "observation_snapshot": _clone_nested(self._observation),
            "selected_tasks": selected_tasks,
            "follower_sensing_durations_s": selected_tau,
            "follower_compression_ratios": selected_eta,
            "energy_components": {
                "sensing_j": sensing_energy,
                "computation_j": computation_energy,
                "transmission_j": transmission_energy,
                "total_j": total_energy,
            },
            "utility_components": utility_components,
            "violated_constraints": violated_constraints,
            "follower_responses": responses,
            "task_urgency_labels": tuple(task.urgency_label for task in self._state.tasks),
            "num_tasks": len(self._state.tasks),
            "raw_data_rate_bps": self.scenario_config.raw_data_rate_bps,
            "task_failure_count": float(np.sum(
                np.asarray(utility_components.get("effective_task_qualities", []), dtype=np.float64) <= 0.0
            )),
            "candidate_pruning": self._candidate_pruning_diagnostics(),
            "bandwidth_allocation": self._bandwidth_allocation_diagnostics(
                action, selected_tasks
            ),
            "candidate_pruning_time_ms": float(self._candidate_pruning_time_ms),
            "follower_solve_time_ms": float(follower_solve_time_ms),
            "aggregation_time_ms": float(aggregation_time_ms),
            # Promoted to top-level so cross-algorithm callbacks (fig1) and
            # _aggregate_optional_episode_metrics can read it without nested lookup.
            "social_welfare": float(utility_components.get("social_welfare", 0.0)),
            "weighted_energy_total": float(utility_components.get("weighted_energy_total", 0.0)),
        }

    def _candidate_pruning_diagnostics(self) -> dict[str, object]:
        """Return episode-static admission counts for provenance and tests."""
        state = self.require_state()
        total_pairs = len(state.vehicles) * len(state.tasks)
        if self._candidate_masks is None:
            counts = [len(state.vehicles)] * len(state.tasks)
            admitted = total_pairs
        else:
            counts = [
                sum(task.task_id in allowed for allowed in self._candidate_masks)
                for task in state.tasks
            ]
            admitted = sum(counts)
        return {
            "enabled": self._candidate_masks is not None,
            "candidate_counts_by_task": np.asarray(counts, dtype=np.int64),
            "candidate_limits_by_task": np.asarray(
                [task.candidate_limit for task in state.tasks], dtype=np.int64
            ),
            "admitted_pair_count": int(admitted),
            "pruned_pair_count": int(total_pairs - admitted),
        }

    def _bandwidth_allocation_diagnostics(
        self,
        action: LeaderAction,
        selected_tasks: np.ndarray,
    ) -> dict[str, object]:
        """Partition allocated bandwidth by candidate and participation status."""

        bandwidth = np.asarray(action.bandwidth, dtype=np.float64)
        eligible = self.candidate_vehicle_mask()
        selected = np.asarray(selected_tasks, dtype=np.int64) >= 0
        total = float(np.sum(bandwidth))
        admitted = float(np.sum(bandwidth[eligible]))
        nonadmitted = float(np.sum(bandwidth[~eligible]))
        selected_sum = float(np.sum(bandwidth[selected]))
        idle_sum = float(np.sum(bandwidth[~selected]))
        eligible_weights = bandwidth[eligible]
        positive_total = float(np.sum(eligible_weights))
        if eligible_weights.size <= 1 or positive_total <= 0.0:
            normalized_entropy = 0.0
        else:
            probabilities = eligible_weights / positive_total
            positive = probabilities[probabilities > 0.0]
            normalized_entropy = float(
                -np.sum(positive * np.log(positive)) / np.log(eligible_weights.size)
            )
        return {
            "candidate_vehicle_count": int(np.sum(eligible)),
            "total": total,
            "admitted": admitted,
            "nonadmitted": nonadmitted,
            "selected": selected_sum,
            "idle": idle_sum,
            "nonadmitted_ratio": nonadmitted / total if total > 0.0 else 0.0,
            "selected_ratio": selected_sum / total if total > 0.0 else 0.0,
            "eligible_normalized_entropy": normalized_entropy,
        }

    # ------------------------------------------------------------------
    # Joint-action helpers used by JDLS / MAPPO baselines.
    # ------------------------------------------------------------------


    def vehicle_observation(
        self,
        vehicle_idx: int,
        prices: np.ndarray,
        bandwidth: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Return a flat per-vehicle observation conditioned on leader decisions."""

        if self._state is None or self._observation is None:
            raise RuntimeError("reset() must be called before vehicle_observation")
        global_obs = self._observation
        prices_arr = np.asarray(prices, dtype=np.float64).reshape(-1)
        bandwidth_arr = np.asarray(bandwidth, dtype=np.float64).reshape(-1)
        return {
            "self_channel": np.asarray([global_obs["channels"][vehicle_idx]], dtype=np.float64),
            "task_beta": np.asarray(global_obs["tasks"]["beta"], dtype=np.float64),
            "task_tau_max_s": np.asarray(global_obs["tasks"]["tau_max_s"], dtype=np.float64),
            "task_length_m": np.asarray(global_obs["tasks"]["length_m"], dtype=np.float64),
            "task_width_m": np.asarray(global_obs["tasks"]["width_m"], dtype=np.float64),
            "self_energy_weight": np.asarray(
                [global_obs["vehicles"]["energy_weight"][vehicle_idx]], dtype=np.float64
            ),
            "self_cpu_cycles_per_s": np.asarray(
                [global_obs["vehicles"]["cpu_cycles_per_s"][vehicle_idx]], dtype=np.float64
            ),
            "self_resolution_coeff": np.asarray(
                [global_obs["vehicles"]["resolution_coeff"][vehicle_idx]], dtype=np.float64
            ),
            "self_sensor_power_w": np.asarray(
                [global_obs["vehicles"]["sensor_power_w"][vehicle_idx]], dtype=np.float64
            ),
            "self_distance_to_rsu_m": np.asarray(
                [global_obs["geometry"]["distance_to_rsu_m"][vehicle_idx]], dtype=np.float64
            ),
            "self_distance_to_roi_m": np.asarray(
                global_obs["geometry"]["distance_to_roi_m"][vehicle_idx], dtype=np.float64
            ),
            "self_occlusion_factor": np.asarray(
                global_obs["geometry"]["occlusion_factor"][vehicle_idx], dtype=np.float64
            ),
            "leader_prices": prices_arr,
            "leader_bandwidth": np.asarray([bandwidth_arr[vehicle_idx]], dtype=np.float64),
        }

    def vehicle_observation_matrix(
        self,
        prices: np.ndarray,
        bandwidth: np.ndarray,
    ) -> np.ndarray:
        """Return all shared-vehicle observations as one ``float32`` matrix.

        Exactness: implementation convenience only.  The columns are in the
        same order as :meth:`vehicle_observation`; this method only removes
        repeated Python dictionary/array construction from MAPPO rollouts.
        """

        if self._state is None or self._observation is None:
            raise RuntimeError("reset() must be called before vehicle_observation_matrix")
        global_obs = self._observation
        V = self.scenario_config.num_vehicles
        prices_arr = np.asarray(prices, dtype=np.float32).reshape(1, -1)
        bandwidth_arr = np.asarray(bandwidth, dtype=np.float32).reshape(V, 1)

        def _column(values: object) -> np.ndarray:
            return np.asarray(values, dtype=np.float32).reshape(V, 1)

        def _repeat_task(values: object) -> np.ndarray:
            row = np.asarray(values, dtype=np.float32).reshape(1, -1)
            return np.broadcast_to(row, (V, row.shape[1]))

        return np.concatenate(
            [
                _column(global_obs["channels"]),
                _repeat_task(global_obs["tasks"]["beta"]),
                _repeat_task(global_obs["tasks"]["tau_max_s"]),
                _repeat_task(global_obs["tasks"]["length_m"]),
                _repeat_task(global_obs["tasks"]["width_m"]),
                _column(global_obs["vehicles"]["energy_weight"]),
                _column(global_obs["vehicles"]["cpu_cycles_per_s"]),
                _column(global_obs["vehicles"]["resolution_coeff"]),
                _column(global_obs["vehicles"]["sensor_power_w"]),
                _column(global_obs["geometry"]["distance_to_rsu_m"]),
                np.asarray(global_obs["geometry"]["distance_to_roi_m"], dtype=np.float32),
                np.asarray(global_obs["geometry"]["occlusion_factor"], dtype=np.float32),
                np.broadcast_to(prices_arr, (V, prices_arr.shape[1])),
                bandwidth_arr,
            ],
            axis=1,
        )

    def step_joint(
        self,
        joint: "JointAction",
        *,
        info_mode: Literal["full", "training"] = "full",
    ) -> tuple[dict[str, object], float, bool, bool, dict[str, object]]:
        """Step the environment with a fully specified joint action.

        Bypasses the follower best-response solver. Used by JDLS / MAPPO.

        ``info_mode="training"`` is an implementation-only fast path.  It
        returns exactly the fields consumed by joint-learning reward shaping
        while skipping observation cloning and diagnostic-tree construction.
        The default ``"full"`` preserves the public evaluation schema.
        """

        if self._terminated or self._state is None or self._observation is None:
            raise RuntimeError("reset() must be called before step_joint()")
        if info_mode not in ("full", "training"):
            raise ValueError("info_mode must be 'full' or 'training'")
        state = self._state
        if (
            len(joint.prices) != self.scenario_config.num_tasks
            or len(joint.bandwidth) != self.scenario_config.num_vehicles
            or len(joint.tau_s) != self.scenario_config.num_vehicles
            or len(joint.eta) != self.scenario_config.num_vehicles
            or len(joint.assignment) != self.scenario_config.num_vehicles
        ):
            raise ValueError("joint action shape mismatch")
        if float(sum(joint.bandwidth)) > 1.0 + self.action_tol:
            raise ValueError("bandwidth shares must sum to at most 1")

        leader_action = LeaderAction(
            prices=tuple(float(p) for p in joint.prices),
            bandwidth=tuple(float(b) for b in joint.bandwidth),
        )

        responses: list[FollowerResponse] = []
        vehicle_utilities: list[float] = []
        for v_idx, (vehicle, channel) in enumerate(
            _build_vehicle_channel_profiles(state)
        ):
            evaluations: list = []
            selected_task = int(joint.assignment[v_idx])
            allowed_tasks = (
                None if self._candidate_masks is None else self._candidate_masks[v_idx]
            )
            selected_eval = None
            bw_v = leader_action.bandwidth[v_idx]
            # Uplink rate depends only on (vehicle, channel, bandwidth_fraction)
            # — not on the task — so precompute it once per vehicle and reuse
            # across all ``num_tasks`` evaluations below.
            precomputed_rate_bps: float | None
            if bw_v > 0.0:
                try:
                    precomputed_rate_bps = uplink_rate_for_vehicle(
                        vehicle=vehicle,
                        channel=channel,
                        bandwidth_fraction=bw_v,
                        params=self.solver_params,
                    )
                except ValueError:
                    precomputed_rate_bps = None
            else:
                precomputed_rate_bps = None
            for task in state.tasks:
                # In the controlled no-analytical-response ablation the joint
                # learner retains candidate selection. An assignment outside
                # this episode's candidate mask is rejected and contributes no
                # evaluation/shaping signal, exactly as an unadmitted pair must.
                if allowed_tasks is not None and task.task_id not in allowed_tasks:
                    continue
                if task.task_id != selected_task:
                    # Full evaluation supplies one entry per task.
                    # Training shaping reads only the assigned/selected entry,
                    # so the lean path also avoids allocating V*(K-1) sentinel
                    # dataclasses.  This changes diagnostics only; reward and
                    # every shaping input are parity-tested below this layer.
                    if info_mode == "full":
                        evaluations.append(infeasible_task_evaluation(task.task_id))
                    continue
                candidate = TaskCandidate(
                    task_id=task.task_id,
                    price_per_quality_unit=leader_action.prices[task.task_id],
                    deadline_s=task.tau_max_s,
                    distance_to_roi_m=state.distance_to_roi_m[v_idx][task.task_id],
                    occlusion_factor=state.occlusion_factors[v_idx][task.task_id],
                )
                evaluation = evaluate_assigned_task(
                    task=candidate,
                    vehicle=vehicle,
                    channel=channel,
                    bandwidth_fraction=bw_v,
                    params=self.solver_params,
                    sensing_duration_s=float(joint.tau_s[v_idx]),
                    compression_ratio=float(joint.eta[v_idx]),
                    precomputed_rate_bps=precomputed_rate_bps,
                )
                # Follower individual rationality: a rational vehicle would
                # only honor an assignment whose conditional utility is
                # strictly positive. Joint-learning baselines (JDLS/MAPPO)
                # may otherwise force participation that contradicts the
                # Stackelberg follower's own incentive constraint.
                if evaluation.feasible and evaluation.utility > 0.0:
                    selected_eval = evaluation
                evaluations.append(evaluation)

            chosen_id = selected_eval.task_id if selected_eval is not None else None
            chosen_utility = float(selected_eval.utility) if selected_eval is not None else 0.0
            vehicle_utilities.append(chosen_utility)
            responses.append(
                FollowerResponse(
                    vehicle_id=vehicle.vehicle_id,
                    bandwidth_fraction=leader_action.bandwidth[v_idx],
                    selected_task_id=chosen_id,
                    selected_utility=chosen_utility,
                    selected_tau_s_s=None if selected_eval is None else selected_eval.tau_s_s,
                    selected_eta=None if selected_eval is None else selected_eval.eta,
                    selected_mu=None,
                    task_evaluations=tuple(evaluations),
                )
            )

        responses_tuple = tuple(responses)
        reward, utility_components = self._compute_reward(state, leader_action, responses_tuple)
        utility_components["vehicle_utilities"] = np.array(vehicle_utilities, dtype=np.float64)
        utility_components["total_vehicle_utility"] = float(np.sum(vehicle_utilities))
        if info_mode == "full":
            violated_constraints = self._collect_constraint_violations(
                state, responses_tuple
            )
            info = self._build_info(
                leader_action,
                responses_tuple,
                utility_components,
                violated_constraints,
            )
        else:
            # Keep this payload deliberately small.  Every field is consumed by
            # compute_joint_learning_shaping; adding evaluation-only diagnostics
            # here would reintroduce the per-step allocation cost this path
            # exists to avoid.
            info = {
                "follower_responses": responses_tuple,
                "utility_components": utility_components,
                "social_welfare": float(
                    utility_components.get("social_welfare", 0.0)
                ),
            }
        info["joint_action"] = {
            "bandwidth": np.array(joint.bandwidth, dtype=np.float64),
            "tau_s": np.array(joint.tau_s, dtype=np.float64),
            "assignment": np.array(joint.assignment, dtype=np.int64),
        }
        if info_mode == "full":
            info["joint_action"].update({
                "prices": np.array(joint.prices, dtype=np.float64),
                "eta": np.array(joint.eta, dtype=np.float64),
            })
            info["vehicle_utilities"] = np.array(
                vehicle_utilities, dtype=np.float64
            )
        info["total_vehicle_utility"] = float(np.sum(vehicle_utilities))
        self._terminated = True
        return self._observation, reward, True, False, info


def _build_vehicle_channel_profiles(
    state: ScenarioState,
) -> list[tuple[VehicleProfile, ChannelProfile]]:
    """Build paired (VehicleProfile, ChannelProfile) for every vehicle in *state*."""
    profiles: list[tuple[VehicleProfile, ChannelProfile]] = []
    for v_idx, vehicle_state in enumerate(state.vehicles):
        vehicle = VehicleProfile(
            vehicle_id=vehicle_state.vehicle_id,
            raw_data_rate_bps=vehicle_state.raw_data_rate_bps,
            resolution_coeff=vehicle_state.resolution_coeff,
            cpu_cycles_per_s=vehicle_state.cpu_cycles_per_s,
            sensor_power_w=vehicle_state.sensor_power_w,
            tx_power_w=vehicle_state.tx_power_w,
            energy_weight=vehicle_state.energy_weight,
        )
        channel_state = state.channels[v_idx]
        channel = ChannelProfile(
            distance_to_rsu_m=channel_state.distance_to_rsu_m,
            fading_power=channel_state.fading_power,
            path_loss_exponent=channel_state.path_loss_exponent,
        )
        profiles.append((vehicle, channel))
    return profiles
