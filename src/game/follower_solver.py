"""Follower-side Stage-II solver for one vehicle under fixed leader decisions.

Exactness note:
- Paper-exact: ``Psi_{i,k}``, ``Omega_{i,k}``,
  ``eta^*_{i,k}``, the projected ``tau^{s*}_{i,k}``, conditional utility
  evaluation, and the one-task-or-none selection rule.
- Paper-consistent approximation: ``mu^*_{i,k}`` is recovered numerically via
  monotone bisection. The maintained derivation gives a
  projected sensing-duration expression together with a separate bisection
  description for ``mu^*``; to keep the numerical step operational, the
  bisection residual uses the unclipped stationarity expression and the final
  sensing duration is then projected back into the feasible interval.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

from src.models.communication import (
    channel_gain,
    transmission_energy,
    transmission_latency,
    uplink_rate,
)
from src.models.computation import (
    MAX_COMPRESSION_RATIO,
    computation_energy,
    computation_latency,
    processed_data_size,
)
from src.models.constraints import latency_feasible, total_latency
from src.models.sensing_quality import (
    raw_data_volume,
    sensing_energy,
    sensing_quality,
    spatial_potential,
)
from src.models.utilities import vehicle_task_utility


@dataclass(frozen=True)
class TaskCandidate:
    task_id: int
    price_per_quality_unit: float
    deadline_s: float
    distance_to_roi_m: float
    occlusion_factor: float


@dataclass(frozen=True)
class VehicleProfile:
    vehicle_id: int
    raw_data_rate_bps: float
    resolution_coeff: float
    cpu_cycles_per_s: float
    sensor_power_w: float
    tx_power_w: float
    energy_weight: float


@dataclass(frozen=True)
class ChannelProfile:
    distance_to_rsu_m: float
    fading_power: float
    path_loss_exponent: float


@dataclass(frozen=True)
class SolverParams:
    """Parameters the follower best-response solver needs per vehicle/task.

    ``compression_override=None`` runs the
    paper's closed-form eta optimization, any float short-circuits the solver
    to that constant value. RDUS passes ``1.0`` (raw upload); the compression
    sweep passes the swept constant.
    """

    sensing_range_m: float
    sensing_decay_exponent: float
    data_efficiency_factor: float
    baseline_complexity_cycles_per_bit: float
    scaling_complexity_cycles_per_bit: float
    complexity_exponent: float
    switched_capacitance_coeff: float
    total_bandwidth_hz: float
    noise_power_w: float
    compression_ratio_max: float = MAX_COMPRESSION_RATIO
    bisection_tol: float = 1e-9
    bisection_max_iters: int = 100
    bracket_growth_factor: float = 2.0
    max_bracket_expansions: int = 100
    selection_tie_tol: float = 1e-12
    compression_override: float | None = None
    # Paper-consistent approximation for raw-upload baselines: when set, the
    # follower must contribute at least this many raw sensing bits if it serves
    # a task. This prevents the raw-upload RDUS variant from exploiting the
    # mathematical tau>=0 lower bound as a near-zero-service corner solution.
    min_service_raw_bits: float | None = None
    # Spatial-potential saturation bound forwarded to sensing_quality.
    # None permits unsaturated formula checks; formal configurations set 50.
    spatial_potential_max: float | None = None


@dataclass(frozen=True)
class TaskEvaluation:
    task_id: int
    feasible: bool
    utility: float
    tau_s_s: float | None
    eta: float | None
    mu: float | None
    quality: float | None
    sensing_latency_s: float | None
    computation_latency_s: float | None
    transmission_latency_s: float | None
    total_latency_s: float | None
    sensing_energy_j: float | None
    computation_energy_j: float | None
    transmission_energy_j: float | None


@dataclass(frozen=True)
class FollowerResponse:
    vehicle_id: int
    bandwidth_fraction: float
    selected_task_id: int | None
    selected_utility: float
    selected_tau_s_s: float | None
    selected_eta: float | None
    selected_mu: float | None
    task_evaluations: tuple[TaskEvaluation, ...]


def unit_energy_intensity(
    *,
    vehicle: VehicleProfile,
    params: SolverParams,
    uplink_rate_bps: float,
    compression_ratio: float,
) -> float:
    """Return ``Psi_{i,k}(eta)`` from the maintained manuscript equations."""

    if uplink_rate_bps <= 0.0:
        raise ValueError("uplink_rate_bps must be positive")
    if not 1.0 <= compression_ratio <= params.compression_ratio_max:
        raise ValueError("compression_ratio is outside the feasible range")
    if params.scaling_complexity_cycles_per_bit < 0.0:
        raise ValueError("scaling_complexity_cycles_per_bit must be non-negative")
    if params.complexity_exponent <= 0.0:
        raise ValueError("complexity_exponent must be positive")

    complexity = (
        params.baseline_complexity_cycles_per_bit
        + params.scaling_complexity_cycles_per_bit
        * (compression_ratio**params.complexity_exponent)
    )
    return vehicle.sensor_power_w + vehicle.raw_data_rate_bps * (
        params.switched_capacitance_coeff * (vehicle.cpu_cycles_per_s**2) * complexity
        + vehicle.tx_power_w / (compression_ratio * uplink_rate_bps)
    )


def unit_latency_intensity(
    *,
    vehicle: VehicleProfile,
    params: SolverParams,
    uplink_rate_bps: float,
    compression_ratio: float,
) -> float:
    """Return ``Omega_{i,k}(eta)`` from the maintained manuscript equations."""

    if uplink_rate_bps <= 0.0:
        raise ValueError("uplink_rate_bps must be positive")
    if not 1.0 <= compression_ratio <= params.compression_ratio_max:
        raise ValueError("compression_ratio is outside the feasible range")

    complexity = (
        params.baseline_complexity_cycles_per_bit
        + params.scaling_complexity_cycles_per_bit
        * (compression_ratio**params.complexity_exponent)
    )
    return 1.0 + vehicle.raw_data_rate_bps * (
        complexity / vehicle.cpu_cycles_per_s
        + 1.0 / (compression_ratio * uplink_rate_bps)
    )


def feasible_sensing_duration_upper_bound(
    *,
    deadline_s: float,
    vehicle: VehicleProfile,
    params: SolverParams,
    uplink_rate_bps: float,
    compression_ratio: float,
) -> float:
    """Return the C2 upper bound ``deadline / Omega(eta, w)``.

    Paper-exact: the maintained latency model is linear in sensing duration,
    ``tau_tot = tau_s * Omega(eta, w)``.  This helper only exposes the known
    feasible interval; it neither evaluates follower utility nor computes a
    best response.
    """

    if deadline_s < 0.0:
        raise ValueError("deadline_s must be non-negative")
    omega = unit_latency_intensity(
        vehicle=vehicle,
        params=params,
        uplink_rate_bps=uplink_rate_bps,
        compression_ratio=compression_ratio,
    )
    return float(deadline_s) / omega


def optimal_eta_given_mu(
    *,
    vehicle: VehicleProfile,
    params: SolverParams,
    uplink_rate_bps: float,
    mu: float,
) -> float:
    """Return the closed-form ``eta^*_{i,k}(mu)`` with the configured box bound.

    Source: the maintained manuscript equations and the bounds in ``docs/manuscript_revision_snippet.tex``.
    """

    if mu < 0.0:
        raise ValueError("mu must be non-negative")
    if uplink_rate_bps <= 0.0:
        raise ValueError("uplink_rate_bps must be positive")
    if params.compression_override is not None:
        # Baselines / sweeps short-circuit the closed-form optimum by pinning
        # eta to a constant (RDUS: 1.0 raw upload; fig7 sweep: swept value).
        return float(params.compression_override)
    if params.scaling_complexity_cycles_per_bit <= 0.0:
        raise ValueError("scaling_complexity_cycles_per_bit must be positive")
    if params.complexity_exponent <= 0.0:
        raise ValueError("complexity_exponent must be positive")

    numerator = vehicle.energy_weight * vehicle.tx_power_w + mu
    denominator = (
        params.complexity_exponent
        * params.scaling_complexity_cycles_per_bit
        * uplink_rate_bps
        * (
            vehicle.energy_weight
            * params.switched_capacitance_coeff
            * (vehicle.cpu_cycles_per_s**2)
            + mu / vehicle.cpu_cycles_per_s
        )
    )
    if denominator <= 0.0:
        raise ValueError("eta closed-form denominator must be positive")

    eta_unconstrained = (numerator / denominator) ** (1.0 / (params.complexity_exponent + 1.0))
    return min(params.compression_ratio_max, max(1.0, eta_unconstrained))


def _stationary_tau_candidate(
    *,
    task: TaskCandidate,
    vehicle: VehicleProfile,
    params: SolverParams,
    spatial_potential_value: float,
    uplink_rate_bps: float,
    compression_ratio: float,
    mu: float,
    omega: float | None = None,
) -> float:
    """Return the unclipped stationarity expression used in the ``tau^s`` solve.

    ``omega`` (the unit latency intensity at ``compression_ratio``) may be passed
    in by callers that already computed it — :func:`solve_latency_multiplier`'s
    ``residual`` and :func:`optimal_tau_given_mu` both need it themselves, and
    recomputing it here doubled the (hot) :func:`unit_latency_intensity` call
    count. ``unit_latency_intensity`` is a deterministic pure function of its
    arguments, so reusing the caller's value is bit-for-bit identical.
    """

    if (
        task.price_per_quality_unit <= 0.0
        or spatial_potential_value <= 0.0
        or params.data_efficiency_factor <= 0.0
        or vehicle.raw_data_rate_bps <= 0.0
    ):
        return 0.0

    psi = unit_energy_intensity(
        vehicle=vehicle,
        params=params,
        uplink_rate_bps=uplink_rate_bps,
        compression_ratio=compression_ratio,
    )
    if omega is None:
        omega = unit_latency_intensity(
            vehicle=vehicle,
            params=params,
            uplink_rate_bps=uplink_rate_bps,
            compression_ratio=compression_ratio,
        )
    denominator = vehicle.energy_weight * psi + mu * omega
    if denominator <= 0.0:
        return math.inf

    return (
        (task.price_per_quality_unit * spatial_potential_value) / denominator
        - 1.0 / (params.data_efficiency_factor * vehicle.raw_data_rate_bps)
    )


def minimum_service_sensing_duration(
    *,
    vehicle: VehicleProfile,
    params: SolverParams,
) -> float:
    """Return the configured raw-service lower bound as a sensing duration.

    Exactness label: paper-consistent approximation. The paper defines the
    follower sensing-duration box with a zero lower bound; this optional floor
    is only used for raw-upload diagnostic baselines whose intended semantics
    require nontrivial uncompressed service.
    """

    if params.min_service_raw_bits is None:
        return 0.0
    if params.min_service_raw_bits <= 0.0:
        return 0.0
    if vehicle.raw_data_rate_bps <= 0.0:
        raise ValueError("raw_data_rate_bps must be positive")
    return float(params.min_service_raw_bits) / vehicle.raw_data_rate_bps


def optimal_tau_given_mu(
    *,
    task: TaskCandidate,
    vehicle: VehicleProfile,
    params: SolverParams,
    spatial_potential_value: float,
    uplink_rate_bps: float,
    compression_ratio: float,
    mu: float,
) -> float:
    """Return the projected closed-form ``tau^{s*}_{i,k}(mu)``.

    Source: the maintained manuscript equations.
    """

    if mu < 0.0:
        raise ValueError("mu must be non-negative")
    if task.deadline_s < 0.0:
        raise ValueError("deadline_s must be non-negative")

    omega = unit_latency_intensity(
        vehicle=vehicle,
        params=params,
        uplink_rate_bps=uplink_rate_bps,
        compression_ratio=compression_ratio,
    )
    lower_bound = minimum_service_sensing_duration(vehicle=vehicle, params=params)
    upper_bound = task.deadline_s / omega
    stationary = _stationary_tau_candidate(
        task=task,
        vehicle=vehicle,
        params=params,
        spatial_potential_value=spatial_potential_value,
        uplink_rate_bps=uplink_rate_bps,
        compression_ratio=compression_ratio,
        mu=mu,
        omega=omega,  # reuse — avoids a redundant unit_latency_intensity call
    )
    return min(max(stationary, lower_bound), upper_bound)


def uplink_rate_for_vehicle(
    *,
    vehicle: VehicleProfile,
    channel: ChannelProfile,
    bandwidth_fraction: float,
    params: SolverParams,
) -> float:
    gain = channel_gain(
        fading_power=channel.fading_power,
        distance_to_rsu_m=channel.distance_to_rsu_m,
        path_loss_exponent=channel.path_loss_exponent,
    )
    return uplink_rate(
        bandwidth_fraction=bandwidth_fraction,
        total_bandwidth_hz=params.total_bandwidth_hz,
        tx_power_w=vehicle.tx_power_w,
        channel_gain_value=gain,
        noise_power_w=params.noise_power_w,
    )


def solve_latency_multiplier(
    *,
    task: TaskCandidate,
    vehicle: VehicleProfile,
    params: SolverParams,
    spatial_potential_value: float,
    uplink_rate_bps: float,
) -> float | None:
    """Solve ``mu^*`` via monotone bisection under the numerical assumption in
    ``docs/revision_experiment_protocol.md``.

    Returns the optimal multiplier, or ``None`` when the deadline cannot be
    satisfied at any finite ``mu`` (the bisection failed to bracket a root
    after :attr:`SolverParams.max_bracket_expansions` iterations). Callers
    interpret ``None`` as "this (vehicle, task, bandwidth) tuple is latency-
    infeasible" and typically short-circuit to :func:`_infeasible_evaluation`.
    """

    if uplink_rate_bps <= 0.0:
        raise ValueError("uplink_rate_bps must be positive")

    def residual(mu: float) -> float:
        eta = optimal_eta_given_mu(
            vehicle=vehicle,
            params=params,
            uplink_rate_bps=uplink_rate_bps,
            mu=mu,
        )
        omega = unit_latency_intensity(
            vehicle=vehicle,
            params=params,
            uplink_rate_bps=uplink_rate_bps,
            compression_ratio=eta,
        )
        tau_stationary = _stationary_tau_candidate(
            task=task,
            vehicle=vehicle,
            params=params,
            spatial_potential_value=spatial_potential_value,
            uplink_rate_bps=uplink_rate_bps,
            compression_ratio=eta,
            mu=mu,
            omega=omega,  # reuse — avoids a redundant unit_latency_intensity call
        )
        return tau_stationary * omega - task.deadline_s

    residual_at_zero = residual(0.0)
    if residual_at_zero <= params.bisection_tol:
        return 0.0

    lower = 0.0
    upper = 1.0
    upper_residual = residual(upper)
    expansions = 0
    while upper_residual > 0.0 and expansions < params.max_bracket_expansions:
        lower = upper
        upper *= params.bracket_growth_factor
        upper_residual = residual(upper)
        expansions += 1

    if upper_residual > 0.0:
        # Could not bracket a root within the allotted expansions: this task
        # is infeasible for this vehicle at the current bandwidth. Return
        # ``None`` instead of raising so the caller can surface an
        # ``_infeasible_evaluation`` directly without the exception round-trip.
        return None

    for _ in range(params.bisection_max_iters):
        mid = 0.5 * (lower + upper)
        mid_residual = residual(mid)
        if abs(mid_residual) <= params.bisection_tol or (upper - lower) <= params.bisection_tol:
            return mid
        if mid_residual > 0.0:
            lower = mid
        else:
            upper = mid
    # Return the final midpoint if the iteration cap is exhausted. The warning
    # reports residual and bracket width for numerical verification.
    final_mid = 0.5 * (lower + upper)
    warnings.warn(
        "solve_latency_multiplier hit bisection_max_iters="
        f"{params.bisection_max_iters} without converging "
        f"(|bracket|={upper - lower:.3e}, residual={residual(final_mid):.3e}); "
        "returning midpoint as a best-effort fallback.",
        RuntimeWarning,
        stacklevel=2,
    )
    return final_mid


def _infeasible_evaluation(task_id: int) -> TaskEvaluation:
    return TaskEvaluation(
        task_id=task_id,
        feasible=False,
        utility=-math.inf,
        tau_s_s=None,
        eta=None,
        mu=None,
        quality=None,
        sensing_latency_s=None,
        computation_latency_s=None,
        transmission_latency_s=None,
        total_latency_s=None,
        sensing_energy_j=None,
        computation_energy_j=None,
        transmission_energy_j=None,
    )


def infeasible_task_evaluation(task_id: int) -> TaskEvaluation:
    """Return the canonical infeasible sentinel for a known-unselected task.

    Joint-learning policies choose exactly one task per vehicle.  Their
    environment can use this public constructor for the other tasks instead
    of entering :func:`evaluate_assigned_task` with a zero sensing duration.
    """

    return _infeasible_evaluation(task_id)


def solve_task_response(
    task: TaskCandidate,
    vehicle: VehicleProfile,
    channel: ChannelProfile,
    bandwidth_fraction: float,
    params: SolverParams,
    *,
    precomputed_rate_bps: float | None = None,
) -> TaskEvaluation:
    """Solve the fixed-task follower subproblem for one candidate task."""

    if bandwidth_fraction <= 0.0:
        return _infeasible_evaluation(task.task_id)

    try:
        spatial = spatial_potential(
            resolution_coeff=vehicle.resolution_coeff,
            occlusion_factor=task.occlusion_factor,
            distance_to_roi_m=task.distance_to_roi_m,
            sensing_range_m=params.sensing_range_m,
            sensing_decay_exponent=params.sensing_decay_exponent,
            max_value=params.spatial_potential_max,
        )
        if precomputed_rate_bps is not None:
            rate_bps = precomputed_rate_bps
        else:
            rate_bps = uplink_rate_for_vehicle(
                vehicle=vehicle,
                channel=channel,
                bandwidth_fraction=bandwidth_fraction,
                params=params,
            )
        mu = solve_latency_multiplier(
            task=task,
            vehicle=vehicle,
            params=params,
            spatial_potential_value=spatial,
            uplink_rate_bps=rate_bps,
        )
        if mu is None:
            return _infeasible_evaluation(task.task_id)
        eta = optimal_eta_given_mu(
            vehicle=vehicle,
            params=params,
            uplink_rate_bps=rate_bps,
            mu=mu,
        )
        tau_s = optimal_tau_given_mu(
            task=task,
            vehicle=vehicle,
            params=params,
            spatial_potential_value=spatial,
            uplink_rate_bps=rate_bps,
            compression_ratio=eta,
            mu=mu,
        )
        if tau_s < minimum_service_sensing_duration(vehicle=vehicle, params=params):
            return _infeasible_evaluation(task.task_id)
        quality = sensing_quality(
            resolution_coeff=vehicle.resolution_coeff,
            occlusion_factor=task.occlusion_factor,
            distance_to_roi_m=task.distance_to_roi_m,
            sensing_range_m=params.sensing_range_m,
            sensing_decay_exponent=params.sensing_decay_exponent,
            sensing_duration_s=tau_s,
            data_efficiency_factor=params.data_efficiency_factor,
            raw_data_rate_bps=vehicle.raw_data_rate_bps,
            spatial_potential_max=params.spatial_potential_max,
            spatial_potential_value=spatial,  # reuse — already computed above for the mu solve
        )
        sense_energy = sensing_energy(vehicle.sensor_power_w, tau_s)
        comp_latency = computation_latency(
            raw_data_rate_bps=vehicle.raw_data_rate_bps,
            sensing_duration_s=tau_s,
            compression_ratio=eta,
            baseline_complexity_cycles_per_bit=params.baseline_complexity_cycles_per_bit,
            scaling_complexity_cycles_per_bit=params.scaling_complexity_cycles_per_bit,
            complexity_exponent=params.complexity_exponent,
            cpu_cycles_per_s=vehicle.cpu_cycles_per_s,
        )
        comp_energy = computation_energy(
            raw_data_rate_bps=vehicle.raw_data_rate_bps,
            sensing_duration_s=tau_s,
            compression_ratio=eta,
            baseline_complexity_cycles_per_bit=params.baseline_complexity_cycles_per_bit,
            scaling_complexity_cycles_per_bit=params.scaling_complexity_cycles_per_bit,
            complexity_exponent=params.complexity_exponent,
            switched_capacitance_coeff=params.switched_capacitance_coeff,
            cpu_cycles_per_s=vehicle.cpu_cycles_per_s,
        )
        processed_bits = processed_data_size(
            raw_data_volume_bits=raw_data_volume(
                raw_data_rate_bps=vehicle.raw_data_rate_bps,
                sensing_duration_s=tau_s,
            ),
            compression_ratio=eta,
        )
        tx_latency = transmission_latency(processed_bits, rate_bps)
        tx_energy = transmission_energy(vehicle.tx_power_w, tx_latency)
        total_latency_s = total_latency(tau_s, comp_latency, tx_latency)
        feasible = latency_feasible(
            sensing_latency_s=tau_s,
            computation_latency_s=comp_latency,
            transmission_latency_s=tx_latency,
            deadline_s=task.deadline_s + params.bisection_tol,
        )
        utility = vehicle_task_utility(
            price_per_quality_unit=task.price_per_quality_unit,
            sensing_quality_value=quality,
            energy_weight=vehicle.energy_weight,
            sensing_energy_j=sense_energy,
            computation_energy_j=comp_energy,
            transmission_energy_j=tx_energy,
        )
    except ValueError:
        return _infeasible_evaluation(task.task_id)

    if not feasible:
        return _infeasible_evaluation(task.task_id)

    return TaskEvaluation(
        task_id=task.task_id,
        feasible=True,
        utility=utility,
        tau_s_s=tau_s,
        eta=eta,
        mu=mu,
        quality=quality,
        sensing_latency_s=tau_s,
        computation_latency_s=comp_latency,
        transmission_latency_s=tx_latency,
        total_latency_s=total_latency_s,
        sensing_energy_j=sense_energy,
        computation_energy_j=comp_energy,
        transmission_energy_j=tx_energy,
    )


def evaluate_assigned_task(
    *,
    task: TaskCandidate,
    vehicle: VehicleProfile,
    channel: ChannelProfile,
    bandwidth_fraction: float,
    params: SolverParams,
    sensing_duration_s: float,
    compression_ratio: float,
    precomputed_rate_bps: float | None = None,
) -> TaskEvaluation:
    """Score a follower assignment without invoking the best-response optimizer.

    Used by joint-learning baselines (JDLS / MAPPO) where the assignment, sensing
    duration, and compression ratio are produced directly by a neural policy.

    Callers that already derived the uplink rate for this (vehicle, bandwidth)
    pair (for example, :meth:`RSULeaderEnv.step_joint` evaluating every task for
    the same vehicle) can pass it via ``precomputed_rate_bps`` to skip a
    redundant ``uplink_rate`` evaluation per task.
    """

    if bandwidth_fraction <= 0.0:
        return _infeasible_evaluation(task.task_id)
    eta = float(min(max(compression_ratio, 1.0), params.compression_ratio_max))
    tau_s = float(max(sensing_duration_s, 0.0))
    if tau_s <= 0.0:
        return _infeasible_evaluation(task.task_id)
    if tau_s < minimum_service_sensing_duration(vehicle=vehicle, params=params):
        return _infeasible_evaluation(task.task_id)
    try:
        if precomputed_rate_bps is not None:
            rate_bps = float(precomputed_rate_bps)
        else:
            rate_bps = uplink_rate_for_vehicle(
                vehicle=vehicle,
                channel=channel,
                bandwidth_fraction=bandwidth_fraction,
                params=params,
            )
        quality = sensing_quality(
            resolution_coeff=vehicle.resolution_coeff,
            occlusion_factor=task.occlusion_factor,
            distance_to_roi_m=task.distance_to_roi_m,
            sensing_range_m=params.sensing_range_m,
            sensing_decay_exponent=params.sensing_decay_exponent,
            sensing_duration_s=tau_s,
            data_efficiency_factor=params.data_efficiency_factor,
            raw_data_rate_bps=vehicle.raw_data_rate_bps,
            spatial_potential_max=params.spatial_potential_max,
        )
        sense_energy = sensing_energy(vehicle.sensor_power_w, tau_s)
        comp_latency = computation_latency(
            raw_data_rate_bps=vehicle.raw_data_rate_bps,
            sensing_duration_s=tau_s,
            compression_ratio=eta,
            baseline_complexity_cycles_per_bit=params.baseline_complexity_cycles_per_bit,
            scaling_complexity_cycles_per_bit=params.scaling_complexity_cycles_per_bit,
            complexity_exponent=params.complexity_exponent,
            cpu_cycles_per_s=vehicle.cpu_cycles_per_s,
        )
        comp_energy = computation_energy(
            raw_data_rate_bps=vehicle.raw_data_rate_bps,
            sensing_duration_s=tau_s,
            compression_ratio=eta,
            baseline_complexity_cycles_per_bit=params.baseline_complexity_cycles_per_bit,
            scaling_complexity_cycles_per_bit=params.scaling_complexity_cycles_per_bit,
            complexity_exponent=params.complexity_exponent,
            switched_capacitance_coeff=params.switched_capacitance_coeff,
            cpu_cycles_per_s=vehicle.cpu_cycles_per_s,
        )
        processed_bits = processed_data_size(
            raw_data_volume_bits=raw_data_volume(
                raw_data_rate_bps=vehicle.raw_data_rate_bps,
                sensing_duration_s=tau_s,
            ),
            compression_ratio=eta,
        )
        tx_latency = transmission_latency(processed_bits, rate_bps)
        tx_energy = transmission_energy(vehicle.tx_power_w, tx_latency)
        total_latency_s = total_latency(tau_s, comp_latency, tx_latency)
        feasible = latency_feasible(
            sensing_latency_s=tau_s,
            computation_latency_s=comp_latency,
            transmission_latency_s=tx_latency,
            deadline_s=task.deadline_s + params.bisection_tol,
        )
        utility = vehicle_task_utility(
            price_per_quality_unit=task.price_per_quality_unit,
            sensing_quality_value=quality,
            energy_weight=vehicle.energy_weight,
            sensing_energy_j=sense_energy,
            computation_energy_j=comp_energy,
            transmission_energy_j=tx_energy,
        )
    except ValueError:
        return _infeasible_evaluation(task.task_id)

    return TaskEvaluation(
        task_id=task.task_id,
        feasible=feasible,
        utility=utility,
        tau_s_s=tau_s,
        eta=eta,
        mu=None,
        quality=quality,
        sensing_latency_s=tau_s,
        computation_latency_s=comp_latency,
        transmission_latency_s=tx_latency,
        total_latency_s=total_latency_s,
        sensing_energy_j=sense_energy,
        computation_energy_j=comp_energy,
        transmission_energy_j=tx_energy,
    )


def select_best_task(
    task_evaluations: tuple[TaskEvaluation, ...] | list[TaskEvaluation],
    tie_tol: float = 1e-12,
) -> int | None:
    """Select the highest-positive-utility task with smallest-id tie-breaking."""

    best_task_id: int | None = None
    best_utility = 0.0
    for evaluation in task_evaluations:
        if not evaluation.feasible:
            continue
        if evaluation.utility <= 0.0:
            continue
        if best_task_id is None:
            best_task_id = evaluation.task_id
            best_utility = evaluation.utility
            continue
        if evaluation.utility > best_utility + tie_tol:
            best_task_id = evaluation.task_id
            best_utility = evaluation.utility
        elif abs(evaluation.utility - best_utility) <= tie_tol and evaluation.task_id < best_task_id:
            best_task_id = evaluation.task_id
            best_utility = evaluation.utility
    return best_task_id


def solve_vehicle_best_response(
    task_candidates: tuple[TaskCandidate, ...] | list[TaskCandidate],
    vehicle: VehicleProfile,
    channel: ChannelProfile,
    bandwidth_fraction: float,
    params: SolverParams,
) -> FollowerResponse:
    """Evaluate all candidate tasks and choose at most one positive-utility task."""

    rate_bps: float | None = None
    if bandwidth_fraction > 0.0:
        try:
            rate_bps = uplink_rate_for_vehicle(
                vehicle=vehicle,
                channel=channel,
                bandwidth_fraction=bandwidth_fraction,
                params=params,
            )
        except ValueError:
            rate_bps = None
    evaluations = tuple(
        solve_task_response(
            task=task,
            vehicle=vehicle,
            channel=channel,
            bandwidth_fraction=bandwidth_fraction,
            params=params,
            precomputed_rate_bps=rate_bps,
        )
        for task in task_candidates
    )
    selected_task_id = select_best_task(evaluations, tie_tol=params.selection_tie_tol)
    selected_eval = next((evaluation for evaluation in evaluations if evaluation.task_id == selected_task_id), None)

    return FollowerResponse(
        vehicle_id=vehicle.vehicle_id,
        bandwidth_fraction=bandwidth_fraction,
        selected_task_id=selected_task_id,
        selected_utility=0.0 if selected_eval is None else selected_eval.utility,
        selected_tau_s_s=None if selected_eval is None else selected_eval.tau_s_s,
        selected_eta=None if selected_eval is None else selected_eval.eta,
        selected_mu=None if selected_eval is None else selected_eval.mu,
        task_evaluations=evaluations,
    )
