"""Scenario sampling for one scheduling cycle.

Exactness label: paper-consistent approximation for synthetic geometry,
occlusion, channel, and compute-capacity distributions. Assumptions are
maintained in docs/revision_experiment_protocol.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np

from src.game.follower_solver import SolverParams
from src.sim.state import ChannelState, ScenarioState, TaskState, VehicleState


def dbm_to_watts(dbm: float) -> float:
    """Convert dBm to watts."""

    return 10.0 ** ((dbm - 30.0) / 10.0)


@dataclass(frozen=True)
class ScenarioConfig:
    num_vehicles: int = 20
    scene_width_m: float = 240.0
    scene_height_m: float = 240.0
    intersection_center_m: tuple[float, float] = (120.0, 120.0)
    road_width_m: float = 12.0
    core_half_size_m: float = 12.0
    rsu_position_m: tuple[float, float] = (138.0, 138.0)
    vehicle_zone_probabilities: tuple[float, float, float, float, float] = (
        0.22,
        0.22,
        0.22,
        0.22,
        0.12,
    )
    # Maintained model: shortest Euclidean distance from the
    # vehicle to the rectangular RoI.
    roi_distance_model: str = "rectangle"
    # Lower bound for vehicle-to-RoI distance; keeps 1/d^ζ finite when a
    # vehicle sits exactly at a task centre.
    distance_floor_m: float = 1.0
    task_length_range_m: tuple[float, float] = (15.0, 35.0)
    task_width_range_m: tuple[float, float] = (4.0, 8.0)
    min_task_distance_m: float = 40.0
    urgency_profiles: tuple[tuple[str, float, float], ...] = (
        ("high", 3.0, 0.1),
        ("mid", 2.0, 0.3),
        ("low", 1.0, 0.5),
    )
    system_bandwidth_hz: float = 20_000_000.0
    bandwidth_cost_coeff_per_hz: float = 0.1 / 1_000_000.0
    sensing_range_m: float = 80.0
    sensing_decay_exponent: float = 2.0
    data_efficiency_factor: float = 1e-6
    baseline_complexity_cycles_per_bit: float = 50.0
    scaling_complexity_cycles_per_bit: float = 2.0
    complexity_exponent: float = 2.5
    # Kappa scales computation energy independently of computation latency.
    # Together with transmit power it determines the compression energy tradeoff.
    switched_capacitance_coeff: float = 3e-31
    # Generic example default. Formal experiments override this
    # explicitly to 0.3 in every algorithm YAML.
    pricing_budget_max: float = 2.5
    path_loss_exponent: float = 3.5
    noise_power_w: float = dbm_to_watts(-95.0)
    tx_power_w: float = dbm_to_watts(37.0)
    raw_data_rate_bps: float = 40_000_000.0
    sensor_power_range_w: tuple[float, float] = (5.0, 10.0)
    resolution_coeff_range: tuple[float, float] = (10.0, 15.0)
    cpu_cycles_per_s_range: tuple[float, float] = (10_000_000_000.0, 20_000_000_000.0)
    # Optional episode-level CPU regime mixture used only by the dedicated
    # CPU robustness experiment. The default ``None`` path deliberately
    # consumes no extra RNG draw.
    # Exactness label: paper-consistent approximation; see the maintained
    # experiment protocol.
    cpu_cycles_per_s_regimes: tuple[tuple[float, float], ...] | None = None
    cpu_regime_probabilities: tuple[float, ...] | None = None
    energy_weight_range: tuple[float, float] = (2.0, 8.0)
    # Default samples occlusion ~Uniform(0,1). Narrower ranges are used by
    # robustness sweeps while keeping per-vehicle heterogeneity.
    occlusion_range: tuple[float, float] = (0.0, 1.0)
    diversity_factor: float = 2.0
    compression_ratio_max: float = 50.0
    # Bisection tolerance on the Lagrange multiplier mu. 1e-9 is tight enough
    # that the derived tau/eta round-trip matches the paper's closed-form
    # optimum to double precision on the default scenario (verified in tests).
    bisection_tol: float = 1e-9
    # bisection_max_iters×bracket_growth_factor^max_bracket_expansions gives
    # ~2^100 headroom; in practice the inner loop converges in <30 iters on
    # non-pathological inputs. Exhausting the iteration cap emits a warning.
    bisection_max_iters: int = 100
    bracket_growth_factor: float = 2.0
    max_bracket_expansions: int = 100
    # Tolerance used when breaking ties between two follower best-responses
    # whose utilities differ only in float noise. 1e-12 is well under the
    # numerical precision of the cost/utility computations themselves.
    selection_tie_tol: float = 1e-12
    # ``raw_upload_mode`` and ``fixed_compression_ratio`` are UX-level flags:
    # the solver only knows about a single ``compression_override``. ``make_solver_params``
    # reconciles the two (``raw_upload_mode=True`` dominates ``fixed_compression_ratio``).
    raw_upload_mode: bool = False
    fixed_compression_ratio: float | None = None
    # Optional raw-service lower bound consumed by the follower solver. Default
    # off for paper-exact TE-PPO; RDUS enables it to avoid near-zero raw-upload
    # corner service that is mathematically feasible but not a meaningful
    # baseline implementation.
    min_service_raw_bits: float | None = None
    # Optional sensor-saturation bound. Generic formula diagnostics may leave
    # it disabled; every formal revision algorithm YAML fixes it to 50.
    spatial_potential_max: float | None = None
    # --- Coverage-aware candidate pruning -----------------------------------
    # Default OFF preserves the generic environment. The maintained TE-PPO
    # configuration enables it explicitly; see the revision experiment
    # protocol for the candidate-cap definition and sensitivity settings.
    enable_candidate_pruning: bool = False
    # Optional pre-filters applied before the top-N_max ranking.
    pruning_occlusion_max: float | None = None
    # r_0's admission role. None disables distance pre-filtering even when the
    # top-N cap is enabled; the full TE-PPO config sets it to 80 m.
    candidate_pruning_radius_m: float | None = None
    # Candidate-cap coverage scale. None uses sensing_range_m; formal
    # sensitivity experiments set it independently of sensing quality.
    candidate_limit_range_m: float | None = None
    # How far each intersection arm's task corridor extends outward from the
    # core, in metres. ``None`` uses ``task_length_range_m[1]`` (35 m), which
    # defines the default corridors. Sweeps over the task count set this to the
    # full road extent (108 m) so each arm can be subdivided into enough
    # non-overlapping regions to host K > 4 tasks; see
    # ``_task_region_rectangles``.
    task_region_span_m: float | None = None

    def __post_init__(self) -> None:
        regimes = self.cpu_cycles_per_s_regimes
        probabilities = self.cpu_regime_probabilities
        if regimes is None:
            if probabilities is not None:
                raise ValueError(
                    "cpu_regime_probabilities requires cpu_cycles_per_s_regimes"
                )
            return
        if not regimes:
            raise ValueError("cpu_cycles_per_s_regimes must not be empty")
        for low, high in regimes:
            if not math.isfinite(float(low)) or not math.isfinite(float(high)):
                raise ValueError("CPU regime endpoints must be finite")
            if float(low) <= 0.0 or float(high) <= float(low):
                raise ValueError("each CPU regime must satisfy 0 < low < high")
        if probabilities is None:
            return
        if len(probabilities) != len(regimes):
            raise ValueError("CPU regime probabilities must match the regime count")
        parsed = tuple(float(value) for value in probabilities)
        if any(not math.isfinite(value) or value < 0.0 for value in parsed):
            raise ValueError("CPU regime probabilities must be finite and nonnegative")
        if not math.isclose(sum(parsed), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("CPU regime probabilities must sum to 1")

    @property
    def num_tasks(self) -> int:
        return len(self.urgency_profiles)

    @property
    def compression_override(self) -> float | None:
        """Resolve the two UX knobs into the single field the solver consumes."""

        if self.raw_upload_mode:
            return 1.0
        if self.fixed_compression_ratio is not None:
            return float(self.fixed_compression_ratio)
        return None


ZONE_WEST: Final[int] = 0
ZONE_EAST: Final[int] = 1
ZONE_SOUTH: Final[int] = 2
ZONE_NORTH: Final[int] = 3
ZONE_CORE: Final[int] = 4


def make_solver_params(config: ScenarioConfig) -> SolverParams:
    """Build follower-solver parameters from scenario defaults."""

    return SolverParams(
        sensing_range_m=config.sensing_range_m,
        sensing_decay_exponent=config.sensing_decay_exponent,
        data_efficiency_factor=config.data_efficiency_factor,
        baseline_complexity_cycles_per_bit=config.baseline_complexity_cycles_per_bit,
        scaling_complexity_cycles_per_bit=config.scaling_complexity_cycles_per_bit,
        complexity_exponent=config.complexity_exponent,
        switched_capacitance_coeff=config.switched_capacitance_coeff,
        total_bandwidth_hz=config.system_bandwidth_hz,
        noise_power_w=config.noise_power_w,
        compression_ratio_max=config.compression_ratio_max,
        bisection_tol=config.bisection_tol,
        bisection_max_iters=config.bisection_max_iters,
        bracket_growth_factor=config.bracket_growth_factor,
        max_bracket_expansions=config.max_bracket_expansions,
        selection_tie_tol=config.selection_tie_tol,
        compression_override=config.compression_override,
        min_service_raw_bits=config.min_service_raw_bits,
        spatial_potential_max=config.spatial_potential_max,
    )


def candidate_limit(task_length_m: float, sensing_range_m: float, diversity_factor: float) -> int:
    r"""Return the coverage-segment candidate cap.

    .. math::

       S_k=\max(1,\lceil a_k/(2r_0)\rceil),\qquad
       N_{\max}^{(k)}=\max(2,\lceil\varrho S_k\rceil).

    Count coverage segments before applying the diversity factor.

    Exactness label: paper-exact; see
    ``docs/revision_experiment_protocol.md``.
    """

    if task_length_m <= 0.0:
        raise ValueError("task_length_m must be positive")
    if sensing_range_m <= 0.0:
        raise ValueError("sensing_range_m must be positive")
    if diversity_factor <= 0.0:
        raise ValueError("diversity_factor must be positive")
    coverage_segments = max(1, math.ceil(task_length_m / (2.0 * sensing_range_m)))
    return max(2, math.ceil(diversity_factor * coverage_segments))


def _sample_point_in_rect(
    rng: np.random.Generator,
    *,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> tuple[float, float]:
    return (float(rng.uniform(xmin, xmax)), float(rng.uniform(ymin, ymax)))


def _vehicle_zone_rectangles(config: ScenarioConfig) -> tuple[tuple[float, float, float, float], ...]:
    cx, cy = config.intersection_center_m
    half_road = config.road_width_m / 2.0
    core_min_x = cx - config.core_half_size_m
    core_max_x = cx + config.core_half_size_m
    core_min_y = cy - config.core_half_size_m
    core_max_y = cy + config.core_half_size_m
    return (
        (0.0, core_min_x, cy - half_road, cy + half_road),
        (core_max_x, config.scene_width_m, cy - half_road, cy + half_road),
        (cx - half_road, cx + half_road, 0.0, core_min_y),
        (cx - half_road, cx + half_road, core_max_y, config.scene_height_m),
        (core_min_x, core_max_x, core_min_y, core_max_y),
    )


def _task_region_rectangles(config: ScenarioConfig) -> tuple[tuple[float, float, float, float], ...]:
    cx, cy = config.intersection_center_m
    half_road = config.road_width_m / 2.0
    # Corridor reach. Defaults to the longest task so a task always fits;
    # widened via task_region_span_m when more regions are needed for K > 4.
    max_task_length = (
        config.task_region_span_m
        if config.task_region_span_m is not None
        else config.task_length_range_m[1]
    )
    core_min_x = cx - config.core_half_size_m
    core_max_x = cx + config.core_half_size_m
    core_min_y = cy - config.core_half_size_m
    core_max_y = cy + config.core_half_size_m
    return (
        (max(0.0, core_min_x - max_task_length), core_min_x, cy - half_road, cy + half_road),
        (core_max_x, min(config.scene_width_m, core_max_x + max_task_length), cy - half_road, cy + half_road),
        (cx - half_road, cx + half_road, max(0.0, core_min_y - max_task_length), core_min_y),
        (cx - half_road, cx + half_road, core_max_y, min(config.scene_height_m, core_max_y + max_task_length)),
    )


def _subdivided_task_regions(
    config: ScenarioConfig,
) -> tuple[tuple[float, float, float, float], ...]:
    """Return enough disjoint task regions to host ``config.num_tasks``.

    The four arm corridors are split along their long axis into
    ``ceil(K / 4)`` equal cells, giving ``4 * ceil(K / 4) >= K`` regions that
    ``_sample_task_centers`` then draws from without replacement.

    With ``K <= 4`` this returns the four corridors unchanged, and the number
    of RNG draws is identical, so a default scenario reproduces the published
    task centres (and every downstream vehicle position) bit for bit.
    """
    corridors = _task_region_rectangles(config)
    subdivisions = math.ceil(config.num_tasks / len(corridors))
    if subdivisions <= 1:
        return corridors

    # A centre may land anywhere inside its cell, so each cell is inset by
    # exactly half the longest possible RoI.  Adjacent centre intervals are
    # then separated by at least that full length.  This conservative rule
    # avoids introducing an undocumented RoI-orientation assumption.
    half_longest = config.task_length_range_m[1] / 2.0

    regions: list[tuple[float, float, float, float]] = []
    for xmin, xmax, ymin, ymax in corridors:
        # Split along whichever axis the corridor actually runs along.
        if (xmax - xmin) >= (ymax - ymin):
            step = (xmax - xmin) / subdivisions
            if step < 2.0 * half_longest:
                raise ValueError(
                    "task corridor cell is shorter than the maximum RoI length"
                )
            inset = half_longest
            regions.extend(
                (
                    xmin + i * step + inset,
                    xmin + (i + 1) * step - inset,
                    ymin,
                    ymax,
                )
                for i in range(subdivisions)
            )
        else:
            step = (ymax - ymin) / subdivisions
            if step < 2.0 * half_longest:
                raise ValueError(
                    "task corridor cell is shorter than the maximum RoI length"
                )
            inset = half_longest
            regions.extend(
                (
                    xmin,
                    xmax,
                    ymin + i * step + inset,
                    ymin + (i + 1) * step - inset,
                )
                for i in range(subdivisions)
            )
    return tuple(regions)


def make_urgency_profiles(num_tasks: int) -> tuple[tuple[str, float, float], ...]:
    """Cycle the paper's three urgency classes out to ``num_tasks`` entries.

    The labels stay exactly ``high`` / ``mid`` / ``low`` on purpose.
    ``episode_metrics_from_info`` buckets task selections by matching those
    three strings and has no fallback branch, so a fourth label would be
    counted in none of the ``task_selection_*`` columns -- silently, with no
    error. Reusing the three keeps the metric schema unchanged and keeps each
    column meaning "selections of tasks at this urgency level", which is still
    the right reading when several tasks share a level.

    Because of that, ``num_tasks`` should be a multiple of three so the urgency
    mix stays balanced; K in {3, 6, 9, 12} is what the scalability sweep uses.
    """
    if num_tasks < 1:
        raise ValueError("num_tasks must be positive")
    base = (("high", 3.0, 0.1), ("mid", 2.0, 0.3), ("low", 1.0, 0.5))
    return tuple(base[i % len(base)] for i in range(num_tasks))


def _sample_vehicle_position(
    rng: np.random.Generator,
    config: ScenarioConfig,
    *,
    rectangles: tuple[tuple[float, float, float, float], ...] | None = None,
    zone_probs: np.ndarray | None = None,
) -> tuple[float, float]:
    # ``rectangles``/``zone_probs`` depend only on ``config`` (no RNG), so the
    # caller (``sample_scenario``) precomputes them once instead of rebuilding
    # them per vehicle. They do not touch the RNG draw sequence, so passing them
    # evaluates the same deterministic geometry as the scalar calculation.
    if rectangles is None:
        rectangles = _vehicle_zone_rectangles(config)
    if zone_probs is None:
        zone_probs = np.array(config.vehicle_zone_probabilities, dtype=np.float64)
    zone = int(rng.choice(len(rectangles), p=zone_probs))
    xmin, xmax, ymin, ymax = rectangles[zone]
    return _sample_point_in_rect(rng, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)


def _sample_task_centers(rng: np.random.Generator, config: ScenarioConfig) -> tuple[tuple[float, float], ...]:
    rectangles = _subdivided_task_regions(config)
    region_indices = rng.choice(len(rectangles), size=config.num_tasks, replace=False)
    centers: list[tuple[float, float]] = []
    for region_idx in region_indices:
        xmin, xmax, ymin, ymax = rectangles[int(region_idx)]
        centers.append(_sample_point_in_rect(rng, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax))
    return tuple(centers)


def _distance(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def _distance_to_axis_aligned_rectangle(
    point: tuple[float, float],
    *,
    center: tuple[float, float],
    length_m: float,
    width_m: float,
) -> float:
    half_length = length_m / 2.0
    half_width = width_m / 2.0
    dx = max(abs(point[0] - center[0]) - half_length, 0.0)
    dy = max(abs(point[1] - center[1]) - half_width, 0.0)
    return math.hypot(dx, dy)


def task_distance_for_vehicle(
    *,
    vehicle_position_m: tuple[float, float],
    task_center_m: tuple[float, float],
    task_length_m: float,
    task_width_m: float,
    roi_distance_model: str,
    distance_floor_m: float,
) -> float:
    r"""Return the regularized vehicle-to-task distance used by the scenario layer.

    Exactness note:
    - Paper-exact: shortest distance to the rectangular RoI.
    - Paper-consistent regularization: the positive distance floor prevents a
      zero-distance singularity before the sensing-potential saturation.
    """

    if distance_floor_m <= 0.0:
        raise ValueError("distance_floor_m must be positive")

    model = roi_distance_model.lower()
    if model == "center":
        distance = _distance(vehicle_position_m, task_center_m)
    elif model == "rectangle":
        distance = _distance_to_axis_aligned_rectangle(
            vehicle_position_m,
            center=task_center_m,
            length_m=task_length_m,
            width_m=task_width_m,
        )
    else:
        raise ValueError(f"unsupported roi_distance_model: {roi_distance_model}")
    return max(distance, distance_floor_m)


def sample_scenario(rng: np.random.Generator, config: ScenarioConfig) -> ScenarioState:
    """Sample one deterministic-paper scenario from the configured RNG state."""

    if config.num_vehicles <= 0:
        raise ValueError("num_vehicles must be positive")
    if config.distance_floor_m <= 0.0:
        raise ValueError("distance_floor_m must be positive")
    if not math.isclose(sum(config.vehicle_zone_probabilities), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("vehicle_zone_probabilities must sum to 1")

    cpu_range = config.cpu_cycles_per_s_range
    if config.cpu_cycles_per_s_regimes is not None:
        regime_count = len(config.cpu_cycles_per_s_regimes)
        probabilities = config.cpu_regime_probabilities
        regime_index = int(
            rng.choice(
                regime_count,
                p=(
                    None
                    if probabilities is None
                    else np.asarray(probabilities, dtype=np.float64)
                ),
            )
        )
        cpu_range = config.cpu_cycles_per_s_regimes[regime_index]

    task_centers = _sample_task_centers(rng, config)
    tasks: list[TaskState] = []
    for task_idx, (urgency_label, beta, tau_max_s) in enumerate(config.urgency_profiles):
        length_m = float(rng.uniform(*config.task_length_range_m))
        width_m = float(rng.uniform(*config.task_width_range_m))
        tasks.append(
            TaskState(
                task_id=task_idx,
                roi_center_m=task_centers[task_idx],
                beta=beta,
                tau_max_s=tau_max_s,
                length_m=length_m,
                width_m=width_m,
                candidate_limit=candidate_limit(
                    task_length_m=length_m,
                    # Decoupled from the quality-model r_0 when the revision
                    # explicitly sets candidate_limit_range_m.
                    sensing_range_m=(
                        config.candidate_limit_range_m
                        if config.candidate_limit_range_m is not None
                        else config.sensing_range_m
                    ),
                    diversity_factor=config.diversity_factor,
                ),
                urgency_label=urgency_label,
            )
        )

    vehicles: list[VehicleState] = []
    channels: list[ChannelState] = []
    # Config-only (RNG-independent) constants, hoisted out of the per-vehicle
    # loop so they are computed once per scenario rather than once per vehicle.
    vehicle_rectangles = _vehicle_zone_rectangles(config)
    vehicle_zone_probs = np.array(config.vehicle_zone_probabilities, dtype=np.float64)
    for vehicle_idx in range(config.num_vehicles):
        position = _sample_vehicle_position(
            rng, config, rectangles=vehicle_rectangles, zone_probs=vehicle_zone_probs
        )
        vehicles.append(
            VehicleState(
                vehicle_id=vehicle_idx,
                position_m=position,
                raw_data_rate_bps=config.raw_data_rate_bps,
                resolution_coeff=float(rng.uniform(*config.resolution_coeff_range)),
                cpu_cycles_per_s=float(rng.uniform(*cpu_range)),
                sensor_power_w=float(rng.uniform(*config.sensor_power_range_w)),
                tx_power_w=config.tx_power_w,
                energy_weight=float(rng.uniform(*config.energy_weight_range)),
            )
        )
        channels.append(
            ChannelState(
                vehicle_id=vehicle_idx,
                distance_to_rsu_m=_distance(position, config.rsu_position_m),
                fading_power=float(rng.exponential(scale=1.0)),
                path_loss_exponent=config.path_loss_exponent,
            )
        )

    # Draw the V*K occlusion block in vehicle-then-task order.
    occlusions = rng.uniform(*config.occlusion_range, size=(config.num_vehicles, len(tasks)))
    distance_to_roi_rows: list[tuple[float, ...]] = []
    occlusion_rows: list[tuple[float, ...]] = []
    for vehicle_idx, vehicle in enumerate(vehicles):
        distance_row = tuple(
            task_distance_for_vehicle(
                vehicle_position_m=vehicle.position_m,
                task_center_m=task.roi_center_m,
                task_length_m=task.length_m,
                task_width_m=task.width_m,
                roi_distance_model=config.roi_distance_model,
                distance_floor_m=config.distance_floor_m,
            )
            for task in tasks
        )
        distance_to_roi_rows.append(distance_row)
        occlusion_rows.append(tuple(float(value) for value in occlusions[vehicle_idx]))

    return ScenarioState(
        tasks=tuple(tasks),
        vehicles=tuple(vehicles),
        channels=tuple(channels),
        distance_to_roi_m=tuple(distance_to_roi_rows),
        occlusion_factors=tuple(occlusion_rows),
    )
