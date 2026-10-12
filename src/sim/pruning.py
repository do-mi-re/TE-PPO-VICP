"""Coverage-aware candidate selection.

Exactness label: paper-exact for the maintained bounded sensing model.

For each task, apply the admission radius and optional occlusion threshold,
rank by descending bounded potential, raw potential, and ascending vehicle ID,
and retain the configured candidate limit. Raw potential resolves saturation
ties. Masks depend only on episode geometry, sensing, and occlusion.
Only admitted pairs enter follower optimization; omitted evaluations contribute
zero quality. Formal experiments disable the optional occlusion threshold.
"""

from __future__ import annotations

from src.models.sensing_quality import spatial_potential
from src.sim.scenario import ScenarioConfig
from src.sim.state import ScenarioState

__all__ = ["candidate_masks"]


def candidate_masks(
    state: ScenarioState, config: ScenarioConfig
) -> tuple[frozenset[int], ...]:
    """Return, per vehicle, the set of task ids it is admitted to compete for.

    The result is episode-static: A_{i,k} depends only on sensor resolution,
    occlusion, geometry, r_0 and zeta -- never on the leader's prices or
    bandwidth -- so a caller may compute this once per ``reset`` and reuse it
    for every step of that episode.
    """
    num_vehicles = len(state.vehicles)
    admitted: list[set[int]] = [set() for _ in range(num_vehicles)]

    for task in state.tasks:
        ranked: list[tuple[float, float, int]] = []
        for vehicle_idx, vehicle in enumerate(state.vehicles):
            distance_m = state.distance_to_roi_m[vehicle_idx][task.task_id]
            occlusion = state.occlusion_factors[vehicle_idx][task.task_id]

            if (
                config.candidate_pruning_radius_m is not None
                and distance_m > config.candidate_pruning_radius_m
            ):
                continue
            if (
                config.pruning_occlusion_max is not None
                and occlusion > config.pruning_occlusion_max
            ):
                continue

            raw_potential = spatial_potential(
                resolution_coeff=vehicle.resolution_coeff,
                occlusion_factor=occlusion,
                distance_to_roi_m=distance_m,
                sensing_range_m=config.sensing_range_m,
                sensing_decay_exponent=config.sensing_decay_exponent,
                max_value=None,
            )
            bounded_potential = (
                raw_potential
                if config.spatial_potential_max is None
                else min(raw_potential, config.spatial_potential_max)
            )
            ranked.append((bounded_potential, raw_potential, vehicle_idx))

        # Bounded model first, raw physical potential only within saturation
        # ties, then vehicle id for fully identical observations.
        ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
        for _, _, vehicle_idx in ranked[: task.candidate_limit]:
            admitted[vehicle_idx].add(task.task_id)

    return tuple(frozenset(tasks) for tasks in admitted)
