"""Utility and welfare formulas.

Exactness label: paper-exact relative to the maintained manuscript equations.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def total_energy(
    sensing_energy_j: float,
    computation_energy_j: float,
    transmission_energy_j: float,
) -> float:
    """Return ``E^{tot}_{i,k} = E^s_{i,k} + E^c_{i,k} + E^{tr}_{i,k}``.

    Source: the maintained manuscript equations.
    """

    if sensing_energy_j < 0.0:
        raise ValueError("sensing_energy_j must be non-negative")
    if computation_energy_j < 0.0:
        raise ValueError("computation_energy_j must be non-negative")
    if transmission_energy_j < 0.0:
        raise ValueError("transmission_energy_j must be non-negative")
    return sensing_energy_j + computation_energy_j + transmission_energy_j


def participant_payment(price_per_quality_unit: float, sensing_quality_value: float) -> float:
    """Return the payment term ``p_k * q_{i,k}`` from the maintained manuscript equations."""

    if price_per_quality_unit < 0.0:
        raise ValueError("price_per_quality_unit must be non-negative")
    if sensing_quality_value < 0.0:
        raise ValueError("sensing_quality_value must be non-negative")
    return price_per_quality_unit * sensing_quality_value


def task_value(task_weight_beta: float, total_task_quality: float) -> float:
    """Return ``beta_k * ln(1 + sum_i x_{i,k} q_{i,k})`` from the maintained manuscript equations."""

    if task_weight_beta < 0.0:
        raise ValueError("task_weight_beta must be non-negative")
    if total_task_quality < 0.0:
        raise ValueError("total_task_quality must be non-negative")
    return task_weight_beta * math.log1p(total_task_quality)


def bandwidth_cost(
    bandwidth_cost_coeff: float,
    total_bandwidth_hz: float,
    allocated_bandwidth_fractions: Sequence[float],
) -> float:
    """Return the maintained bandwidth-cost term ``xi * B_0 * sum_i w_i``."""

    if bandwidth_cost_coeff < 0.0:
        raise ValueError("bandwidth_cost_coeff must be non-negative")
    if total_bandwidth_hz < 0.0:
        raise ValueError("total_bandwidth_hz must be non-negative")
    total_fraction = 0.0
    for bandwidth_fraction in allocated_bandwidth_fractions:
        if bandwidth_fraction < 0.0:
            raise ValueError("allocated bandwidth fractions must be non-negative")
        total_fraction += bandwidth_fraction
    return bandwidth_cost_coeff * total_bandwidth_hz * total_fraction


def vehicle_task_utility(
    price_per_quality_unit: float,
    sensing_quality_value: float,
    energy_weight: float,
    sensing_energy_j: float,
    computation_energy_j: float,
    transmission_energy_j: float,
) -> float:
    """Return conditional vehicle utility for one task from the maintained manuscript equations."""

    if energy_weight < 0.0:
        raise ValueError("energy_weight must be non-negative")
    payment = participant_payment(
        price_per_quality_unit=price_per_quality_unit,
        sensing_quality_value=sensing_quality_value,
    )
    energy_cost = energy_weight * total_energy(
        sensing_energy_j=sensing_energy_j,
        computation_energy_j=computation_energy_j,
        transmission_energy_j=transmission_energy_j,
    )
    return payment - energy_cost


def vehicle_utility(
    task_selection_flags: Sequence[int],
    prices_per_task: Sequence[float],
    qualities_per_task: Sequence[float],
    energy_weight: float,
    sensing_energies_j: Sequence[float],
    computation_energies_j: Sequence[float],
    transmission_energies_j: Sequence[float],
) -> float:
    """Return ``U_i^{veh}`` from the maintained manuscript equations."""

    lengths = {
        len(task_selection_flags),
        len(prices_per_task),
        len(qualities_per_task),
        len(sensing_energies_j),
        len(computation_energies_j),
        len(transmission_energies_j),
    }
    if len(lengths) != 1:
        raise ValueError("vehicle_utility inputs must have the same length")

    total = 0.0
    for selected, price, quality, sense_e, comp_e, tx_e in zip(
        task_selection_flags,
        prices_per_task,
        qualities_per_task,
        sensing_energies_j,
        computation_energies_j,
        transmission_energies_j,
        strict=True,
    ):
        if selected not in (0, 1):
            raise ValueError("task_selection_flags must contain only 0 or 1")
        if selected == 1:
            total += vehicle_task_utility(
                price_per_quality_unit=price,
                sensing_quality_value=quality,
                energy_weight=energy_weight,
                sensing_energy_j=sense_e,
                computation_energy_j=comp_e,
                transmission_energy_j=tx_e,
            )
    return total


def rsu_utility(
    task_weights_beta: Sequence[float],
    qualities_by_task: Sequence[Sequence[float]],
    selections_by_task: Sequence[Sequence[int]],
    prices_by_task: Sequence[float],
    total_bandwidth_hz: float,
    bandwidth_cost_coeff: float,
    allocated_bandwidth_fractions: Sequence[float],
) -> float:
    """Return ``U^{RSU}`` from the maintained manuscript equations."""

    task_count = len(task_weights_beta)
    if not (
        len(qualities_by_task) == len(selections_by_task) == len(prices_by_task) == task_count
    ):
        raise ValueError("task-level RSU inputs must have the same outer length")

    total_value = 0.0
    total_payment = 0.0
    for beta_k, qualities_k, selections_k, price_k in zip(
        task_weights_beta,
        qualities_by_task,
        selections_by_task,
        prices_by_task,
        strict=True,
    ):
        if len(qualities_k) != len(selections_k):
            raise ValueError("per-task qualities and selections must have the same length")
        task_quality = 0.0
        for quality, selected in zip(qualities_k, selections_k, strict=True):
            if selected not in (0, 1):
                raise ValueError("selections_by_task must contain only 0 or 1")
            if quality < 0.0:
                raise ValueError("qualities_by_task must be non-negative")
            if selected == 1:
                task_quality += quality
                total_payment += participant_payment(
                    price_per_quality_unit=price_k,
                    sensing_quality_value=quality,
                )
        total_value += task_value(task_weight_beta=beta_k, total_task_quality=task_quality)

    return total_value - total_payment - bandwidth_cost(
        bandwidth_cost_coeff=bandwidth_cost_coeff,
        total_bandwidth_hz=total_bandwidth_hz,
        allocated_bandwidth_fractions=allocated_bandwidth_fractions,
    )


def social_welfare(
    task_weights_beta: Sequence[float],
    qualities_by_task: Sequence[Sequence[float]],
    selections_by_task: Sequence[Sequence[int]],
    energy_weights_by_vehicle: Sequence[float],
    sensing_energies_by_task: Sequence[Sequence[float]],
    computation_energies_by_task: Sequence[Sequence[float]],
    transmission_energies_by_task: Sequence[Sequence[float]],
    total_bandwidth_hz: float,
    bandwidth_cost_coeff: float,
    allocated_bandwidth_fractions: Sequence[float],
) -> float:
    """Return social welfare ``W`` from the maintained manuscript equations."""

    task_count = len(task_weights_beta)
    if not (
        len(qualities_by_task)
        == len(selections_by_task)
        == len(sensing_energies_by_task)
        == len(computation_energies_by_task)
        == len(transmission_energies_by_task)
        == task_count
    ):
        raise ValueError("task-level welfare inputs must have the same outer length")

    total_value = 0.0
    weighted_energy = 0.0
    vehicle_count = len(energy_weights_by_vehicle)

    for beta_k, qualities_k, selections_k, sense_k, comp_k, tx_k in zip(
        task_weights_beta,
        qualities_by_task,
        selections_by_task,
        sensing_energies_by_task,
        computation_energies_by_task,
        transmission_energies_by_task,
        strict=True,
    ):
        if not (
            len(qualities_k)
            == len(selections_k)
            == len(sense_k)
            == len(comp_k)
            == len(tx_k)
            == vehicle_count
        ):
            raise ValueError("each task row must align with energy_weights_by_vehicle")

        task_quality = 0.0
        for vehicle_idx, (quality, selected, sense_e, comp_e, tx_e) in enumerate(
            zip(qualities_k, selections_k, sense_k, comp_k, tx_k, strict=True)
        ):
            if selected not in (0, 1):
                raise ValueError("selections_by_task must contain only 0 or 1")
            if quality < 0.0:
                raise ValueError("qualities_by_task must be non-negative")
            if energy_weights_by_vehicle[vehicle_idx] < 0.0:
                raise ValueError("energy_weights_by_vehicle must be non-negative")
            if selected == 1:
                task_quality += quality
                weighted_energy += energy_weights_by_vehicle[vehicle_idx] * total_energy(
                    sensing_energy_j=sense_e,
                    computation_energy_j=comp_e,
                    transmission_energy_j=tx_e,
                )
        total_value += task_value(task_weight_beta=beta_k, total_task_quality=task_quality)

    return total_value - weighted_energy - bandwidth_cost(
        bandwidth_cost_coeff=bandwidth_cost_coeff,
        total_bandwidth_hz=total_bandwidth_hz,
        allocated_bandwidth_fractions=allocated_bandwidth_fractions,
    )
