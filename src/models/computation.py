"""Deterministic computation-model formulas.

Exactness label: paper-exact relative to the maintained manuscript equations.
"""

from __future__ import annotations

from src.models.sensing_quality import raw_data_volume

MAX_COMPRESSION_RATIO = 50.0


def processed_data_size(raw_data_volume_bits: float, compression_ratio: float) -> float:
    """Return ``tilde D_{i,k} = D_{i,k} / eta_{i,k}``.

    Source: the maintained manuscript equations, with bound ``1 <= eta <= 50``.
    """

    if raw_data_volume_bits < 0.0:
        raise ValueError("raw_data_volume_bits must be non-negative")
    if not 1.0 <= compression_ratio <= MAX_COMPRESSION_RATIO:
        raise ValueError("compression_ratio must be within [1, 50]")
    return raw_data_volume_bits / compression_ratio


def compression_complexity(
    compression_ratio: float,
    baseline_complexity_cycles_per_bit: float,
    scaling_complexity_cycles_per_bit: float,
    complexity_exponent: float,
) -> float:
    """Return ``C(eta) = epsilon_0 + epsilon_1 * eta^nu``.

    Source: the maintained manuscript equations.
    """

    if not 1.0 <= compression_ratio <= MAX_COMPRESSION_RATIO:
        raise ValueError("compression_ratio must be within [1, 50]")
    if baseline_complexity_cycles_per_bit < 0.0:
        raise ValueError("baseline_complexity_cycles_per_bit must be non-negative")
    if scaling_complexity_cycles_per_bit < 0.0:
        raise ValueError("scaling_complexity_cycles_per_bit must be non-negative")
    if complexity_exponent <= 0.0:
        raise ValueError("complexity_exponent must be positive")

    return baseline_complexity_cycles_per_bit + (
        scaling_complexity_cycles_per_bit * (compression_ratio**complexity_exponent)
    )


def computation_latency(
    raw_data_rate_bps: float,
    sensing_duration_s: float,
    compression_ratio: float,
    baseline_complexity_cycles_per_bit: float,
    scaling_complexity_cycles_per_bit: float,
    complexity_exponent: float,
    cpu_cycles_per_s: float,
) -> float:
    """Return computation latency ``tau^c_{i,k}`` from the maintained manuscript equations."""

    if cpu_cycles_per_s <= 0.0:
        raise ValueError("cpu_cycles_per_s must be positive")

    raw_bits = raw_data_volume(
        raw_data_rate_bps=raw_data_rate_bps,
        sensing_duration_s=sensing_duration_s,
    )
    complexity = compression_complexity(
        compression_ratio=compression_ratio,
        baseline_complexity_cycles_per_bit=baseline_complexity_cycles_per_bit,
        scaling_complexity_cycles_per_bit=scaling_complexity_cycles_per_bit,
        complexity_exponent=complexity_exponent,
    )
    return (raw_bits * complexity) / cpu_cycles_per_s


def computation_energy(
    raw_data_rate_bps: float,
    sensing_duration_s: float,
    compression_ratio: float,
    baseline_complexity_cycles_per_bit: float,
    scaling_complexity_cycles_per_bit: float,
    complexity_exponent: float,
    switched_capacitance_coeff: float,
    cpu_cycles_per_s: float,
) -> float:
    """Return ``E^c_{i,k} = kappa * F_i^2 * [D_{i,k} * C(eta_{i,k})]``.

    Source: the maintained manuscript equations.
    """

    if switched_capacitance_coeff < 0.0:
        raise ValueError("switched_capacitance_coeff must be non-negative")
    if cpu_cycles_per_s <= 0.0:
        raise ValueError("cpu_cycles_per_s must be positive")

    raw_bits = raw_data_volume(
        raw_data_rate_bps=raw_data_rate_bps,
        sensing_duration_s=sensing_duration_s,
    )
    complexity = compression_complexity(
        compression_ratio=compression_ratio,
        baseline_complexity_cycles_per_bit=baseline_complexity_cycles_per_bit,
        scaling_complexity_cycles_per_bit=scaling_complexity_cycles_per_bit,
        complexity_exponent=complexity_exponent,
    )
    return switched_capacitance_coeff * (cpu_cycles_per_s**2) * raw_bits * complexity
