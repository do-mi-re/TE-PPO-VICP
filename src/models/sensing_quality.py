"""Deterministic sensing-quality formulas used by the maintained model.

All functions in this module operate on SI units and are side-effect free.
The maintained semantic contract is recorded in
``docs/revision_experiment_protocol.md``.
"""

from __future__ import annotations

import math


def raw_data_volume(raw_data_rate_bps: float, sensing_duration_s: float) -> float:
    """Return ``D_{i,k} = s_i * tau^s_{i,k}`` from the maintained manuscript equations."""

    if raw_data_rate_bps < 0.0:
        raise ValueError("raw_data_rate_bps must be non-negative")
    if sensing_duration_s < 0.0:
        raise ValueError("sensing_duration_s must be non-negative")
    return raw_data_rate_bps * sensing_duration_s


def sensing_energy(sensor_power_w: float, sensing_duration_s: float) -> float:
    """Return ``E^s_{i,k} = P_{s,i} * tau^s_{i,k}``."""

    if sensor_power_w < 0.0:
        raise ValueError("sensor_power_w must be non-negative")
    if sensing_duration_s < 0.0:
        raise ValueError("sensing_duration_s must be non-negative")
    return sensor_power_w * sensing_duration_s


def spatial_potential(
    resolution_coeff: float,
    occlusion_factor: float,
    distance_to_roi_m: float,
    sensing_range_m: float,
    sensing_decay_exponent: float,
    max_value: float | None = None,
) -> float:
    """Return the bounded spatial potential ``A_{i,k}``.

    Exactness note on ``max_value``:
    - Paper-exact when ``max_value`` is set: finite sensor resolution
      imposes a near-RoI saturation, so ``A`` is clamped to ``max_value``.
    - Paper-exact raw term when ``max_value`` is ``None``: the pre-saturation
      potential is used to break ties in bounded candidate ranking.
    """

    if resolution_coeff < 0.0:
        raise ValueError("resolution_coeff must be non-negative")
    if not 0.0 <= occlusion_factor <= 1.0:
        raise ValueError("occlusion_factor must be within [0, 1]")
    if distance_to_roi_m <= 0.0:
        raise ValueError("distance_to_roi_m must be positive")
    if sensing_range_m <= 0.0:
        raise ValueError("sensing_range_m must be positive")
    if sensing_decay_exponent < 0.0:
        raise ValueError("sensing_decay_exponent must be non-negative")
    if max_value is not None and max_value <= 0.0:
        raise ValueError("max_value must be positive when set")

    potential = (
        resolution_coeff
        * (1.0 - occlusion_factor)
        * math.pow(distance_to_roi_m / sensing_range_m, -sensing_decay_exponent)
    )
    if max_value is not None:
        return min(potential, max_value)
    return potential


def temporal_gain(
    sensing_duration_s: float,
    data_efficiency_factor: float,
    raw_data_rate_bps: float,
) -> float:
    """Return ``G(tau^s_{i,k}) = ln(1 + gamma * s_i * tau^s_{i,k})``.

    Source: the maintained manuscript equations.
    """

    if sensing_duration_s < 0.0:
        raise ValueError("sensing_duration_s must be non-negative")
    if data_efficiency_factor < 0.0:
        raise ValueError("data_efficiency_factor must be non-negative")
    if raw_data_rate_bps < 0.0:
        raise ValueError("raw_data_rate_bps must be non-negative")

    return math.log1p(data_efficiency_factor * raw_data_rate_bps * sensing_duration_s)


def sensing_quality(
    resolution_coeff: float,
    occlusion_factor: float,
    distance_to_roi_m: float,
    sensing_range_m: float,
    sensing_decay_exponent: float,
    sensing_duration_s: float,
    data_efficiency_factor: float,
    raw_data_rate_bps: float,
    spatial_potential_max: float | None = None,
    spatial_potential_value: float | None = None,
) -> float:
    """Return ``q_{i,k}(tau^s_{i,k}) = A_{i,k} * G(tau^s_{i,k})``.

    Source: the maintained manuscript equations, with ``A_{i,k}`` defined in ``docs/manuscript_revision_snippet.tex``.
    ``spatial_potential_max`` forwards the optional spatial-potential bound to
    :func:`spatial_potential` (see its exactness note). ``spatial_potential_value``
    lets a caller that already computed ``A_{i,k}`` (e.g. the follower solver's
    ``solve_task_response``, which needs it for the latency-multiplier solve) pass
    it in to avoid recomputing the same deterministic value — bit-for-bit
    identical to recomputing it here.
    """

    if spatial_potential_value is None:
        spatial_potential_value = spatial_potential(
            resolution_coeff=resolution_coeff,
            occlusion_factor=occlusion_factor,
            distance_to_roi_m=distance_to_roi_m,
            sensing_range_m=sensing_range_m,
            sensing_decay_exponent=sensing_decay_exponent,
            max_value=spatial_potential_max,
        )
    return spatial_potential_value * temporal_gain(
        sensing_duration_s=sensing_duration_s,
        data_efficiency_factor=data_efficiency_factor,
        raw_data_rate_bps=raw_data_rate_bps,
    )
