"""Deterministic communication-model formulas.

Exactness label: paper-exact relative to the maintained manuscript equations.
"""

from __future__ import annotations

import math


def channel_gain(
    fading_power: float,
    distance_to_rsu_m: float,
    path_loss_exponent: float,
) -> float:
    """Return ``g_i = |tilde h_i|^2 * d_i^{-alpha}`` from the maintained manuscript equations."""

    if fading_power < 0.0:
        raise ValueError("fading_power must be non-negative")
    if distance_to_rsu_m <= 0.0:
        raise ValueError("distance_to_rsu_m must be positive")
    if path_loss_exponent < 0.0:
        raise ValueError("path_loss_exponent must be non-negative")

    return fading_power * (distance_to_rsu_m**(-path_loss_exponent))


def uplink_rate(
    bandwidth_fraction: float,
    total_bandwidth_hz: float,
    tx_power_w: float,
    channel_gain_value: float,
    noise_power_w: float,
) -> float:
    """Return ``R_i(w_i) = w_i * B_0 * log2(1 + P_i g_i / sigma^2)``.

    Source: the maintained manuscript equations.
    """

    if bandwidth_fraction <= 0.0:
        raise ValueError("bandwidth_fraction must be positive")
    if total_bandwidth_hz <= 0.0:
        raise ValueError("total_bandwidth_hz must be positive")
    if tx_power_w < 0.0:
        raise ValueError("tx_power_w must be non-negative")
    if channel_gain_value < 0.0:
        raise ValueError("channel_gain_value must be non-negative")
    if noise_power_w <= 0.0:
        raise ValueError("noise_power_w must be positive")

    snr = (tx_power_w * channel_gain_value) / noise_power_w
    return bandwidth_fraction * total_bandwidth_hz * math.log2(1.0 + snr)


def transmission_latency(processed_data_volume_bits: float, uplink_rate_bps: float) -> float:
    """Return ``tau^{tr}_{i,k} = tilde D_{i,k} / R_i(w_i)`` from the maintained manuscript equations."""

    if processed_data_volume_bits < 0.0:
        raise ValueError("processed_data_volume_bits must be non-negative")
    if uplink_rate_bps <= 0.0:
        raise ValueError("uplink_rate_bps must be positive")
    return processed_data_volume_bits / uplink_rate_bps


def transmission_energy(tx_power_w: float, transmission_latency_s: float) -> float:
    """Return ``E^{tr}_{i,k} = P_i * tau^{tr}_{i,k}`` from the maintained manuscript equations."""

    if tx_power_w < 0.0:
        raise ValueError("tx_power_w must be non-negative")
    if transmission_latency_s < 0.0:
        raise ValueError("transmission_latency_s must be non-negative")
    return tx_power_w * transmission_latency_s
