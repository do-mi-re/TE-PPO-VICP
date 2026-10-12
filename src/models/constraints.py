"""Deterministic latency-constraint helpers.

Exactness label: paper-exact relative to the maintained manuscript equations.
"""

from __future__ import annotations


def total_latency(
    sensing_latency_s: float,
    computation_latency_s: float,
    transmission_latency_s: float,
) -> float:
    """Return ``tau^{tot}_{i,k} = tau^s + tau^c + tau^{tr}``.

    Source: the maintained additive sensing/computation/transmission latency definition.
    """

    if sensing_latency_s < 0.0:
        raise ValueError("sensing_latency_s must be non-negative")
    if computation_latency_s < 0.0:
        raise ValueError("computation_latency_s must be non-negative")
    if transmission_latency_s < 0.0:
        raise ValueError("transmission_latency_s must be non-negative")
    return sensing_latency_s + computation_latency_s + transmission_latency_s


def latency_feasible(
    sensing_latency_s: float,
    computation_latency_s: float,
    transmission_latency_s: float,
    deadline_s: float,
) -> bool:
    """Return whether ``tau^s + tau^c + tau^{tr} <= tau^{max}``.

    Source: the maintained follower deadline constraint C2.
    """

    if deadline_s < 0.0:
        raise ValueError("deadline_s must be non-negative")
    return total_latency(
        sensing_latency_s=sensing_latency_s,
        computation_latency_s=computation_latency_s,
        transmission_latency_s=transmission_latency_s,
    ) <= deadline_s
