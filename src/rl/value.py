"""Critic-side configuration helpers for the leader PPO implementation.

Exactness label: paper-consistent approximation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CriticNetworkConfig:
    """Configuration for the PPO critic branch."""

    hidden_sizes: tuple[int, ...] = (256, 256)
    activation: str = "relu"
