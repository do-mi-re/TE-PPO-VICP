"""Shared validation scheduling and checkpoint-selection contracts.

Exactness label: implementation convenience only.  This module does not alter
the game, reward, or optimizer.  It makes the validation/model-selection
semantics identical across the SB3, JDLS, and MAPPO training engines.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

SELECTION_MANIFEST = "selection.json"
SB3_SELECTED_MODEL = "selected_model.zip"
SB3_LAST_MODEL = "last_model.zip"
CUSTOM_SELECTED_MODEL = "selected_model.pt"
CUSTOM_LAST_MODEL = "last_model.pt"
SELECTED_VECNORMALIZE = "selected_vecnormalize.pkl"
LAST_TRAIN_VECNORMALIZE = "last_train_vecnormalize.pkl"
CPU_MAXIMIN_SELECTION_RULE = "maximin_cpu_band_social_welfare"


def selected_model_name(algorithm: str) -> str:
    """Return the formal selected-checkpoint filename for an algorithm."""

    return CUSTOM_SELECTED_MODEL if algorithm in {"jdls", "mappo"} else SB3_SELECTED_MODEL


def resolve_evaluation_interval(
    evaluation: Mapping[str, Any],
    *,
    default: int,
) -> int:
    """Resolve validation cadence to physical environment steps.

    Cadence uses ``interval_env_steps``, independent of vector-env count.
    """

    if "eval_freq" in evaluation:
        raise ValueError("Use evaluation.interval_env_steps for cadence")
    interval = int(evaluation.get("interval_env_steps", default))
    if interval <= 0:
        raise ValueError("evaluation interval must be positive")
    return interval


@dataclass
class EvaluationSchedule:
    """Track periodic and forced-terminal evaluations in physical steps."""

    interval_env_steps: int
    force_terminal_evaluation: bool = True
    next_timestep: int = 0
    last_evaluated_timestep: int | None = None

    def __post_init__(self) -> None:
        if self.interval_env_steps <= 0:
            raise ValueError("interval_env_steps must be positive")
        if self.next_timestep <= 0:
            self.next_timestep = self.interval_env_steps

    def periodic_due(self, timestep: int) -> bool:
        """Return true once when ``timestep`` reaches the next interval."""

        if timestep < self.next_timestep:
            return False
        while self.next_timestep <= timestep:
            self.next_timestep += self.interval_env_steps
        return self._mark_if_new(timestep)

    def terminal_due(self, timestep: int) -> bool:
        """Return true when a distinct terminal evaluation is required."""

        return self.force_terminal_evaluation and self._mark_if_new(timestep)

    def _mark_if_new(self, timestep: int) -> bool:
        if self.last_evaluated_timestep == int(timestep):
            return False
        self.last_evaluated_timestep = int(timestep)
        return True


def should_replace_checkpoint(
    best: Mapping[str, Any] | None,
    *,
    social_welfare: float,
) -> bool:
    """Select a strict social-welfare improvement; ties keep the earlier model."""

    return best is None or float(social_welfare) > float(
        best.get("social_welfare", float("-inf"))
    )


@dataclass(frozen=True)
class RobustCheckpointScore:
    """Lexicographic maximin score for a fixed validation-cell grid."""

    minimum_cell_social_welfare: float
    macro_mean_social_welfare: float
    cell_social_welfare_means: tuple[float, ...]


def cpu_validation_ranges(
    evaluation: Mapping[str, Any],
) -> tuple[tuple[float, float], ...]:
    """Return a validated optional CPU checkpoint-selection grid."""

    raw = evaluation.get("cpu_capacity_ranges_cycles_per_s")
    rule = evaluation.get("checkpoint_selection")
    if raw is None:
        if rule not in (None, "mean_social_welfare"):
            raise ValueError("CPU maximin selection requires a CPU validation grid")
        return ()
    if rule != CPU_MAXIMIN_SELECTION_RULE:
        raise ValueError("CPU validation grid requires maximin checkpoint selection")
    ranges = tuple(tuple(float(endpoint) for endpoint in band) for band in raw)
    if not ranges:
        raise ValueError("CPU validation grid must not be empty")
    for band in ranges:
        if len(band) != 2 or not 0.0 < band[0] < band[1]:
            raise ValueError("every CPU validation range must satisfy 0 < low < high")
    return ranges


def validation_grid_digest(ranges: tuple[tuple[float, float], ...]) -> str:
    payload = json.dumps(ranges, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def robust_checkpoint_score(
    social_welfare: list[float] | tuple[float, ...],
    *,
    cell_count: int,
    seeds_per_cell: int,
) -> RobustCheckpointScore:
    """Reduce cell-major validation observations to the preregistered score."""

    if cell_count <= 0 or seeds_per_cell <= 0:
        raise ValueError("validation cell and seed counts must be positive")
    values = tuple(float(value) for value in social_welfare)
    if len(values) != cell_count * seeds_per_cell:
        raise ValueError("validation observations do not match the declared grid")
    if any(not value == value or value in (float("inf"), float("-inf")) for value in values):
        raise ValueError("validation social welfare must be finite")
    cell_means = tuple(
        sum(values[start:start + seeds_per_cell]) / seeds_per_cell
        for start in range(0, len(values), seeds_per_cell)
    )
    return RobustCheckpointScore(
        minimum_cell_social_welfare=min(cell_means),
        macro_mean_social_welfare=sum(cell_means) / len(cell_means),
        cell_social_welfare_means=cell_means,
    )


def should_replace_robust_checkpoint(
    best: RobustCheckpointScore | None,
    candidate: RobustCheckpointScore,
) -> bool:
    """Maximize worst-cell welfare, then macro mean; exact ties keep earlier."""

    if best is None:
        return True
    return (
        candidate.minimum_cell_social_welfare,
        candidate.macro_mean_social_welfare,
    ) > (
        best.minimum_cell_social_welfare,
        best.macro_mean_social_welfare,
    )


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish a JSON manifest after its referenced files exist."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8"
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
