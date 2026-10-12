"""Immutable run identities for final experiments.

Exactness label: implementation convenience only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from src.evaluation.sweep import (
    parse_sweep_value,
)

RunPurpose = Literal[
    "sweep_metric",
    "baseline_line",
    "timing_scalability",
    "pruning_sensitivity",
    "cpu_robustness",
]


# Derived sweeps key their scenario override on a field other than the sweep
# param (num_tasks sets urgency_profiles), so
# display_value cannot read the grid value back out of the override dict. Each
# such sweep records the prefix parse_sweep_value stamps on its run-name suffix,
# and the value is recovered by stripping it.
_DERIVED_SUFFIX_PREFIXES: dict[str, str] = {
    "num_tasks": "k",
}


def _display_value_from_suffix(sweep_param: str, suffix: str) -> float:
    """Recover a derived sweep's grid value from the suffix it produced."""
    prefix = _DERIVED_SUFFIX_PREFIXES.get(sweep_param)
    if prefix is None or not suffix.startswith(prefix):
        raise ValueError(
            f"cannot recover a display value for sweep_param={sweep_param!r} "
            f"from suffix {suffix!r}; add it to _DERIVED_SUFFIX_PREFIXES"
        )
    try:
        return float(suffix[len(prefix):])
    except ValueError as exc:
        raise ValueError(
            f"suffix {suffix!r} for sweep_param={sweep_param!r} does not end "
            "in a numeric grid value"
        ) from exc


@dataclass(frozen=True)
class RunSpec:
    """One (algorithm, seed, scenario) that must be trained exactly once.

    ``run_name`` is the dedup key — identical run_name means identical
    artifacts on disk, regardless of which figure asked for it.
    """
    algorithm: str
    train_seed: int
    run_name_suffix: str
    # scenario_overrides stored as sorted tuple of (key, value) so RunSpec
    # stays hashable; values nested via ``_to_hashable``. Access via the
    # ``scenario_overrides`` property.
    scenario_overrides_items: tuple[tuple[str, Any], ...]
    sweep_param: str | None
    purpose: RunPurpose
    # Optional dotted-path overrides for controlled experiment protocols.
    config_overrides_items: tuple[tuple[str, Any], ...] = field(default_factory=tuple)
    # Optional arm identity for controlled experiments that share one
    # algorithm implementation but must remain distinct in aggregation.
    experiment_arm: str | None = None

    @property
    def run_name(self) -> str:
        suffix = f"_{self.run_name_suffix}" if self.run_name_suffix else ""
        return f"{self.algorithm}_s{self.train_seed}{suffix}"

    def run_dir(self, root: Path | str = "artifacts/sweep") -> Path:
        return Path(root) / self.run_name

    @property
    def scenario_overrides(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in self.scenario_overrides_items:
            # cpu_cycles_per_s_range was tupled for hashability; restore list.
            out[key] = list(value) if isinstance(value, tuple) else value
        return out

    @property
    def config_overrides(self) -> dict[str, Any]:
        return {key: value for key, value in self.config_overrides_items}

    @property
    def display_value(self) -> float | None:
        """Derived from sweep_param + scenario_overrides via parse_sweep_value."""
        if self.sweep_param is None:
            return None
        overrides = self.scenario_overrides
        if self.sweep_param in overrides:
            return parse_sweep_value(self.sweep_param, overrides[self.sweep_param])[2]
        # Derived sweeps whose grid value is not itself a ScenarioConfig field
        # (num_tasks -> urgency_profiles)
        # cannot be read back from the override dict, so recover the value from
        # the run-name suffix, which parse_sweep_value produced from it.
        return _display_value_from_suffix(self.sweep_param, self.run_name_suffix)

    def to_worker_dict(self) -> dict[str, Any]:
        """Plain-dict view for pickling to worker processes. RunSpec itself is
        picklable, but workers want the dict form of scenario_overrides."""
        return {
            "algorithm": self.algorithm,
            "train_seed": self.train_seed,
            "run_name_suffix": self.run_name_suffix,
            "run_name": self.run_name,
            "scenario_overrides": self.scenario_overrides,
            "sweep_param": self.sweep_param,
            "display_value": self.display_value,
            "purpose": self.purpose,
            "config_overrides": self.config_overrides,
            "experiment_arm": self.experiment_arm,
        }
