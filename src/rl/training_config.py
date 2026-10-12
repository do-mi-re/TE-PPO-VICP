"""Resolve effective training configuration defaults without framework imports.

Exactness label: implementation convenience only.  This module freezes the
configuration values consumed by orchestration, provenance, and the SB3
trainer; it does not alter the mathematical environment or reward.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

SB3_ALGORITHMS = frozenset({"proposed", "ebas", "spis", "rdus"})

SB3_ALGORITHM_DEFAULTS: dict[str, Any] = {
    "total_timesteps": 1_000,
    "learning_rate": 3e-4,
    "n_envs": 1,
    "n_steps": 64,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 1.0,
    "clip_range": 0.2,
    "clip_range_vf": None,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "target_kl": None,
    "action_logit_bound": 8.0,
    "initial_price_fraction": 0.5,
    "device": "cpu",
}


def resolve_effective_training_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the complete configuration whose values training will consume.

    Only SB3-backed algorithms use the common defaults above.  JDLS and MAPPO
    own separate custom-loop configuration contracts and are left unchanged.
    """

    resolved = deepcopy(config)
    raw_algorithm = resolved.get("algorithm")
    if not isinstance(raw_algorithm, dict):
        return resolved
    algorithm = dict(raw_algorithm)
    name = str(algorithm.get("name", "proposed")).lower()
    if name in SB3_ALGORITHMS:
        algorithm = {**SB3_ALGORITHM_DEFAULTS, **algorithm}
        resolved["algorithm"] = algorithm
    if name != "dbas":
        from src.rl.checkpoint_selection import resolve_evaluation_interval

        evaluation = dict(resolved.get("evaluation") or {})
        default_interval = (
            100 * int(algorithm.get("n_envs", 1))
            if name in SB3_ALGORITHMS
            else int(algorithm.get("rollout_steps", 1))
        )
        evaluation["interval_env_steps"] = resolve_evaluation_interval(
            evaluation, default=default_interval
        )
        resolved["evaluation"] = evaluation
    return resolved
