from __future__ import annotations

from src.evaluation.provenance import config_digest
from src.rl.training_config import resolve_effective_training_config


def test_sb3_defaults_are_part_of_effective_config_and_digest():
    raw = {"algorithm": {"name": "proposed", "total_timesteps": 8}}

    resolved = resolve_effective_training_config(raw)

    assert resolved["algorithm"]["gamma"] == 0.99
    assert resolved["algorithm"]["clip_range"] == 0.2
    assert "gamma" not in raw["algorithm"]
    assert config_digest(raw) == config_digest(resolved)


def test_custom_trainers_do_not_acquire_sb3_only_defaults():
    raw = {"algorithm": {"name": "jdls", "total_timesteps": 8}}

    resolved = resolve_effective_training_config(raw)

    assert resolved["algorithm"] == raw["algorithm"]
    assert resolved["evaluation"] == {"interval_env_steps": 1}
