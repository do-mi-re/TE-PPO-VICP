"""Shared checkpoint scheduling and selection semantics."""

from __future__ import annotations

from src.rl.checkpoint_selection import (
    EvaluationSchedule,
    resolve_evaluation_interval,
    should_replace_checkpoint,
)


def test_schedule_starts_at_interval_and_forces_distinct_terminal() -> None:
    schedule = EvaluationSchedule(3072)
    observed = [
        timestep
        for timestep in range(32, 50_001, 32)
        if schedule.periodic_due(timestep)
    ]
    if schedule.terminal_due(50_000):
        observed.append(50_000)

    assert observed[0] == 3072
    assert observed[-2:] == [49_152, 50_000]
    assert len(observed) == len(set(observed))


def test_schedule_does_not_duplicate_scheduled_terminal() -> None:
    schedule = EvaluationSchedule(8)
    assert schedule.periodic_due(8)
    assert not schedule.terminal_due(8)


def test_selector_ignores_service_metadata_and_retains_earlier_tie() -> None:
    best = {"social_welfare": 4.0, "service_valid": 0.0}
    assert should_replace_checkpoint(best, social_welfare=5.0)
    assert not should_replace_checkpoint(best, social_welfare=4.0)
    assert not should_replace_checkpoint(best, social_welfare=3.0)


def test_interval_is_measured_in_environment_steps() -> None:
    assert resolve_evaluation_interval({"interval_env_steps": 3072}, default=100) == 3072
    assert resolve_evaluation_interval({}, default=100) == 100
