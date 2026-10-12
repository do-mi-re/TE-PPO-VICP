"""Tests for runtime-only training progress reporting."""

from __future__ import annotations

import io
from unittest.mock import patch

import numpy as np
import torch

from src.rl.progress import TrainingProgress, resolve_progress_mode


class _Stream(io.StringIO):
    def __init__(self, *, tty: bool) -> None:
        super().__init__()
        self.tty = tty

    def isatty(self) -> bool:
        return self.tty


def test_auto_mode_uses_bar_for_tty_and_log_for_parallel_worker() -> None:
    with patch.dict("os.environ", {}, clear=True):
        assert resolve_progress_mode(stream=_Stream(tty=True)) == "bar"
        assert resolve_progress_mode(stream=_Stream(tty=False)) == "log"
    with patch.dict("os.environ", {"TEPPO_PARALLEL_WORKER": "1"}, clear=True):
        assert resolve_progress_mode(stream=_Stream(tty=True)) == "log"


def test_log_progress_is_monotonic_and_reports_metric_at_five_percent() -> None:
    now = [0.0]
    output = _Stream(tty=False)
    progress = TrainingProgress(
        total=100,
        label="jdls_s123",
        mode="log",
        stream=output,
        clock=lambda: now[0],
    )

    now[0] = 1.0
    progress.update(4)
    now[0] = 2.0
    progress.update(5, social_welfare=1.25)
    progress.update(3)  # an out-of-order observation must not move backwards
    now[0] = 10.0
    progress.update(150)  # SB3 rollout overshoot is clamped to the budget
    progress.close()

    lines = output.getvalue().splitlines()
    assert len(lines) == 3  # start, first 5% boundary, completion
    assert "0/100" in lines[0]
    assert "5/100" in lines[1]
    assert "eval_sw=+1.250" in lines[1]
    assert "100/100" in lines[2]
    assert progress.completed == 100


def test_progress_observation_does_not_consume_numpy_or_torch_rng() -> None:
    np.random.seed(123)
    torch.manual_seed(123)
    expected_numpy = np.random.random(4)
    expected_torch = torch.rand(4)

    np.random.seed(123)
    torch.manual_seed(123)
    progress = TrainingProgress(total=10, label="rng-check", mode="off")
    progress.phase("formal training")
    progress.update(5, social_welfare=-2.0)
    progress.update(10)
    progress.close()

    np.testing.assert_array_equal(np.random.random(4), expected_numpy)
    torch.testing.assert_close(torch.rand(4), expected_torch, rtol=0.0, atol=0.0)
