"""Training progress display shared by SB3, JDLS, and MAPPO.

Exactness label: implementation convenience only.  This module observes
completed environment steps and evaluation metrics; it never mutates model,
environment, optimizer, or random-number-generator state.
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from collections.abc import Callable
from typing import Literal, TextIO

from tqdm import tqdm

ProgressMode = Literal["auto", "bar", "log", "off"]
_VALID_MODES = frozenset({"auto", "bar", "log", "off"})


def resolve_progress_mode(
    mode: str | None = None,
    *,
    stream: TextIO | None = None,
) -> Literal["bar", "log", "off"]:
    """Resolve the runtime-only progress mode.

    ``TEPPO_PROGRESS`` may force ``bar``, ``log``, or ``off``.  In ``auto``
    mode, scheduled worker processes use stable log lines so their output does
    not corrupt the parent scheduler's task bar.  Other interactive terminals
    use a dynamic bar; redirected output uses stable log lines.
    """

    output = stream or sys.stderr
    requested = str(mode or os.environ.get("TEPPO_PROGRESS", "auto")).lower()
    if requested not in _VALID_MODES:
        warnings.warn(
            f"ignoring invalid TEPPO_PROGRESS={requested!r}; using auto",
            RuntimeWarning,
            stacklevel=2,
        )
        requested = "auto"
    if requested != "auto":
        return requested  # type: ignore[return-value]
    if os.environ.get("TEPPO_PARALLEL_WORKER") == "1":
        return "log"
    return "bar" if output.isatty() else "log"


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class TrainingProgress:
    """Display monotonic environment-step progress without affecting training."""

    def __init__(
        self,
        *,
        total: int,
        label: str,
        mode: str | None = None,
        stream: TextIO | None = None,
        log_fraction: float = 0.05,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if total <= 0:
            raise ValueError("progress total must be positive")
        if not 0.0 < log_fraction <= 1.0:
            raise ValueError("log_fraction must lie in (0, 1]")
        self.total = int(total)
        self.label = str(label)
        self.stream = stream or sys.stderr
        self.mode = resolve_progress_mode(mode, stream=self.stream)
        self.log_fraction = float(log_fraction)
        self.clock = clock
        self.started_at = self.clock()
        self.completed = 0
        self.latest_social_welfare: float | None = None
        self._next_log_fraction = self.log_fraction
        self._closed = False
        self._bar = None
        if self.mode == "bar":
            self._bar = tqdm(
                total=self.total,
                desc=self.label,
                unit="step",
                dynamic_ncols=True,
                mininterval=1.0,
                file=self.stream,
            )
        elif self.mode == "log":
            self._write_log_line()

    def __enter__(self) -> TrainingProgress:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def phase(self, message: str) -> None:
        """Emit a concise phase marker without disturbing a dynamic bar."""

        text = f"[{self.label}] {message}"
        if self.mode == "bar":
            tqdm.write(text, file=self.stream)
        elif self.mode == "log":
            print(text, file=self.stream, flush=True)

    def restart_timing(self) -> None:
        """Start speed/ETA timing when the formal training loop begins."""

        self.started_at = self.clock()

    def update(
        self,
        completed: int,
        *,
        social_welfare: float | None = None,
    ) -> None:
        """Advance to an absolute environment-step count.

        Counts are clamped to the requested budget because SB3 completes its
        current vectorized rollout and may therefore report a small overshoot.
        """

        if self._closed:
            return
        new_completed = min(self.total, max(self.completed, int(completed)))
        advanced = new_completed > self.completed
        if social_welfare is not None:
            self.latest_social_welfare = float(social_welfare)
        if self._bar is not None:
            self._bar.update(new_completed - self.completed)
            if self.latest_social_welfare is not None:
                self._bar.set_postfix(
                    eval_sw=f"{self.latest_social_welfare:+.3f}", refresh=False
                )
        self.completed = new_completed
        if self.mode == "log" and advanced:
            fraction = self.completed / self.total
            if fraction >= self._next_log_fraction or self.completed >= self.total:
                self._write_log_line()
                while self._next_log_fraction <= fraction:
                    self._next_log_fraction += self.log_fraction

    def set_social_welfare(self, value: float) -> None:
        """Update the latest validation metric without advancing progress."""

        self.latest_social_welfare = float(value)
        if self._bar is not None:
            self._bar.set_postfix(eval_sw=f"{value:+.3f}", refresh=False)

    def close(self) -> None:
        if self._closed:
            return
        if self._bar is not None:
            self._bar.close()
        self._closed = True

    def _write_log_line(self) -> None:
        elapsed = max(0.0, self.clock() - self.started_at)
        speed = self.completed / elapsed if elapsed > 0.0 else 0.0
        eta = (self.total - self.completed) / speed if speed > 0.0 else 0.0
        metric = (
            f", eval_sw={self.latest_social_welfare:+.3f}"
            if self.latest_social_welfare is not None
            else ""
        )
        print(
            f"[progress] {self.label}: {self.completed:,}/{self.total:,} "
            f"({100.0 * self.completed / self.total:5.1f}%), "
            f"{speed:,.1f} step/s, elapsed={_format_duration(elapsed)}, "
            f"eta={_format_duration(eta)}{metric}",
            file=self.stream,
            flush=True,
        )


__all__ = ["ProgressMode", "TrainingProgress", "resolve_progress_mode"]
