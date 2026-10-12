"""Shared file logging for experiment orchestration.

Exactness label: implementation convenience only.

The project logger writes timestamped records and optionally captures warnings.
Algorithm-specific training loggers remain available alongside this handler.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

__all__ = [
    "DEFAULT_LOG_PATH",
    "get_logger",
    "setup_run_logging",
]


DEFAULT_LOG_PATH: Final[Path] = Path("outputs/run.log")

_ROOT_LOGGER_NAME: Final[str] = "copsens_boyi"

# Module-level flag so repeated setup_run_logging() calls (e.g. within a
# single process that runs evaluate → plot) don't duplicate handlers.
_configured_paths: set[str] = set()


def setup_run_logging(
    log_path: str | Path = DEFAULT_LOG_PATH,
    *,
    level: int = logging.INFO,
    capture_warnings: bool = True,
) -> logging.Logger:
    """Attach a shared file handler for the copsens_boyi logger tree.

    Returns the root project logger. Safe to call more than once per process:
    a second call with the same ``log_path`` is a no-op.

    Parameters
    ----------
    log_path
        Destination file. Parent directories are created if needed.
    level
        Log level for the root copsens_boyi logger. Individual module loggers
        can override via :func:`logging.Logger.setLevel`.
    capture_warnings
        If True, also routes :func:`warnings.warn` through the logging system
        (as ``WARNING`` records from the ``py.warnings`` logger) so that
        warnings from the existing ``sweep.py`` / ``ppo.py`` / ``parallel.py``
        degradation paths also land in ``run.log``.
    """

    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    resolved = str(log_file.resolve())

    project_logger = logging.getLogger(_ROOT_LOGGER_NAME)
    project_logger.setLevel(level)
    # Don't propagate to root: we manage our own handlers.
    project_logger.propagate = False

    if resolved in _configured_paths:
        return project_logger

    handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    project_logger.addHandler(handler)

    if capture_warnings:
        # Make ``warnings.warn(..., RuntimeWarning)`` flow to the same file,
        # labelled ``py.warnings``. This keeps parallel.py's GPU-probe and
        # sweep.py's NaN-aggregation warnings from vanishing when stdout is
        # redirected elsewhere.
        logging.captureWarnings(True)
        py_warnings = logging.getLogger("py.warnings")
        # Attach our FileHandler directly to ``py.warnings`` so captured
        # warnings always land in run.log, independent of whatever (if
        # anything) is attached to the root logger. Without this, a bare
        # process with no root-logger handlers would silently drop them.
        if handler not in py_warnings.handlers:
            py_warnings.addHandler(handler)

    _configured_paths.add(resolved)
    project_logger.info("setup_run_logging attached handler at %s", resolved)
    return project_logger


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``copsens_boyi`` tree.

    If the caller passes ``__name__`` like ``"src.rl.train"``, we namespace it
    under the project root so all handlers added by :func:`setup_run_logging`
    apply uniformly.
    """

    if name.startswith(_ROOT_LOGGER_NAME):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")
