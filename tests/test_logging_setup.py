"""Unit tests for :mod:`src.logging_utils`.

The logging helper must be
safe to call multiple times (idempotent), create missing parent directories,
and route ``warnings.warn`` output into the run log. These properties are
easy to regress because Python's ``logging`` module has a lot of shared
state; the tests below pin each of them down.
"""

from __future__ import annotations

import logging
import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import src.logging_utils as logging_utils
from src.logging_utils import (
    _ROOT_LOGGER_NAME,
    get_logger,
    setup_run_logging,
)


class TestSetupRunLogging(unittest.TestCase):
    """Exercise idempotency, directory creation, and warnings routing."""

    def setUp(self) -> None:  # noqa: D401 — test setup
        # Snapshot and clear shared logger state so each test starts clean.
        project_logger = logging.getLogger(_ROOT_LOGGER_NAME)
        self._saved_handlers = list(project_logger.handlers)
        self._saved_level = project_logger.level
        self._saved_propagate = project_logger.propagate
        project_logger.handlers.clear()

        py_warnings_logger = logging.getLogger("py.warnings")
        self._saved_py_warnings_handlers = list(py_warnings_logger.handlers)
        py_warnings_logger.handlers.clear()

        self._saved_configured = set(logging_utils._configured_paths)
        logging_utils._configured_paths.clear()

    def tearDown(self) -> None:  # noqa: D401 — test cleanup
        project_logger = logging.getLogger(_ROOT_LOGGER_NAME)
        for handler in project_logger.handlers:
            handler.close()
        project_logger.handlers[:] = self._saved_handlers
        project_logger.setLevel(self._saved_level)
        project_logger.propagate = self._saved_propagate

        py_warnings_logger = logging.getLogger("py.warnings")
        for handler in py_warnings_logger.handlers:
            handler.close()
        py_warnings_logger.handlers[:] = self._saved_py_warnings_handlers

        logging_utils._configured_paths.clear()
        logging_utils._configured_paths.update(self._saved_configured)
        logging.captureWarnings(False)

    def test_second_call_same_path_is_idempotent(self) -> None:
        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "run.log"
            setup_run_logging(log_path)
            project_logger = logging.getLogger(_ROOT_LOGGER_NAME)
            handler_count_first = len(project_logger.handlers)

            setup_run_logging(log_path)
            handler_count_second = len(project_logger.handlers)

        self.assertEqual(handler_count_first, handler_count_second)
        self.assertEqual(handler_count_first, 1)

    def test_creates_missing_parent_directory(self) -> None:
        with TemporaryDirectory() as tmp:
            # Parent directory does not yet exist; setup must create it.
            log_path = Path(tmp) / "nested" / "dir" / "run.log"
            self.assertFalse(log_path.parent.exists())

            setup_run_logging(log_path)

            self.assertTrue(log_path.parent.is_dir())

    def test_capture_warnings_routes_into_log_file(self) -> None:
        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "run.log"
            setup_run_logging(log_path)

            # Emit a warning that ``captureWarnings(True)`` should route
            # through ``py.warnings`` and into our FileHandler.
            warnings.warn("probe warning surface", RuntimeWarning, stacklevel=2)

            # Flush all handlers so the buffered text lands on disk before
            # reading back.
            for handler in logging.getLogger(_ROOT_LOGGER_NAME).handlers:
                handler.flush()
            for handler in logging.getLogger("py.warnings").handlers:
                handler.flush()

            contents = log_path.read_text(encoding="utf-8")

        self.assertIn("probe warning surface", contents)

    def test_get_logger_namespaces_under_project_root(self) -> None:
        logger = get_logger("subsystem")
        self.assertEqual(logger.name, f"{_ROOT_LOGGER_NAME}.subsystem")

        already_prefixed = get_logger(f"{_ROOT_LOGGER_NAME}.other")
        self.assertEqual(already_prefixed.name, f"{_ROOT_LOGGER_NAME}.other")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
