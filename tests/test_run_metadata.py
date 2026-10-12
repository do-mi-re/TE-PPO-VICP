"""Unit tests for :mod:`src.rl.run_metadata`.

The run-metadata collector is called from every training path in a
best-effort try/except, which means bugs in it would be silently logged.
These tests exercise the happy path plus the JSON-serialisation contract.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.rl import run_metadata
from src.rl.run_metadata import (
    collect_hardware_record,
    collect_run_metadata,
    hardware_identity,
    write_run_metadata,
)


class TestRunMetadata(unittest.TestCase):
    def test_collect_returns_required_sections(self) -> None:
        meta = collect_run_metadata()
        for key in ("recorded_at_utc", "hostname", "platform",
                    "python", "git", "packages", "cuda", "env"):
            self.assertIn(key, meta)
        # ``packages`` must list the libraries we rely on.
        for pkg in ("numpy", "torch", "stable-baselines3", "gymnasium"):
            self.assertIn(pkg, meta["packages"])

    def test_collect_output_is_json_serialisable(self) -> None:
        meta = collect_run_metadata()
        # Will raise if any value is not json-dumpable.
        serialized = json.dumps(meta, sort_keys=True)
        round_trip = json.loads(serialized)
        self.assertEqual(round_trip["hostname"], meta["hostname"])

    def test_write_run_metadata_creates_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            out_path = write_run_metadata(run_dir)
            self.assertTrue(out_path.exists())
            with out_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["hostname"], collect_run_metadata()["hostname"])

    def test_write_run_metadata_creates_missing_parent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "nested" / "subdir"
            out_path = write_run_metadata(run_dir)
            self.assertTrue(out_path.exists())
            self.assertEqual(out_path.parent, run_dir)

    def test_git_section_has_commit_or_error(self) -> None:
        meta = collect_run_metadata()
        git = meta["git"]
        # On repo checkouts we get ``commit``; on tarball installs without
        # ``git`` available we get ``error``. Exactly one is present.
        self.assertTrue(("commit" in git) ^ ("error" in git),
                        f"unexpected git payload: {git}")

    def test_system_ram_is_not_overwritten_by_gpu_memory(self) -> None:
        fake_psutil = SimpleNamespace(
            cpu_count=lambda logical=True: 96 if logical else 48,
            virtual_memory=lambda: SimpleNamespace(total=128 * 1024**3),
        )

        def command(args: list[str], **_: object) -> str | None:
            if args[0] == "nvidia-smi":
                return "\n".join(
                    ["NVIDIA GeForce RTX 3090, 24576, 555.0"] * 4
                )
            if args[:3] == ["git", "rev-parse", "HEAD"]:
                return "abc123"
            return None

        with (
            patch.dict(sys.modules, {"psutil": fake_psutil}),
            patch.object(run_metadata, "_fallback_host_capacity", return_value=(48, 96, 128.0)),
            patch.object(run_metadata, "_command", side_effect=command),
            patch.object(run_metadata, "_cpu_model", return_value="Example CPU"),
            patch.object(
                run_metadata,
                "_visible_cuda_info",
                return_value=(False, [], None, None, 1),
            ),
        ):
            record = collect_hardware_record()

        self.assertEqual(record["memory_total_gib"], 128.0)
        self.assertEqual(record["physical_gpu_count"], 4)
        self.assertEqual(record["physical_gpus"][0]["memory_total_gib"], 24.0)

    def test_hardware_identity_ignores_worker_gpu_pinning(self) -> None:
        record = {
            "machine": "x86_64",
            "system": "Linux",
            "cpu_model": "Example CPU",
            "physical_cpu_count": 48,
            "logical_cpu_count": 96,
            "memory_total_gib": 128.0,
            "cuda_visible_devices": "0",
            "gpus": [{"visible_index": 0, "name": "RTX 3090"}],
            "physical_gpus": [
                {"physical_index": index, "name": "RTX 3090", "memory_total_gib": 24.0}
                for index in range(4)
            ],
        }
        repinned = deepcopy(record)
        repinned["cuda_visible_devices"] = "3"
        repinned["git_dirty"] = True

        self.assertEqual(hardware_identity(record), hardware_identity(repinned))

    def test_capacity_fallback_is_used_without_psutil(self) -> None:
        with (
            patch.dict(sys.modules, {"psutil": None}),
            patch.object(run_metadata, "_fallback_host_capacity", return_value=(48, 96, 128.0)),
            patch.object(run_metadata, "_command", return_value=None),
            patch.object(run_metadata, "_cpu_model", return_value="Example CPU"),
            patch.object(
                run_metadata,
                "_visible_cuda_info",
                return_value=(False, [], None, None, 1),
            ),
        ):
            record = collect_hardware_record()
        self.assertEqual(record["physical_cpu_count"], 48)
        self.assertEqual(record["logical_cpu_count"], 96)
        self.assertEqual(record["memory_total_gib"], 128.0)


if __name__ == "__main__":
    unittest.main()
