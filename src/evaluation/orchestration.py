"""Shared orchestration API for experiment launchers.

Exactness label: implementation convenience only.

The adapter exposes run planning, workers, provenance, hardware records, and
manifest writing from run.py for the configured experiment scripts.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from run import (
    _MODEL_TASK_VRAM_MB,
    TrainPlanEntry,
    _append_manifest,
    _build_train_plan,
    _materialize_config,
    _new_run_id,
    _report_train_plan,
    _run_tasks_with_manifest,
    _train_worker,
    _training_task_cpu_slots,
    _utcnow,
    _write_training_record,
)
from src.evaluation.provenance import (
    TRAIN_FINGERPRINT_FILE,
    config_digest,
    train_freshness,
)

MODEL_TASK_VRAM_MB = _MODEL_TASK_VRAM_MB


def append_manifest(path: Path, record: dict[str, Any]) -> None:
    _append_manifest(path, record)


def utcnow() -> str:
    return _utcnow()


def new_run_id() -> str:
    return _new_run_id()


def run_tasks_with_manifest(
    tasks: list[Any],
    *,
    parallel: bool,
    max_workers: int | None,
    fail_fast: bool,
    manifest: Path,
    on_success: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    return _run_tasks_with_manifest(
        tasks,
        parallel=parallel,
        max_workers=max_workers,
        fail_fast=fail_fast,
        manifest=manifest,
        on_success=on_success,
    )


def materialize_config(
    spec_dict: dict[str, Any],
    eval_seeds: tuple[int, ...],
) -> dict[str, Any]:
    return _materialize_config(spec_dict, eval_seeds)


def training_task_cpu_slots(
    config: dict[str, Any],
    *,
    sb3_cpu_slots: int | None = None,
) -> int:
    """Return the scheduler reservation without changing training config."""

    return _training_task_cpu_slots(config, sb3_cpu_slots=sb3_cpu_slots)


def train_worker(
    spec_dict: dict[str, Any],
    eval_seeds: tuple[int, ...],
    output_root: str,
    execution_mode: str = "scheduled",
) -> dict[str, Any]:
    return _train_worker(
        spec_dict,
        eval_seeds,
        output_root,
        execution_mode,
    )


def build_train_plan(
    trainable: list[Any],
    output_root: str | Path,
    eval_seeds: tuple[int, ...],
    *,
    force: bool,
    retrain_stale: bool,
) -> list[TrainPlanEntry]:
    return _build_train_plan(
        trainable,
        output_root,
        eval_seeds,
        force=force,
        retrain_stale=retrain_stale,
    )


def report_train_plan(plan: list[TrainPlanEntry]) -> None:
    _report_train_plan(plan)


def write_training_record(result: dict[str, Any], *, run_id: str) -> None:
    _write_training_record(result, run_id=run_id)


__all__ = [
    "MODEL_TASK_VRAM_MB",
    "TRAIN_FINGERPRINT_FILE",
    "TrainPlanEntry",
    "append_manifest",
    "build_train_plan",
    "config_digest",
    "materialize_config",
    "new_run_id",
    "report_train_plan",
    "run_tasks_with_manifest",
    "train_freshness",
    "train_worker",
    "training_task_cpu_slots",
    "utcnow",
    "write_training_record",
]
