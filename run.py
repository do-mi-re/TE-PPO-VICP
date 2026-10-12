"""Shared experiment workers and training plans.

Exactness label: implementation convenience only.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

# cuBLAS reductions (matmul, bmm) are non-deterministic on Ampere+ GPUs unless
# this env var is set *before* torch/cuBLAS is first imported. ``run.py`` is
# imported by experiment launchers, so this setting reaches the main process
# (not just SubprocVecEnv children) sees it. ``setdefault`` lets the shell
# override at launch time without losing our default.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

from src.evaluation.provenance import (
    TRAIN_FINGERPRINT_FILE as _TRAIN_FINGERPRINT_FILE,
)
from src.evaluation.provenance import (
    config_digest as _train_config_digest,
)
from src.evaluation.provenance import (
    train_freshness as _train_freshness,
)
from src.logging_utils import get_logger

_logger = get_logger("run")


def _append_manifest(path: Path, record: dict) -> None:
    """Append one JSON line. Each call opens/closes so a crash leaves a
    parseable prefix behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_run_id() -> str:
    """Return a sortable, collision-resistant invocation identifier."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def _run_tasks_with_manifest(
    tasks: list,  # list[TaskSpec]
    *,
    parallel: bool,
    max_workers: int | None,
    fail_fast: bool,
    manifest: Path,
    on_success: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Execute tasks (parallel or serial) and stream events to ``manifest``.

    Shared between TRAIN and EVALUATE so their failure-handling and manifest
    format stay in lockstep. ``on_success`` persists per-task metadata while
    results are still available, including when a sibling parallel task fails.
    Raises ``SystemExit(1)`` on parallel failure and re-raises on serial
    failure; both paths flush terminal events first.
    """
    from src.parallel import ParallelTaskErrors, run_parallel_tasks

    if parallel:
        try:
            results = run_parallel_tasks(
                tasks, max_workers=max_workers, fail_fast=fail_fast,
            )
        except ParallelTaskErrors as exc:
            done_ts = _utcnow()
            for result in exc.results:
                if result.get("status") != "ok":
                    continue
                _append_manifest(manifest, {
                    "event": "done", "label": result.get("label"),
                    "status": "ok", "elapsed_s": result.get("elapsed_s"),
                    "timestamp": done_ts,
                })
                if on_success is not None:
                    on_success(result)
            print(f"\n  ERROR: {exc}", file=sys.stderr)
            for f in exc.failures:
                print(f"    {f['label']}: {f.get('error_type', '?')}: "
                      f"{f.get('error', '?')}", file=sys.stderr)
                tb = f.get("error_traceback")
                if tb:
                    print("\n".join("      " + line
                                    for line in tb.rstrip().splitlines()),
                          file=sys.stderr)
                _append_manifest(manifest, {
                    "event": "failed", "label": f.get("label"),
                    "error_type": f.get("error_type"), "error": f.get("error"),
                    "elapsed_s": f.get("elapsed_s"), "timestamp": done_ts,
                })
            raise SystemExit(1) from exc
    else:
        results = []
        for task in tasks:
            start = time.time()
            try:
                out = task.func(*task.args, **task.kwargs)
                result = {
                    "status": "ok", "label": task.label,
                    "result": out, "elapsed_s": time.time() - start,
                }
                results.append(result)
                _append_manifest(manifest, {
                    "event": "done", "label": result.get("label"),
                    "status": "ok", "elapsed_s": result.get("elapsed_s"),
                    "timestamp": _utcnow(),
                })
                if on_success is not None:
                    on_success(result)
            except Exception as exc:  # noqa: BLE001 — propagate after manifest write
                _append_manifest(manifest, {
                    "event": "failed", "label": task.label,
                    "error_type": type(exc).__name__, "error": str(exc),
                    "elapsed_s": time.time() - start, "timestamp": _utcnow(),
                })
                raise

    if parallel:
        done_ts = _utcnow()
        for r in results:
            _append_manifest(manifest, {
                "event": "done", "label": r.get("label"),
                "status": r.get("status"), "elapsed_s": r.get("elapsed_s"),
                "timestamp": done_ts,
            })
            if r.get("status") == "ok" and on_success is not None:
                on_success(r)
    return results


# ---------------------------------------------------------------------------
# Config materialization (shared by train & evaluate workers)
# ---------------------------------------------------------------------------

def _materialize_config(spec_dict: dict, eval_seeds: tuple[int, ...]) -> dict:
    """Build the full training/eval config for one RunSpec.

    Used by training workers and expected-config checks during evaluation.
    The plain spec dictionary keeps the worker payload small.
    """
    from src.evaluation.sweep import (
        build_sweep_config,
        load_algorithm_config,
    )

    base_config = load_algorithm_config(spec_dict["algorithm"])
    config = build_sweep_config(
        spec_dict["algorithm"],
        base_config=base_config,
        scenario_overrides=spec_dict["scenario_overrides"],
        train_seed=spec_dict["train_seed"],
        eval_seeds=eval_seeds,
        run_name_suffix=spec_dict["run_name_suffix"],
    )
    for dotted, value in (spec_dict.get("config_overrides") or {}).items():
        parts = str(dotted).split(".")
        node = config
        for part in parts[:-1]:
            child = node.setdefault(part, {})
            if not isinstance(child, dict):
                raise ValueError(
                    f"cannot apply config override {dotted!r}: {part!r} is not a mapping"
                )
            node = child
        node[parts[-1]] = value
    return config


# ---------------------------------------------------------------------------
# TRAIN stage
# ---------------------------------------------------------------------------

# Per-task GPU admission and reservation budget for the configured [256,256]
# policies. Override TEPPO_MODEL_VRAM_MB for larger networks or batches.
# DBAS is a CPU heuristic and uses a zero GPU reservation.
_MODEL_TASK_VRAM_MB = float(os.environ.get("TEPPO_MODEL_VRAM_MB", "1500"))


def _training_task_cpu_slots(
    config: dict,
    *,
    sb3_cpu_slots: int | None = None,
) -> int:
    """Return scheduler CPU slots for one materialized training run.

    SB3 leaders own ``n_envs`` environment subprocesses plus their trainer.
    By default their process count is used as the conservative reservation.
    Experiment orchestration may instead pass an effective-core reservation;
    this changes only outer scheduling and never the materialized training
    config. JDLS/MAPPO use an in-process batched rollout and therefore need one
    slot; native Torch/BLAS threads are separately capped by :class:`TaskSpec`.
    """

    algorithm = config.get("algorithm", {})
    if "n_envs" in algorithm:
        process_slots = max(1, int(algorithm["n_envs"])) + 1
        if sb3_cpu_slots is None:
            return process_slots
        requested = int(sb3_cpu_slots)
        if not 1 <= requested <= process_slots:
            raise ValueError(
                "sb3_cpu_slots must lie between 1 and the SB3 trainer/process "
                f"count ({process_slots}), got {requested}"
            )
        return requested
    return 1

# Workers persist a configuration fingerprint beside the selected checkpoint.
def _train_worker(spec_dict: dict, eval_seeds: tuple[int, ...], output_root: str, execution_mode: str = "scheduled") -> dict:
    """Picklable worker: train exactly one spec; no evaluation."""
    from src.rl.run_metadata import collect_hardware_record
    from src.rl.train import train_from_config

    config = _materialize_config(spec_dict, eval_seeds)
    summary = train_from_config(config, output_root=output_root)
    # Persist the config fingerprint AFTER a successful train (a crashed run thus
    # leaves none -> retrains). Best-effort: a write failure just forces a
    # retrain next time, never a stale skip.
    digest = _train_config_digest(config)
    if digest is not None:
        try:
            (Path(summary["run_dir"]) / _TRAIN_FINGERPRINT_FILE).write_text(
                digest, encoding="utf-8")
        except OSError:
            pass
    # Discard paths that don't outlive the worker to keep the pickled return
    # value small.
    return {
        "run_name": spec_dict["run_name"],
        "run_dir": str(summary["run_dir"]),
        "algorithm": spec_dict["algorithm"],
        "train_seed": spec_dict["train_seed"],
        "purpose": spec_dict.get("purpose"),
        "config_digest": digest,
        "execution_mode": execution_mode,
        "hardware": collect_hardware_record(),
    }


@dataclass(frozen=True)
class TrainPlanEntry:
    """One immutable decision in a train invocation."""

    spec: object
    run_dir: Path
    expected_digest: str | None
    freshness: str  # missing | partial | fresh | stale
    action: str  # skip | train | retrain | blocked


def _build_train_plan(
    trainable: list,
    output_root: str | Path,
    eval_seeds: tuple[int, ...],
    *,
    force: bool,
    retrain_stale: bool,
) -> list[TrainPlanEntry]:
    """Classify specs once for dry-run reporting and real execution."""
    plan: list[TrainPlanEntry] = []
    for spec in trainable:
        run_dir = spec.run_dir(output_root)
        expected_config = _materialize_config(
            spec.to_worker_dict(), eval_seeds
        )
        expected = _train_config_digest(expected_config)
        if not run_dir.exists():
            freshness = "missing"
        elif not (run_dir / "summary.json").exists():
            freshness = "partial"
        else:
            freshness = _train_freshness(
                run_dir, expected, expected_config=expected_config
            )

        if force:
            action = "retrain" if run_dir.exists() else "train"
        elif freshness == "fresh":
            action = "skip"
        elif freshness == "missing":
            action = "train"
        elif retrain_stale:
            action = "retrain"
        else:
            action = "blocked"
        plan.append(TrainPlanEntry(spec, run_dir, expected, freshness, action))
    return plan


def _report_train_plan(plan: list[TrainPlanEntry]) -> None:
    """Print a precomputed plan without touching artifacts or manifests."""
    counts = {
        name: sum(entry.freshness == name for entry in plan)
        for name in ("fresh", "missing", "partial", "stale")
    }
    actions = {
        name: sum(entry.action == name for entry in plan)
        for name in ("skip", "train", "retrain", "blocked")
    }
    print(f"  Total runs required: {len(plan)}")
    print(f"    fresh                  : {counts['fresh']}")
    print(f"    missing                : {counts['missing']}")
    print(f"    partial                : {counts['partial']}")
    print(f"    stale                  : {counts['stale']}")
    print(
        "    would skip/train/retrain: "
        f"{actions['skip']}/{actions['train']}/{actions['retrain']}"
    )
    print(f"    BLOCKED                : {actions['blocked']}")
    blocked = [entry for entry in plan if entry.action == "blocked"]
    for entry in blocked[:20]:
        print(f"      {entry.freshness}: {entry.spec.run_name}")
    if len(blocked) > 20:
        print(f"      ... and {len(blocked) - 20} more")
    print("\n  (planning is read-only: nothing was trained or deleted)")


def _write_training_record(result: dict, *, run_id: str) -> None:
    """Atomically persist scheduler timing beside a successfully trained run."""
    payload = result.get("result") or {}
    run_dir = Path(payload["run_dir"])
    elapsed_s = float(result.get("elapsed_s", 0.0))
    finished = datetime.now(timezone.utc)
    scenario: dict = {}
    algorithm_config: dict = {}
    try:
        import yaml
        snapshot = yaml.safe_load(
            (run_dir / "config_snapshot.yaml").read_text(encoding="utf-8")
        ) or {}
        scenario = dict(snapshot.get("scenario", {}))
        algorithm_config = dict(snapshot.get("algorithm", {}))
    except (OSError, ValueError, TypeError):
        pass
    urgency_profiles = scenario.get("urgency_profiles")
    num_tasks = len(urgency_profiles) if urgency_profiles else 3
    try:
        import torch
        torch_version = torch.__version__
    except ImportError:  # pragma: no cover - training itself requires torch
        torch_version = "unavailable"
    hardware = dict(payload.get("hardware") or {})
    hardware_payload = hardware or {
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "logical_cpu_count": os.cpu_count(),
        "torch": torch_version,
        "device": algorithm_config.get("device", "unknown"),
    }
    from src.rl.run_metadata import hardware_identity
    hardware_id = hardware_identity(hardware_payload)
    record = {
        "schema_version": 2,
        "run_id": run_id,
        "run_name": payload["run_name"],
        "algorithm": payload["algorithm"],
        "train_seed": payload["train_seed"],
        "config_digest": payload.get("config_digest"),
        "purpose": payload.get("purpose"),
        "started_at": (finished - timedelta(seconds=elapsed_s)).isoformat(),
        "finished_at": finished.isoformat(),
        "elapsed_s": elapsed_s,
        "total_timesteps": algorithm_config.get("total_timesteps"),
        "num_vehicles": scenario.get("num_vehicles", 20),
        "num_tasks": num_tasks,
        "device": algorithm_config.get("device", "unknown"),
        "torch_version": torch_version,
        "logical_cpu_count": os.cpu_count(),
        "hardware_id": hardware_id,
        "hardware": hardware,
        "timing_clock": "time.monotonic",
        "execution_mode": payload.get("execution_mode", "scheduled"),
        "artifact_path": str(run_dir),
    }
    target = run_dir / "training_record.json"
    temporary = run_dir / f".training_record.{os.getpid()}.tmp"
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(target)
