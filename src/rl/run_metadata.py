"""Canonical runtime metadata for reproducible experiments.

Exactness label: implementation convenience only. Metadata collection never
changes simulator, follower-solver, or training semantics.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as importlib_metadata
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:  # Keep diagnostics import-safe even outside a training environment.
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


__all__ = [
    "collect_hardware_record",
    "collect_run_metadata",
    "hardware_identity",
    "write_run_metadata",
]

HARDWARE_SCHEMA_VERSION = 1

_TRACKED_ENV_VARS: tuple[str, ...] = (
    "CUDA_VISIBLE_DEVICES",
    "PYTHONPATH",
    "PYTHONHASHSEED",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "TEPPO_STRICT_VECENV",
    "CUDA_DEVICE_ORDER",
    "TEPPO_CPU_CAPACITY",
    "TQDM_DISABLE",
)


def _command(args: list[str], *, cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            args, cwd=cwd, check=False, capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip() if result.returncode == 0 else ""
    return value or None


def _cpu_model() -> str:
    if platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    if platform.system() == "Darwin":
        value = _command(["sysctl", "-n", "machdep.cpu.brand_string"])
        if value:
            return value
    return platform.processor() or "unknown"


def _linux_memory_gib() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return float(line.split()[1]) / (1024**2)
    except (OSError, ValueError, IndexError):
        pass
    return None


def _linux_physical_cpu_count() -> int | None:
    """Best-effort physical-core count without third-party dependencies."""

    try:
        records = Path("/proc/cpuinfo").read_text(encoding="utf-8").split("\n\n")
    except OSError:
        return None
    cores: set[tuple[str, str]] = set()
    for record in records:
        fields: dict[str, str] = {}
        for line in record.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                fields[key.strip()] = value.strip()
        if "physical id" in fields and "core id" in fields:
            cores.add((fields["physical id"], fields["core id"]))
    return len(cores) or None


def _fallback_host_capacity() -> tuple[int | None, int, float | None]:
    logical = os.cpu_count() or 1
    if platform.system() == "Linux":
        return _linux_physical_cpu_count(), logical, _linux_memory_gib()
    if platform.system() == "Darwin":
        physical_raw = _command(["sysctl", "-n", "hw.physicalcpu"])
        logical_raw = _command(["sysctl", "-n", "hw.logicalcpu"])
        memory_raw = _command(["sysctl", "-n", "hw.memsize"])
        try:
            physical = int(physical_raw) if physical_raw else None
        except ValueError:
            physical = None
        try:
            logical = int(logical_raw) if logical_raw else logical
        except ValueError:
            pass
        try:
            memory = float(memory_raw) / (1024**3) if memory_raw else None
        except ValueError:
            memory = None
        return physical, logical, memory
    return None, logical, None


def _git_info(repo_root: Path) -> dict[str, Any]:
    commit = _command(["git", "rev-parse", "HEAD"], cwd=repo_root)
    branch = _command(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
    status = _command(["git", "status", "--porcelain"], cwd=repo_root)
    if commit is None:
        return {"error": "git metadata unavailable"}
    return {"commit": commit, "branch": branch or "unknown", "dirty": bool(status)}


def _package_versions() -> dict[str, str]:
    names = (
        "numpy", "torch", "stable-baselines3", "gymnasium", "pyyaml",
        "matplotlib", "psutil", "tqdm",
    )
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:  # pragma: no cover
            versions[name] = "<not installed>"
    return versions


def _visible_cuda_info(
) -> tuple[bool, list[dict[str, Any]], str | None, int | None, int | None]:
    if torch is None or not torch.cuda.is_available():
        return False, [], None, None, None
    devices: list[dict[str, Any]] = []
    for index in range(torch.cuda.device_count()):
        try:
            props = torch.cuda.get_device_properties(index)
            devices.append({
                "visible_index": index,
                "name": props.name,
                "memory_total_gib": round(props.total_memory / (1024**3), 3),
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count,
            })
        except Exception as exc:  # pragma: no cover - driver-specific
            devices.append({
                "visible_index": index,
                "error": f"{type(exc).__name__}: {exc}",
            })
    return (
        True,
        devices,
        torch.version.cuda,
        torch.backends.cudnn.version(),
        torch.get_num_threads(),
    )


def collect_hardware_record(*, repo_root: str | Path | None = None) -> dict[str, Any]:
    """Collect one flat, JSON-serializable host/runtime record."""

    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    repo_root = Path(repo_root)

    physical_cpus, logical_cpus, system_memory_gib = _fallback_host_capacity()
    try:
        import psutil

        physical_cpus = psutil.cpu_count(logical=False) or physical_cpus
        logical_cpus = psutil.cpu_count(logical=True) or logical_cpus
        system_memory_gib = psutil.virtual_memory().total / (1024**3)
    except ImportError:  # pragma: no cover - fallback helpers remain available
        pass

    cuda_available, visible_gpus, cuda_version, cudnn_version, torch_threads = (
        _visible_cuda_info()
    )
    physical_gpu_rows = _command([
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ])
    physical_gpus: list[dict[str, Any]] = []
    if physical_gpu_rows:
        for physical_index, line in enumerate(physical_gpu_rows.splitlines()):
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 3:
                continue
            name, memory_mib, driver = parts
            try:
                gpu_memory_gib = round(float(memory_mib) / 1024, 3)
            except ValueError:
                gpu_memory_gib = None
            physical_gpus.append({
                "physical_index": physical_index,
                "name": name,
                "memory_total_gib": gpu_memory_gib,
                "driver": driver,
            })

    git = _git_info(repo_root)
    return {
        "hardware_schema_version": HARDWARE_SCHEMA_VERSION,
        "hostname": socket.gethostname(),
        "machine": platform.machine(),
        "system": platform.system(),
        "release": platform.release(),
        "cpu_model": _cpu_model(),
        "physical_cpu_count": physical_cpus,
        "logical_cpu_count": logical_cpus,
        "memory_total_gib": (
            round(system_memory_gib, 3) if system_memory_gib is not None else None
        ),
        "python": platform.python_version(),
        "torch": torch.__version__ if torch is not None else "unavailable",
        "torch_threads": torch_threads,
        "cuda_available": cuda_available,
        "cuda": cuda_version,
        "cudnn": cudnn_version,
        "nvidia_driver": physical_gpus[0]["driver"] if physical_gpus else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_count": len(visible_gpus),
        "gpus": visible_gpus,
        "physical_gpu_count": len(physical_gpus),
        "physical_gpus": physical_gpus,
        "git_commit": git.get("commit", "unknown"),
        "git_dirty": bool(git.get("dirty", False)),
        "timing_clock": "time.perf_counter/time.monotonic",
    }


def hardware_identity(record: dict[str, Any]) -> str:
    """Return a stable physical-host ID, independent of worker GPU pinning."""

    payload = {
        key: record.get(key)
        for key in (
            "machine", "system", "cpu_model", "physical_cpu_count",
            "logical_cpu_count", "memory_total_gib",
        )
    }
    payload["physical_gpus"] = [
        {"name": gpu.get("name"), "memory_total_gib": gpu.get("memory_total_gib")}
        for gpu in record.get("physical_gpus", [])
    ]
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def collect_run_metadata(*, repo_root: str | Path | None = None) -> dict[str, Any]:
    """Build the complete reproducibility record written beside a run."""

    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    repo_root = Path(repo_root)
    hardware = collect_hardware_record(repo_root=repo_root)
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": hardware["hostname"],
        "platform": platform.platform(),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "git": _git_info(repo_root),
        "packages": _package_versions(),
        "cpu": {
            "physical_count": hardware["physical_cpu_count"],
            "logical_count": hardware["logical_cpu_count"],
            "memory_total_gb": hardware["memory_total_gib"],
            "torch_num_threads": hardware["torch_threads"],
            "torch_num_interop_threads": (
                torch.get_num_interop_threads() if torch is not None else None
            ),
        },
        "cuda": {
            "available": hardware["cuda_available"],
            "cuda_runtime": hardware["cuda"],
            "device_count": hardware["gpu_count"],
            "devices": hardware["gpus"],
        },
        "hardware": hardware,
        "hardware_id": hardware_identity(hardware),
        "numpy_random_bit_generator": type(np.random.default_rng().bit_generator).__name__,
        "env": {name: os.environ.get(name) for name in _TRACKED_ENV_VARS},
    }


def write_run_metadata(run_dir: str | Path, *, repo_root: str | Path | None = None) -> Path:
    """Write ``run_metadata.json`` atomically and return its path."""

    run_dir_path = Path(run_dir)
    run_dir_path.mkdir(parents=True, exist_ok=True)
    out_path = run_dir_path / "run_metadata.json"
    try:
        payload = collect_run_metadata(repo_root=repo_root)
    except Exception as exc:  # pragma: no cover
        payload = {"error": f"collect_run_metadata failed: {type(exc).__name__}: {exc}"}
    temporary = run_dir_path / f".run_metadata.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(out_path)
    return out_path
