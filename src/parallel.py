"""Resource-aware parallel task scheduler.

Detects available GPU/CPU resources and schedules training/evaluation
tasks to maximize resource utilization.

Scheduler semantics
-------------------
- **CPU gating**: instantaneous idle capacity is sampled once at scheduler
  startup, then in-flight ``cpus_needed`` reservations gate each submission.
  This avoids double-counting our own workers through the lagging one-minute
  load average. A task is always allowed when the pending queue is empty
  (escape hatch for ``cpus_needed > available capacity``).
- **GPU gating**: the driver's live ``torch.cuda.mem_get_info`` is re-probed
  once per scheduling tick. For each in-flight task still within
  :data:`GPU_ALLOCATION_GRACE_S` seconds of ``executor.submit`` (i.e. its
  worker hasn't yet called ``.to('cuda')``), the task's reserve is
  subtracted from the probed free so two back-to-back submissions on the
  same GPU don't both count the full pre-allocation free memory. After
  the grace window we trust the driver to reflect the allocation. This
  design reacts to co-tenant VRAM changes on shared servers — capacity
  expands when they release, shrinks when they claim.
- **Feasibility fail-fast**: at startup we compute a *theoretical* per-GPU
  capacity (``total_memory - safety_margin``). If any task's
  ``vram_needed_mb`` exceeds the largest such capacity, we raise
  ``ValueError`` immediately — no amount of co-tenant churn can make it
  fit.
- **Multi-GPU correctness**: when more than one GPU is present, each task
  runs in a fresh worker process (``max_tasks_per_child=1``). Otherwise a
  reused worker's first CUDA initialization would pin it to whichever GPU
  it saw first, making later ``CUDA_VISIBLE_DEVICES`` assignments no-ops.
- **Stable first-fit queueing**: each tick scans queued tasks in original
  order and submits the first one that fits current CPU/GPU reservations.
  A temporarily blocked large/GPU task therefore cannot idle resources that
  a later small/CPU-only task can use; result order remains input order.
- **Failure handling**: by default all tasks run to completion; any
  failures are surfaced via :class:`ParallelTaskErrors` at the end. Pass
  ``fail_fast=True`` to abort on the first failing task (queued work is
  cancelled; running workers finish because ``ProcessPoolExecutor``
  cannot kill them). ``KeyboardInterrupt`` follows the same abort path.
"""

from __future__ import annotations

import multiprocessing
import os
import time
import traceback
import warnings
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, TypedDict

import torch
from tqdm import tqdm

from src.logging_utils import get_logger

_logger = get_logger("parallel")

# A GPU with at least this much free memory (MB) is considered available
# for another training run.
GPU_FREE_THRESHOLD_MB = 5_000  # 5 GB

# Headroom left per GPU for the CUDA context, cuDNN workspaces, driver
# overhead, and anything else outside PyTorch's allocator. 512 MB is
# comfortable for typical RL workloads.
GPU_SAFETY_MARGIN_MB = 512.0

# Time window during which a just-submitted task's reserve is subtracted
# from the driver's live free-memory probe. ``spawn`` + ``import torch`` +
# CUDA context init + model ``.to('cuda')`` typically takes 3-10 s, but under
# heavy CPU contention (this is a CPU-bound workload — many env-worker
# processes competing for cores) a worker's cold start can stretch well past
# that, so the default is a conservative 30 s. The window gives headroom so
# back-to-back submissions on the same GPU don't both think the full
# pre-allocation free memory is available. After it, the driver's probe is
# trusted to reflect the allocation and the reserve stops being subtracted.
#
# Known limitation: if a task's cold start exceeds the window and the driver's
# probe still doesn't show the allocation, the scheduler can over-admit and
# OOM. Raise it via ``TEPPO_GPU_ALLOC_GRACE_S`` if that happens; the only cost
# is slower admission when genuine capacity has freed up.
GPU_ALLOCATION_GRACE_S = float(os.environ.get("TEPPO_GPU_ALLOC_GRACE_S", "30"))

# GPU indices whose probe already failed once in this process. Without this,
# ``_find_free_gpu`` — which runs every scheduling tick — would emit the same
# ``RuntimeWarning`` hundreds of times during a large sweep on a broken device.
_gpu_probe_failed: set[int] = set()


def _cpu_admission_capacity(resources: SystemResources) -> int:
    """Resolve usable CPU slots, honoring an optional server/cgroup cap."""

    detected = max(1, int(resources.available_cpus))
    configured = os.environ.get("TEPPO_CPU_CAPACITY")
    if configured is None:
        return detected
    try:
        requested = int(configured)
    except ValueError:
        warnings.warn(
            f"ignoring invalid TEPPO_CPU_CAPACITY={configured!r}",
            RuntimeWarning,
            stacklevel=2,
        )
        return detected
    if requested < 1:
        warnings.warn(
            f"ignoring non-positive TEPPO_CPU_CAPACITY={requested}",
            RuntimeWarning,
            stacklevel=2,
        )
        return detected
    return min(detected, requested)


def _warn_gpu_probe_once(index: int, exc: Exception, *, reason: str) -> None:
    """Emit a RuntimeWarning for a GPU probe failure, at most once per index."""

    if index in _gpu_probe_failed:
        return
    warnings.warn(
        f"GPU {index} probe failed ({type(exc).__name__}: {exc}); {reason}.",
        RuntimeWarning,
        stacklevel=3,
    )
    _gpu_probe_failed.add(index)


# ---------------------------------------------------------------------------
# Resource detection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GPUInfo:
    index: int
    total_memory_mb: float
    free_memory_mb: float


@dataclass(frozen=True)
class SystemResources:
    cpu_count: int
    load_avg_1m: float
    available_cpus: float
    gpus: tuple[GPUInfo, ...] = ()


def get_system_resources() -> SystemResources:
    """Snapshot current system resource availability."""
    cpu_count = os.cpu_count() or 1
    try:
        load_1m = os.getloadavg()[0]
    except (OSError, AttributeError):
        load_1m = 0.0
    # A one-minute load average is valuable diagnostic metadata but is a poor
    # admission signal for a batch scheduler: completed workers remain in it
    # for minutes. Prefer an instantaneous sample when psutil is available.
    try:
        import psutil

        utilization = float(psutil.cpu_percent(interval=0.1)) / 100.0
        available_cpus = max(0.0, cpu_count * (1.0 - utilization))
    except Exception:  # pragma: no cover - optional diagnostic dependency
        available_cpus = max(0.0, cpu_count - load_1m)

    gpus: list[GPUInfo] = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                free, total = torch.cuda.mem_get_info(i)
                gpus.append(GPUInfo(
                    index=i,
                    total_memory_mb=total / (1024 * 1024),
                    free_memory_mb=free / (1024 * 1024),
                ))
            except Exception as exc:
                # Surface driver/config failures instead of swallowing them,
                # but dedupe so a persistently broken probe doesn't flood the log.
                _warn_gpu_probe_once(i, exc, reason="treating as unavailable")

    return SystemResources(
        cpu_count=cpu_count,
        load_avg_1m=load_1m,
        available_cpus=available_cpus,
        gpus=tuple(gpus),
    )


def _cpu_can_accept(
    cpus_needed: int,
    pending_cpu_total: int = 0,
    *,
    cpu_count: int | None = None,
) -> bool:
    """Return whether the scheduler's CPU reservation can fit another task.

    ``cpu_count`` is the admission capacity captured at scheduler startup,
    after subtracting instantaneous external utilization. During the run we
    use only explicit in-flight reservations; consulting ``getloadavg`` here
    would count settled workers twice and stall later waves for minutes.

    *cpu_count* defaults to ``os.cpu_count()`` but callers may supply the
    value from a resource snapshot (e.g. ``SystemResources.cpu_count``) to
    ensure consistency with the snapshot used to initialise the scheduler.
    """
    if cpu_count is None:
        cpu_count = os.cpu_count() or 1
    return (cpu_count - pending_cpu_total) >= cpus_needed


def _theoretical_gpu_capacity(gpus: tuple[GPUInfo, ...]) -> dict[int, float]:
    """Return the hardware-max per-GPU capacity used for the feasibility check.

    ``capacity[i] = total_memory - safety_margin``. Does NOT clamp by the
    initial free-memory snapshot — on a shared server we want a task that
    needs, say, 18 GB on a 24 GB card to wait for a co-tenant to release
    memory rather than fail the sweep at startup.
    """

    return {
        g.index: max(0.0, g.total_memory_mb - GPU_SAFETY_MARGIN_MB)
        for g in gpus
    }


def _probe_gpu_free_memory(indices: list[int]) -> dict[int, float]:
    """Re-probe the driver for live free VRAM on each GPU in ``indices``.

    Returns ``{i: free_mb}`` for every GPU whose probe succeeded. Failed
    probes are routed through :func:`_warn_gpu_probe_once` (dedup'd) and
    omitted from the returned dict; the scheduler then treats that GPU as
    unavailable for this tick and retries next tick.
    """

    if not torch.cuda.is_available():
        return {}
    free_mb: dict[int, float] = {}
    for i in indices:
        try:
            free_bytes, _ = torch.cuda.mem_get_info(i)
        except Exception as exc:
            _warn_gpu_probe_once(i, exc, reason="skipping for this scheduling tick")
            continue
        free_mb[i] = free_bytes / (1024 * 1024)
    return free_mb


def _find_free_gpu(
    effective_free_mb: dict[int, float],
    vram_needed_mb: float = GPU_FREE_THRESHOLD_MB,
) -> int | None:
    """Pick the GPU whose effective free memory is largest and >= ``vram_needed_mb``.

    Pure function. Ties break by :class:`dict` iteration order (insertion
    order in CPython). Returns ``None`` when no GPU fits. The scheduler
    owns computing ``effective_free_mb``; see the module docstring.
    """

    best_idx: int | None = None
    best_free: float = 0.0
    for i, free in effective_free_mb.items():
        if free >= vram_needed_mb and free > best_free:
            best_idx = i
            best_free = free
    return best_idx


# ---------------------------------------------------------------------------
# Task wrapper (runs in subprocess)
# ---------------------------------------------------------------------------

def _worker_wrapper(
    func: Callable[..., Any],
    args: tuple,
    kwargs: dict[str, Any],
    cuda_device: str | None,
    task_label: str,
    cpu_threads: int,
) -> dict[str, Any]:
    """Execute one task with GPU isolation and bounded native thread pools.

    Independent experiment processes provide the outer parallelism.  Letting
    every tiny policy network start ``cpu_count`` OpenMP threads causes severe
    oversubscription on multi-run servers, so each task gets an explicit
    intra-process thread budget.  Subprocess vector environments inherit the
    same limit.
    """
    # Child trainers select stable 5%-interval progress lines instead of
    # competing dynamic tqdm bars.  Each scheduled worker is process-isolated,
    # so these runtime-only markers cannot leak into the parent process.
    os.environ["TEPPO_PARALLEL_WORKER"] = "1"
    if cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    cpu_threads = max(1, int(cpu_threads))
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[variable] = str(cpu_threads)
    torch.set_num_threads(cpu_threads)
    try:
        # Fresh workers normally permit this call. Keep the wrapper robust if
        # a library initialized the inter-op pool during module import.
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    t0 = time.time()
    try:
        result = func(*args, **kwargs)
        elapsed = time.time() - t0
        return {
            "status": "ok",
            "label": task_label,
            "result": result,
            "elapsed_s": elapsed,
        }
    except Exception as exc:
        # Preserve the full traceback so the parent process can surface a
        # real stack instead of just an error string. ``traceback.format_exc``
        # must be called inside the ``except`` block while ``sys.exc_info`` is
        # populated.
        return {
            "status": "error",
            "label": task_label,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "error_traceback": traceback.format_exc(),
            "elapsed_s": time.time() - t0,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class TaskSpec:
    """Describes one parallelizable unit of work."""
    func: Callable[..., Any]
    args: tuple = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    label: str = ""
    cpus_needed: int = 2
    # Admission gate: scheduler only submits when
    # cpu_count - load_avg - pending_cpu_total >= cpus_needed.
    cpu_threads: int = 1
    # Native/OpenMP threads used by the task's main process. Environment
    # subprocesses inherit this value. Keep this at 1 when outer task/env
    # parallelism is already active.
    vram_needed_mb: float = GPU_FREE_THRESHOLD_MB
    # Admission gate (GPU): the scheduler only assigns this task to a GPU whose
    # ``capacity - reserved`` is >= this value.
    vram_reserve_mb: float | None = None
    # How much VRAM to deduct from the GPU's budget while this task runs.
    # ``None`` (default) means use ``vram_needed_mb``. Set this to a smaller
    # value when the task's actual VRAM footprint is less than the admission
    # gate so that multiple tasks can share the same GPU.


class TaskSuccess(TypedDict):
    """Result payload for a task that completed without raising."""

    status: Literal["ok"]
    label: str
    result: Any
    elapsed_s: float


class TaskFailure(TypedDict):
    """Result payload for a task that raised an exception inside the worker."""

    status: Literal["error"]
    label: str
    error: str
    error_type: str
    error_traceback: str
    elapsed_s: float


class TaskCancelled(TypedDict):
    """Result payload for a task interrupted by ``KeyboardInterrupt``."""

    status: Literal["cancelled"]
    label: str
    error: str
    error_type: str
    error_traceback: str
    elapsed_s: float


# Every in-band result the scheduler surfaces. Callers can narrow by the
# ``status`` tag; the unified shape (``label``, ``elapsed_s``) is always
# present, which is what :mod:`run.py` and test fakes rely on.
TaskResult = TaskSuccess | TaskFailure | TaskCancelled


class ParallelTaskErrors(RuntimeError):
    """Raised when one or more parallel tasks fail."""

    def __init__(
        self,
        failures: list[TaskFailure],
        results: list[TaskResult] | None = None,
    ) -> None:
        labels = [f["label"] for f in failures]
        super().__init__(
            f"{len(failures)} task(s) failed: {', '.join(labels)}"
        )
        self.failures = failures
        # Preserve successes completed in the same invocation.  Orchestrators
        # need them to write complete timing manifests/sidecars even when a
        # sibling task fails.
        self.results = list(results or failures)


def run_parallel_tasks(
    tasks: list[TaskSpec],
    *,
    max_workers: int | None = None,
    device_strategy: str = "auto",
    fail_fast: bool = False,
) -> list[TaskResult]:
    """Run tasks in parallel with resource-aware scheduling.

    Parameters
    ----------
    tasks : list of TaskSpec
    max_workers : hard cap on concurrent tasks (None = resource-gated only)
    device_strategy : "auto" | "cpu" | "gpu"
    fail_fast : when True, the first failing task stops new submissions and
        best-effort cancels pending futures; pending work that is already
        running continues to completion (ProcessPoolExecutor cannot kill
        running workers). All failures collected so far are surfaced via
        :class:`ParallelTaskErrors`.

    Returns
    -------
    list of result dicts, one per task, in input order. Entries for
    tasks that were never submitted (fail_fast abort) are ``None``.

    Raises
    ------
    ValueError
        If any task's ``vram_needed_mb`` exceeds the largest GPU's capacity
        (would otherwise deadlock the scheduler).
    ParallelTaskErrors
        If any task failed. In ``fail_fast`` mode, raised as soon as the
        pending queue drains; otherwise raised after all tasks complete.
    """
    if not tasks:
        return []

    res = get_system_resources()
    cpu_admission_capacity = _cpu_admission_capacity(res)
    use_gpu = bool(res.gpus) and device_strategy in ("auto", "gpu")

    # B3: fail-fast if any task is larger than the biggest GPU's hardware
    # capacity. We check against ``total - safety_margin``, not the initial
    # free snapshot — on a shared server a co-tenant may release VRAM later.
    theoretical_cap: dict[int, float] = (
        _theoretical_gpu_capacity(res.gpus) if use_gpu else {}
    )
    if use_gpu:
        max_cap = max(theoretical_cap.values(), default=0.0)
        max_need = max(t.vram_needed_mb for t in tasks)
        if max_need > max_cap:
            raise ValueError(
                f"task vram_needed_mb={max_need:.0f} exceeds largest GPU capacity "
                f"({max_cap:.0f} MB after {GPU_SAFETY_MARGIN_MB:.0f} MB safety margin). "
                f"No GPU can host this task — scheduling would deadlock."
            )

    # Pool size / concurrency cap.
    # ``ProcessPoolExecutor`` is lazy: setting ``max_workers=len(tasks)`` does
    # NOT eagerly spawn 200 processes; it only caps the high-water mark.
    # Per-submission CPU/GPU gating handles actual resource pressure.
    submit_limit = min(max_workers, len(tasks)) if max_workers is not None else len(tasks)
    submit_limit = max(1, submit_limit)
    if not use_gpu and max_workers is None:
        submit_limit = min(submit_limit, res.cpu_count)
    pool_size = submit_limit

    _log_resource_summary(res, max_workers, len(tasks), min(t.vram_needed_mb for t in tasks))

    ctx = multiprocessing.get_context("spawn")

    # B1: on multi-GPU systems, a reused worker's first CUDA init pins it
    # to whatever CUDA_VISIBLE_DEVICES was set on its first task, making
    # subsequent ``os.environ`` reassignments no-ops. Forcing 1 task per
    # child process guarantees each task sees a fresh CUDA context. We pay
    # the ~1–3 s torch-import cost per task in exchange for correctness.
    # Single-GPU and CPU-only pools reuse workers as before — PyTorch's
    # caching allocator reclaims between tasks; if a long sweep starts
    # fragmenting VRAM on single-GPU, revisit this heuristic.
    max_tasks_per_child = 1 if use_gpu and len(res.gpus) > 1 else None

    # Slots are ``None`` only transiently during a fail_fast abort or
    # ``KeyboardInterrupt``; both paths raise before ``return results``,
    # so the returned list is always fully populated.
    results: list[TaskResult | None] = [None] * len(tasks)
    pending: dict[Future, int] = {}
    # Maps each in-flight Future to ``(gpu_index, reserve_mb, submitted_at)``.
    # ``submitted_at`` drives the allocation grace window: within the first
    # ``GPU_ALLOCATION_GRACE_S`` seconds after ``executor.submit``, the task's
    # reserve is subtracted from the driver's live probe because the worker
    # hasn't yet allocated VRAM. After the grace window we trust the probe
    # to reflect the allocation. ``gpu_index`` is ``None`` for CPU-only tasks.
    future_gpu: dict[Future, tuple[int | None, float, float]] = {}
    pending_cpu_total = 0  # sum of cpus_needed for in-flight tasks

    # B9: exponential backoff on the "waiting for resources" log line.
    wait_backoff_s = 5.0
    last_wait_log = 0.0
    aborted = False
    keyboard_interrupted = False
    pool_broken = False

    bar = tqdm(total=len(tasks), unit="task", dynamic_ncols=True, desc="  [parallel]")
    executor = ProcessPoolExecutor(
        max_workers=pool_size,
        mp_context=ctx,
        max_tasks_per_child=max_tasks_per_child,
    )
    try:
        try:
            unscheduled = list(range(len(tasks)))

            while unscheduled or pending:
                # Re-probe live GPU memory on each scheduling tick.
                # On a shared server a co-tenant's allocations change mid-run.
                # We re-probe once per outer iteration, subtract reserves of
                # recently-submitted tasks that haven't yet allocated (grace
                # window), and cache ``effective_free`` for use in the inner
                # admission loop.
                now = time.time()
                if use_gpu:
                    probe_free = _probe_gpu_free_memory(list(theoretical_cap))
                    unallocated: dict[int, float] = {}
                    for fut_info in future_gpu.values():
                        g_idx, reserve_mb, t_submit = fut_info
                        if g_idx is not None and (now - t_submit) < GPU_ALLOCATION_GRACE_S:
                            unallocated[g_idx] = unallocated.get(g_idx, 0.0) + reserve_mb
                    effective_free = {
                        i: max(0.0, probe_free.get(i, 0.0) - unallocated.get(i, 0.0))
                        for i in theoretical_cap
                        if i in probe_free  # probe-failed GPUs stay unavailable this tick
                    }
                else:
                    effective_free = {}

                # --- submit as many tasks as the limits allow ----------------
                cpu_blocked = False
                gpu_blocked = False
                while unscheduled and len(pending) < submit_limit:
                    selected: tuple[int, int, int | None, float, str | None] | None = None
                    for queue_pos, task_idx in enumerate(unscheduled):
                        spec = tasks[task_idx]

                        # An idle scheduler always admits one task even when
                        # its declared CPU need exceeds the detected capacity.
                        # Otherwise skip blocked tasks and keep scanning for a
                        # smaller fitting task instead of stalling the queue.
                        if pending and not _cpu_can_accept(
                            spec.cpus_needed,
                            pending_cpu_total,
                            cpu_count=cpu_admission_capacity,
                        ):
                            cpu_blocked = True
                            continue

                        # Pure-CPU tasks may pass a GPU-blocked task. This is
                        # important for mixed training/evaluation manifests.
                        if use_gpu and spec.vram_needed_mb > 0:
                            gpu_idx = _find_free_gpu(
                                effective_free, spec.vram_needed_mb
                            )
                            if gpu_idx is None:
                                gpu_blocked = True
                                continue
                            reserve = (
                                spec.vram_reserve_mb
                                if spec.vram_reserve_mb is not None
                                else spec.vram_needed_mb
                            )
                            cuda_dev: str | None = str(gpu_idx)
                        else:
                            gpu_idx = None
                            reserve = 0.0
                            cuda_dev = None
                        selected = (queue_pos, task_idx, gpu_idx, reserve, cuda_dev)
                        break

                    if selected is None:
                        break
                    queue_pos, task_idx, gpu_idx, reserve, cuda_dev = selected
                    spec = tasks[task_idx]
                    if gpu_idx is not None:
                        effective_free[gpu_idx] = max(
                            0.0, effective_free[gpu_idx] - reserve
                        )

                    try:
                        fut = executor.submit(
                            _worker_wrapper,
                            spec.func, spec.args, spec.kwargs,
                            cuda_dev, spec.label or f"task-{task_idx}",
                            spec.cpu_threads,
                        )
                    except BrokenProcessPool as exc:
                        # A worker was killed (OOM-killer / segfault), so the
                        # pool is dead and no further submits can succeed. Record
                        # every not-yet-submitted task as a failure — so they
                        # surface in ParallelTaskErrors instead of silently
                        # vanishing — then stop submitting and drain whatever is
                        # still in flight (their futures resolve with the same
                        # error, caught by fut.result() below).
                        pool_broken = True
                        for j in unscheduled:
                            if results[j] is None:
                                results[j] = {
                                    "status": "error",
                                    "label": tasks[j].label,
                                    "error": f"not submitted: process pool broken ({exc})",
                                    "error_type": type(exc).__name__,
                                    "error_traceback": "",
                                    "elapsed_s": 0.0,
                                }
                        _logger.error(
                            "process pool broken at %s; %d task(s) not submitted, "
                            "%d still in flight.",
                            spec.label, len(unscheduled), len(pending),
                        )
                        unscheduled.clear()
                        break
                    unscheduled.pop(queue_pos)
                    pending[fut] = task_idx
                    pending_cpu_total += spec.cpus_needed
                    # Freshly call ``time.time()`` (not the tick-start ``now``)
                    # so the grace window starts from the actual submission
                    # instant — tick setup work isn't counted against it.
                    future_gpu[fut] = (gpu_idx, reserve, time.time())
                    gpu_tag = f" -> GPU {gpu_idx}" if gpu_idx is not None else ""
                    tqdm.write(f"  [parallel] START {spec.label}{gpu_tag}")
                    _logger.info("START %s%s", spec.label, gpu_tag)
                    bar.set_postfix(running=len(pending), queued=len(unscheduled))
                    # B9: reset backoff whenever we successfully submit a task.
                    wait_backoff_s = 5.0
                    last_wait_log = 0.0

                if not pending:
                    if not unscheduled:
                        break  # all tasks submitted and collected
                    now = time.time()
                    if now - last_wait_log >= wait_backoff_s:
                        tqdm.write(
                            f"  [parallel] waiting for resources "
                            f"({len(unscheduled)} tasks queued)..."
                        )
                        last_wait_log = now
                        wait_backoff_s = min(wait_backoff_s * 2.0, 300.0)
                    time.sleep(5)
                    continue

                # Three distinct wait strategies:
                if gpu_blocked or cpu_blocked:
                    timeout: float | None = 5.0  # re-probe external resources
                else:
                    timeout = None  # at submit_limit cap — wait for a pool slot
                done, _ = wait(pending, timeout=timeout, return_when=FIRST_COMPLETED)
                if not done:
                    continue  # timeout — re-probe resources
                for fut in done:
                    idx = pending.pop(fut)
                    pending_cpu_total -= tasks[idx].cpus_needed
                    # VRAM release is observed by the next tick's driver probe
                    # once the worker exits — no explicit bookkeeping needed.
                    future_gpu.pop(fut, None)

                    # A fail_fast abort cancels queued futures; they surface here
                    # as "done" but never ran. Leave results[idx] = None (same as
                    # never-submitted) instead of logging a spurious CancelledError
                    # failure that would inflate ParallelTaskErrors.
                    if fut.cancelled():
                        bar.update(1)
                        continue

                    try:
                        result = fut.result()
                    except Exception as exc:
                        result = {
                            "status": "error",
                            "label": tasks[idx].label,
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                            "error_traceback": "",
                            "elapsed_s": 0.0,
                        }
                    results[idx] = result
                    bar.update(1)
                    bar.set_postfix(running=len(pending), queued=len(unscheduled))
                    _log_task_done(result)

                    # B4: first failure under fail_fast aborts further submissions.
                    if fail_fast and result["status"] == "error" and not aborted:
                        aborted = True
                        unscheduled.clear()  # stop submitting new work
                        for queued_fut in list(pending):
                            queued_fut.cancel()
                        _logger.warning(
                            "fail_fast: aborting after %s failed; %d pending, "
                            "%d never submitted.",
                            result["label"], len(pending), sum(1 for r in results if r is None),
                        )
        except KeyboardInterrupt:
            # B6: user interrupt — mark unfinished tasks and re-raise.
            keyboard_interrupted = True
            _logger.warning(
                "KeyboardInterrupt: cancelling %d pending task(s).",
                len(pending),
            )
            for fut in list(pending):
                idx = pending.get(fut)
                fut.cancel()
                if idx is not None and results[idx] is None:
                    results[idx] = {
                        "status": "cancelled",
                        "label": tasks[idx].label,
                        "error": "KeyboardInterrupt",
                        "error_type": "KeyboardInterrupt",
                        "error_traceback": "",
                        "elapsed_s": 0.0,
                    }
            raise
    finally:
        bar.close()
        # ``cancel_futures=True`` only stops queued work; running tasks
        # continue. On fail_fast abort or KeyboardInterrupt, don't block
        # waiting for them — leave them to be reaped with the parent.
        executor.shutdown(
            wait=not (aborted or keyboard_interrupted or pool_broken),
            cancel_futures=True,
        )

    failures: list[TaskFailure] = [
        r for r in results if r is not None and r["status"] == "error"  # type: ignore[misc]
    ]
    if failures:
        raise ParallelTaskErrors(failures, [r for r in results if r is not None])

    # By here every slot is filled (exceptional paths raised earlier).
    return [r for r in results if r is not None]


def _log_resource_summary(
    res: SystemResources,
    max_workers: int | None,
    n_tasks: int,
    min_vram_needed_mb: float = GPU_FREE_THRESHOLD_MB,
) -> None:
    gpu_info = ""
    if res.gpus:
        gpu_strs = [
            f"GPU{g.index}({g.free_memory_mb:.0f}/{g.total_memory_mb:.0f}MB)"
            for g in res.gpus
        ]
        gpu_info = f", GPUs: {', '.join(gpu_strs)}"
    cap = f"max_workers={max_workers}" if max_workers is not None else "max_workers=auto"
    summary_line = (
        f"  [parallel] {n_tasks} tasks, {cap}, scheduling=dynamic, "
        f"CPUs={res.cpu_count} (startup_available={res.available_cpus:.1f}, "
        f"load1m={res.load_avg_1m:.1f}){gpu_info}"
    )
    print(summary_line)
    _logger.info(summary_line.strip())
    if res.gpus:
        gpu_line = (
            f"  [parallel] GPU: multi-task-per-GPU (VRAM-gated), "
            f"vram_needed >= {min_vram_needed_mb:.0f} MB"
        )
        print(gpu_line)
        _logger.info(gpu_line.strip())


def _log_task_done(result: TaskResult) -> None:
    elapsed = result.get("elapsed_s", 0)
    minutes, seconds = divmod(elapsed, 60)
    label = result.get("label", "?")
    if result["status"] == "ok":
        msg = f"  [parallel] DONE {label} ({int(minutes)}m {seconds:.1f}s)"
        tqdm.write(msg)
        _logger.info("DONE %s elapsed=%.1fs", label, elapsed)
    else:
        msg = (
            f"  [parallel] FAIL {label} ({int(minutes)}m {seconds:.1f}s): "
            f"{result.get('error_type', '?')}: {result.get('error', '?')}"
        )
        tqdm.write(msg)
        _logger.error(
            "FAIL %s elapsed=%.1fs %s: %s",
            label,
            elapsed,
            result.get("error_type", "?"),
            result.get("error", "?"),
        )
