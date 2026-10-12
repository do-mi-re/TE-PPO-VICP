"""Regression tests for src.parallel scheduling logic.

These tests exercise the scheduler control-flow — early-exit, concurrency
caps, failure propagation, and resource-gated waiting — using fast
in-process helpers (no real training).  Every test forces CPU-only mode
via ``device_strategy="cpu"`` unless it specifically exercises GPU paths.
"""

from __future__ import annotations

import time
import unittest
from unittest.mock import patch

from src.parallel import (
    GPUInfo,
    ParallelTaskErrors,
    SystemResources,
    TaskSpec,
    _cpu_admission_capacity,
    _find_free_gpu,
    _theoretical_gpu_capacity,
    run_parallel_tasks,
)

# ---------------------------------------------------------------------------
# Picklable helpers — must be top-level for spawn-based ProcessPoolExecutor.
# ---------------------------------------------------------------------------

def _return_value(x: object) -> object:
    """Trivially return its argument."""
    return x


def _raise_error(msg: str) -> None:
    raise RuntimeError(msg)


def _hard_exit(_x: object) -> None:
    """Kill the worker process abruptly (simulates an OOM-killer/segfault).

    A bare ``os._exit`` skips all cleanup, so the parent's
    ``ProcessPoolExecutor`` sees the process vanish and enters the broken state.
    """
    import os
    os._exit(1)


def _record_timing(duration: float) -> dict:
    """Sleep for *duration* seconds and return wall-clock start/end."""
    start = time.time()
    time.sleep(duration)
    end = time.time()
    return {"start": start, "end": end}


def _thread_limits() -> dict[str, object]:
    import os

    import torch

    return {
        "omp": os.environ.get("OMP_NUM_THREADS"),
        "mkl": os.environ.get("MKL_NUM_THREADS"),
        "torch": torch.get_num_threads(),
    }


def _peak_concurrency(timings: list[dict]) -> int:
    """Reconstruct peak concurrency from start/end timestamps."""
    events: list[tuple[float, int]] = []
    for t in timings:
        events.append((t["start"], +1))
        events.append((t["end"], -1))
    events.sort()
    peak = 0
    current = 0
    for _, delta in events:
        current += delta
        peak = max(peak, current)
    return peak


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunParallelTasks(unittest.TestCase):
    """Core scheduler behaviour."""

    def test_empty_task_list(self) -> None:
        self.assertEqual(run_parallel_tasks([]), [])

    def test_explicit_cpu_capacity_caps_detected_idle_slots(self) -> None:
        resources = SystemResources(
            cpu_count=96,
            load_avg_1m=0.0,
            available_cpus=92.0,
        )
        with patch.dict("os.environ", {"TEPPO_CPU_CAPACITY": "48"}):
            self.assertEqual(_cpu_admission_capacity(resources), 48)

    def test_all_succeed_preserves_order(self) -> None:
        tasks = [
            TaskSpec(func=_return_value, args=(i,), label=f"t{i}")
            for i in range(5)
        ]
        results = run_parallel_tasks(tasks, device_strategy="cpu")
        self.assertEqual(len(results), 5)
        for i, r in enumerate(results):
            self.assertEqual(r["status"], "ok")
            self.assertEqual(r["result"], i)
            self.assertEqual(r["label"], f"t{i}")

    def test_worker_applies_explicit_thread_budget(self) -> None:
        result = run_parallel_tasks(
            [
                TaskSpec(
                    func=_thread_limits,
                    label="threads",
                    cpus_needed=1,
                    cpu_threads=2,
                )
            ],
            max_workers=1,
            device_strategy="cpu",
        )[0]
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["result"], {"omp": "2", "mkl": "2", "torch": 2})

    def test_single_failure_raises(self) -> None:
        tasks = [
            TaskSpec(func=_return_value, args=(1,), label="good"),
            TaskSpec(func=_raise_error, args=("boom",), label="bad"),
        ]
        with self.assertRaises(ParallelTaskErrors) as ctx:
            run_parallel_tasks(tasks, device_strategy="cpu")
        exc = ctx.exception
        self.assertEqual(len(exc.failures), 1)
        self.assertEqual(exc.failures[0]["label"], "bad")
        self.assertIn("boom", exc.failures[0]["error"])
        self.assertEqual(
            [result["label"] for result in exc.results if result["status"] == "ok"],
            ["good"],
        )

    def test_multiple_failures_reported(self) -> None:
        tasks = [
            TaskSpec(func=_raise_error, args=("err1",), label="bad1"),
            TaskSpec(func=_raise_error, args=("err2",), label="bad2"),
            TaskSpec(func=_return_value, args=(0,), label="good"),
        ]
        with self.assertRaises(ParallelTaskErrors) as ctx:
            run_parallel_tasks(tasks, device_strategy="cpu")
        labels = {f["label"] for f in ctx.exception.failures}
        self.assertEqual(labels, {"bad1", "bad2"})


class TestFailFast(unittest.TestCase):
    """``fail_fast=True`` stops submitting after the first failure."""

    def test_fail_fast_skips_unsubmitted_tasks(self) -> None:
        """When fail_fast=True, tasks submitted after a failure are skipped.

        We use ``max_workers=1`` + a long-sleeping good task to force serial
        execution: the failing task runs first, then the scheduler aborts
        before the second good task gets submitted.
        """
        tasks = [
            TaskSpec(func=_raise_error, args=("boom",), label="bad"),
            TaskSpec(func=_record_timing, args=(0.1,), label="never_runs"),
            TaskSpec(func=_record_timing, args=(0.1,), label="also_never"),
        ]
        with self.assertRaises(ParallelTaskErrors) as ctx:
            run_parallel_tasks(
                tasks,
                max_workers=1,
                device_strategy="cpu",
                fail_fast=True,
            )
        failures = ctx.exception.failures
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["label"], "bad")

    def test_fail_fast_false_runs_all_tasks_even_after_failure(self) -> None:
        """Default behaviour (fail_fast=False) runs all tasks before raising."""
        tasks = [
            TaskSpec(func=_raise_error, args=("boom",), label="bad"),
            TaskSpec(func=_return_value, args=(42,), label="good"),
        ]
        with self.assertRaises(ParallelTaskErrors) as ctx:
            run_parallel_tasks(tasks, device_strategy="cpu", fail_fast=False)
        # The good task still ran and produced a result — just overshadowed
        # by the failing task's exception. Its success remains available to
        # callers that must persist per-task provenance.
        self.assertEqual(len(ctx.exception.failures), 1)
        self.assertEqual(
            [r["result"] for r in ctx.exception.results if r["status"] == "ok"],
            [42],
        )


class TestMaxWorkersCap(unittest.TestCase):
    """``max_workers`` must limit peak concurrency."""

    def test_max_workers_limits_concurrency(self) -> None:
        # 6 tasks, each sleeps 0.5s.  max_workers=2 means at most 2 overlap.
        tasks = [
            TaskSpec(
                func=_record_timing,
                args=(0.5,),
                label=f"t{i}",
                cpus_needed=1,
            )
            for i in range(6)
        ]
        results = run_parallel_tasks(
            tasks, max_workers=2, device_strategy="cpu",
        )
        peak = _peak_concurrency([r["result"] for r in results])
        self.assertLessEqual(peak, 2, f"peak concurrency {peak} exceeded max_workers=2")

    def test_cpu_gating_serializes_when_always_rejected(self) -> None:
        """When _cpu_can_accept always returns False, tasks run serially.

        The bypass ensures at least one task runs when pending is empty.
        Since the check always rejects, no second task is submitted while
        one is running — they execute one at a time.
        """
        tasks = [
            TaskSpec(func=_record_timing, args=(0.3,), label=f"t{i}", cpus_needed=99)
            for i in range(3)
        ]
        # _cpu_can_accept(99) always False on a 10-core machine (no mock needed).
        results = run_parallel_tasks(tasks, device_strategy="cpu")

        self.assertEqual(len(results), 3)
        timings = [r["result"] for r in results]

        # Verify serial: each task starts after the previous ends.
        peak = _peak_concurrency(timings)
        self.assertEqual(peak, 1, "Tasks should run serially when CPU is overloaded")

    def test_effective_four_slot_jobs_run_two_at_a_time_on_ten_cpus(self) -> None:
        resources = SystemResources(
            cpu_count=10,
            load_avg_1m=0.0,
            available_cpus=10.0,
        )
        tasks = [
            TaskSpec(
                func=_record_timing,
                args=(0.3,),
                label=f"effective-core-{index}",
                cpus_needed=4,
            )
            for index in range(3)
        ]

        with (
            patch("src.parallel.get_system_resources", return_value=resources),
            patch.dict("os.environ", {"TEPPO_CPU_CAPACITY": "10"}),
        ):
            results = run_parallel_tasks(
                tasks, max_workers=3, device_strategy="cpu"
            )

        self.assertEqual(_peak_concurrency([r["result"] for r in results]), 2)

    def test_stable_first_fit_bypasses_blocked_large_task(self) -> None:
        """A large queued task must not idle a CPU slot a smaller task can use."""
        resources = SystemResources(
            cpu_count=4,
            load_avg_1m=0.0,
            available_cpus=4.0,
        )
        tasks = [
            TaskSpec(
                func=_record_timing, args=(0.4,), label="occupier", cpus_needed=3
            ),
            TaskSpec(
                func=_record_timing, args=(0.1,), label="blocked-large", cpus_needed=3
            ),
            TaskSpec(
                func=_record_timing, args=(0.1,), label="fitting-small", cpus_needed=1
            ),
        ]
        with patch("src.parallel.get_system_resources", return_value=resources):
            results = run_parallel_tasks(
                tasks, max_workers=2, device_strategy="cpu"
            )

        occupier, blocked, fitting = [row["result"] for row in results]
        self.assertLess(fitting["start"], occupier["end"])
        self.assertGreaterEqual(blocked["start"], occupier["end"])
        self.assertEqual([row["label"] for row in results], [
            "occupier", "blocked-large", "fitting-small",
        ])


class TestResourceWaitRetry(unittest.TestCase):
    """Scheduler must retry when no resources are available, not exit."""

    def test_gpu_mode_waits_when_no_gpu_free(self) -> None:
        """Simulate GPU mode where no GPU is free initially.

        ``_probe_gpu_free_memory`` returns an empty map the first 2 ticks,
        then a usable GPU. The scheduler must wait and retry, eventually
        running all tasks rather than exiting early.
        """
        probe_calls = 0

        def fake_probe(indices: list[int]) -> dict[int, float]:
            nonlocal probe_calls
            probe_calls += 1
            if probe_calls <= 2:
                return {}  # no GPU visibly free
            return {0: 10_000.0}

        fake_res = SystemResources(
            cpu_count=4,
            load_avg_1m=0.0,
            available_cpus=4.0,
            gpus=(GPUInfo(index=0, total_memory_mb=24_000, free_memory_mb=20_000),),
        )

        tasks = [
            TaskSpec(func=_return_value, args=(i,), label=f"t{i}")
            for i in range(3)
        ]

        with (
            patch("src.parallel.get_system_resources", return_value=fake_res),
            patch("src.parallel._probe_gpu_free_memory", side_effect=fake_probe),
            patch("src.parallel.time.sleep"),  # don't actually sleep
        ):
            results = run_parallel_tasks(tasks, device_strategy="auto")

        # All 3 tasks must have completed — the scheduler must NOT have
        # exited early when no GPU was visible.
        self.assertEqual(len(results), 3)
        for i, r in enumerate(results):
            self.assertEqual(r["status"], "ok")
            self.assertEqual(r["result"], i)

    def test_cpus_needed_exceeds_cpu_count_no_deadlock(self) -> None:
        """cpus_needed > cpu_count must not deadlock — first task always runs."""
        tasks = [
            TaskSpec(func=_return_value, args=(i,), label=f"t{i}", cpus_needed=99)
            for i in range(3)
        ]
        # _cpu_can_accept(99) always returns False on a 10-core machine,
        # but `pending` is empty so the check is bypassed for task 0.
        # After task 0 completes, task 1 gets the same bypass, etc.
        results = run_parallel_tasks(tasks, device_strategy="cpu")
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertEqual(r["status"], "ok")

    def test_no_tasks_dropped_under_resource_pressure(self) -> None:
        """Even when CPU is tight, all tasks eventually run."""
        tasks = [
            TaskSpec(
                func=_return_value,
                args=(i,),
                label=f"t{i}",
                cpus_needed=99,
            )
            for i in range(4)
        ]

        call_count = 0

        def allow_one_at_a_time(cpus_needed: int, pending_cpu_total: int = 0, *, cpu_count: int | None = None) -> bool:
            nonlocal call_count
            call_count += 1
            return True  # let them through — real gating tested elsewhere

        with patch("src.parallel._cpu_can_accept", side_effect=allow_one_at_a_time):
            results = run_parallel_tasks(tasks, device_strategy="cpu")

        self.assertEqual(len(results), 4)
        for r in results:
            self.assertEqual(r["status"], "ok")


class TestLoadAwareConcurrency(unittest.TestCase):
    """CPU concurrency must respect existing system load dynamically."""

    def test_high_load_reduces_concurrency(self) -> None:
        """When _cpu_can_accept always rejects, tasks run serially (peak=1)."""
        tasks = [
            TaskSpec(
                func=_record_timing,
                args=(0.3,),
                label=f"t{i}",
                cpus_needed=2,
            )
            for i in range(4)
        ]

        # Always reject — tasks run one at a time via the bypass-for-first rule.
        with patch("src.parallel._cpu_can_accept", return_value=False):
            results = run_parallel_tasks(tasks, device_strategy="cpu")

        self.assertEqual(len(results), 4)
        for r in results:
            self.assertEqual(r["status"], "ok")
        peak = _peak_concurrency([r["result"] for r in results])
        self.assertEqual(peak, 1, f"peak concurrency {peak} should be 1 under load")


class TestMaxWorkersValidation(unittest.TestCase):
    """Zero or negative max_workers must not crash the pool."""

    def test_zero_max_workers_still_runs(self) -> None:
        """run_parallel_tasks clamps submit_limit to >= 1."""
        tasks = [
            TaskSpec(func=_return_value, args=(42,), label="t0"),
        ]
        results = run_parallel_tasks(
            tasks, max_workers=0, device_strategy="cpu",
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "ok")
        self.assertEqual(results[0]["result"], 42)

    def test_negative_max_workers_still_runs(self) -> None:
        tasks = [
            TaskSpec(func=_return_value, args=(7,), label="t0"),
        ]
        results = run_parallel_tasks(
            tasks, max_workers=-3, device_strategy="cpu",
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "ok")


class TestFindFreeGpu(unittest.TestCase):
    """Pure-function tests for ``_find_free_gpu`` over an effective-free map.

    The scheduler computes ``effective_free_mb`` each tick as
    ``driver_probe - in_flight_reserves_still_in_grace``. This helper just
    picks the best GPU from that map — no driver probe of its own.
    """

    def test_picks_gpu_with_most_free(self) -> None:
        """With equal need, picks the GPU with the most effective free."""
        self.assertEqual(
            _find_free_gpu({0: 10_000, 1: 20_000}),
            1,
        )

    def test_skips_gpus_below_needed(self) -> None:
        """GPU below ``vram_needed_mb`` is ineligible even if it's the best."""
        # GPU0 has 0 MB free (fully used); GPU1 has 10 GB → GPU1 wins.
        self.assertEqual(
            _find_free_gpu({0: 0, 1: 10_000}),
            1,
        )

    def test_returns_none_when_all_below_threshold(self) -> None:
        self.assertIsNone(
            _find_free_gpu({0: 0, 1: 0})
        )

    def test_returns_none_when_below_custom_threshold(self) -> None:
        """GPU with 3 GB free < default 5 GB threshold → no fit."""
        self.assertIsNone(
            _find_free_gpu({0: 3_000})
        )

    def test_eligible_when_vram_available(self) -> None:
        self.assertEqual(
            _find_free_gpu({0: 15_000}),
            0,
        )

    def test_respects_vram_needed(self) -> None:
        """GPU with 6 GB free rejects a 7 GB task but accepts a 5 GB one."""
        self.assertIsNone(
            _find_free_gpu({0: 6_000}, vram_needed_mb=7_000)
        )
        self.assertEqual(
            _find_free_gpu({0: 6_000}, vram_needed_mb=5_000),
            0,
        )

    def test_empty_map_returns_none(self) -> None:
        """If no GPU appears in the map (all probes failed), return None."""
        self.assertIsNone(_find_free_gpu({}))


class TestTheoreticalGpuCapacity(unittest.TestCase):
    """Theoretical capacity is total memory minus the safety margin.
    Admission uses a separate live-memory probe."""

    def test_capacity_subtracts_safety_margin_from_total(self) -> None:
        gpus = (
            GPUInfo(index=0, total_memory_mb=24_000, free_memory_mb=24_000),
        )
        capacity = _theoretical_gpu_capacity(gpus)
        self.assertAlmostEqual(capacity[0], 23_488.0, places=1)

    def test_capacity_ignores_initial_free(self) -> None:
        """Even if the GPU is heavily used at startup, capacity is hardware max.

        The actual admission check will later use the live probe.
        """
        gpus = (
            GPUInfo(index=0, total_memory_mb=24_000, free_memory_mb=1_000),
        )
        capacity = _theoretical_gpu_capacity(gpus)
        self.assertAlmostEqual(capacity[0], 23_488.0, places=1)

    def test_capacity_never_negative(self) -> None:
        """If total is tiny, capacity clamps to 0 rather than going negative."""
        gpus = (
            GPUInfo(index=0, total_memory_mb=100, free_memory_mb=100),
        )
        capacity = _theoretical_gpu_capacity(gpus)
        self.assertEqual(capacity[0], 0.0)


class TestVramOvercommitFailFast(unittest.TestCase):
    """B3: a task whose VRAM need exceeds every GPU's capacity must raise early."""

    def test_task_exceeding_gpu_capacity_raises_value_error(self) -> None:
        fake_res = SystemResources(
            cpu_count=32,
            load_avg_1m=0.0,
            available_cpus=32.0,
            gpus=(GPUInfo(index=0, total_memory_mb=8_000, free_memory_mb=8_000),),
        )
        tasks = [
            TaskSpec(
                func=_return_value,
                args=(0,),
                label="huge",
                vram_needed_mb=100_000,  # 100 GB — no single GPU can host.
            ),
        ]
        with patch("src.parallel.get_system_resources", return_value=fake_res):
            with self.assertRaises(ValueError) as ctx:
                run_parallel_tasks(tasks, device_strategy="auto")
        self.assertIn("exceeds largest GPU capacity", str(ctx.exception))


class TestGpuMultiTenancyScheduling(unittest.TestCase):
    """Integration test: scheduler distributes tasks across GPUs by memory."""

    def test_tasks_spread_across_gpus(self) -> None:
        """With 2 GPUs and 4 tasks, all tasks complete successfully.

        The real ``_find_free_gpu`` picks the GPU with the most remaining
        effective free; within a tick, admitting a task on GPU 0 decrements
        its effective free, so the next admission naturally picks GPU 1.
        """
        fake_res = SystemResources(
            cpu_count=32,
            load_avg_1m=0.0,
            available_cpus=32.0,
            gpus=(
                GPUInfo(index=0, total_memory_mb=24000, free_memory_mb=20000),
                GPUInfo(index=1, total_memory_mb=24000, free_memory_mb=20000),
            ),
        )

        tasks = [
            TaskSpec(func=_return_value, args=(i,), label=f"t{i}")
            for i in range(4)
        ]

        with (
            patch("src.parallel.get_system_resources", return_value=fake_res),
            patch("src.parallel._probe_gpu_free_memory",
                  return_value={0: 20_000.0, 1: 20_000.0}),
        ):
            results = run_parallel_tasks(tasks, device_strategy="auto")

        self.assertEqual(len(results), 4)
        for r in results:
            self.assertEqual(r["status"], "ok")

    def test_waits_when_all_gpus_full(self) -> None:
        """When a single GPU can only fit one task at a time, the scheduler serialises.

        ``vram_needed_mb`` defaults to 5000; the probed free is 8000, so after
        admitting one task the remaining headroom (8000 - 5000 reserve = 3000)
        is below the next task's 5000 threshold until the first task exits the
        grace window AND completes.
        """
        fake_res = SystemResources(
            cpu_count=32,
            load_avg_1m=0.0,
            available_cpus=32.0,
            gpus=(GPUInfo(index=0, total_memory_mb=24000, free_memory_mb=8000),),
        )

        tasks = [
            TaskSpec(func=_return_value, args=(i,), label=f"t{i}")
            for i in range(3)
        ]

        with (
            patch("src.parallel.get_system_resources", return_value=fake_res),
            patch("src.parallel._probe_gpu_free_memory", return_value={0: 8_000.0}),
        ):
            results = run_parallel_tasks(tasks, device_strategy="auto")

        # All 3 tasks must complete even though only 1 GPU slot at a time
        self.assertEqual(len(results), 3)
        for i, r in enumerate(results):
            self.assertEqual(r["status"], "ok")
            self.assertEqual(r["result"], i)


class TestGpuMultiTenancy(unittest.TestCase):
    """Multi-tenancy: multiple tasks may share a GPU when VRAM allows."""

    def test_tasks_stack_on_single_gpu_when_vram_allows(self) -> None:
        """Two tasks share one GPU when their combined reservation fits in VRAM.

        ``_probe_gpu_free_memory`` is patched to 20 GB. Tasks need 8 GB each
        but only reserve 2 GB footprint. Task 0 admitted (20 - 2 = 18 GB
        remaining), task 1 sees 18 GB >= 8 GB → admitted too, stacked.
        """
        fake_res = SystemResources(
            cpu_count=32,
            load_avg_1m=0.0,
            available_cpus=32.0,
            gpus=(GPUInfo(index=0, total_memory_mb=24_000, free_memory_mb=20_000),),
        )

        task_duration = 0.5
        tasks = [
            TaskSpec(
                func=_record_timing,
                args=(task_duration,),
                label=f"t{i}",
                vram_needed_mb=8_000,   # admission gate: need 8 GB remaining
                vram_reserve_mb=2_000,  # actual footprint: only 2 GB reserved
            )
            for i in range(2)
        ]

        with (
            patch("src.parallel.get_system_resources", return_value=fake_res),
            patch("src.parallel._probe_gpu_free_memory", return_value={0: 20_000.0}),
        ):
            results = run_parallel_tasks(tasks, device_strategy="auto")

        self.assertEqual(len(results), 2)
        for r in results:
            self.assertEqual(r["status"], "ok")

        t0 = results[0]["result"]
        t1 = results[1]["result"]

        # Both tasks should overlap — combined reservation (4 GB) << 20 GB cap.
        overlap_start = max(t0["start"], t1["start"])
        overlap_end = min(t0["end"], t1["end"])
        self.assertGreater(
            overlap_end - overlap_start, 0.0,
            "Tasks with small vram_reserve_mb should stack on one GPU and overlap",
        )

    def test_tasks_serialize_when_vram_exhausted(self) -> None:
        """Tasks that do NOT fit together in VRAM run serially on the same GPU.

        Probed free is 8 GB. Each task needs 5 GB and reserves 5 GB. After
        admitting task 0, effective remaining = 8 - 5 = 3 GB < 5 GB, so
        task 1 must wait until task 0 completes (then the probe reflects
        the freed memory and task 1 gets admitted).
        """
        fake_res = SystemResources(
            cpu_count=32,
            load_avg_1m=0.0,
            available_cpus=32.0,
            gpus=(GPUInfo(index=0, total_memory_mb=24_000, free_memory_mb=8_000),),
        )

        task_duration = 0.5
        tasks = [
            TaskSpec(func=_record_timing, args=(task_duration,), label=f"t{i}")
            for i in range(2)
        ]

        # Probe always reports 8 GB free — but the scheduler's grace-window
        # accounting correctly subtracts the in-flight task's reserve.
        with (
            patch("src.parallel.get_system_resources", return_value=fake_res),
            patch("src.parallel._probe_gpu_free_memory", return_value={0: 8_000.0}),
        ):
            results = run_parallel_tasks(tasks, device_strategy="auto")

        self.assertEqual(len(results), 2)
        for r in results:
            self.assertEqual(r["status"], "ok")

        t0 = results[0]["result"]
        t1 = results[1]["result"]

        # Task 1 must not start until task 0 finishes (VRAM exhausted).
        self.assertGreaterEqual(
            t1["start"], t0["end"] - 0.05,
            "Task 1 should start only after task 0 ends when VRAM is exhausted",
        )

    def test_tasks_parallelize_across_two_gpus(self) -> None:
        """Two tasks on different GPUs must overlap in time.

        Real ``_find_free_gpu`` picks max remaining free; within-tick
        decrement ensures the second admission goes to the other GPU.
        """
        fake_res = SystemResources(
            cpu_count=32,
            load_avg_1m=0.0,
            available_cpus=32.0,
            gpus=(
                GPUInfo(index=0, total_memory_mb=24_000, free_memory_mb=20_000),
                GPUInfo(index=1, total_memory_mb=24_000, free_memory_mb=20_000),
            ),
        )

        task_duration = 0.5
        tasks = [
            TaskSpec(func=_record_timing, args=(task_duration,), label=f"t{i}")
            for i in range(2)
        ]

        with (
            patch("src.parallel.get_system_resources", return_value=fake_res),
            patch("src.parallel._probe_gpu_free_memory",
                  return_value={0: 20_000.0, 1: 20_000.0}),
        ):
            results = run_parallel_tasks(tasks, device_strategy="auto")

        self.assertEqual(len(results), 2)
        for r in results:
            self.assertEqual(r["status"], "ok")

        t0 = results[0]["result"]
        t1 = results[1]["result"]

        # Tasks must overlap in time — if they were serialized on the same
        # GPU there'd be no overlap.
        overlap_start = max(t0["start"], t1["start"])
        overlap_end = min(t0["end"], t1["end"])
        self.assertGreater(
            overlap_end - overlap_start, 0.0,
            "Tasks on different GPUs should run concurrently",
        )


class TestWorkerPoolConfiguration(unittest.TestCase):
    """B1: multi-GPU runs must use ``max_tasks_per_child=1``.

    Worker reuse would re-run ``os.environ['CUDA_VISIBLE_DEVICES'] = ...`` on
    an already-initialized CUDA context, silently pinning every task on a
    given worker to that worker's first GPU. We can't exercise real CUDA
    here, but we *can* assert the scheduler hands ``ProcessPoolExecutor`` the
    right ``max_tasks_per_child`` value for each topology.
    """

    @staticmethod
    def _spy_executor() -> tuple[type, list[dict]]:
        """Build a ``ProcessPoolExecutor`` stand-in that records init kwargs.

        Uses real :class:`concurrent.futures.Future` objects (pre-resolved)
        so the scheduler's ``wait(...)`` call works without spawning actual
        subprocesses.
        """

        from concurrent.futures import Future

        init_calls: list[dict] = []

        class _FakeExecutor:
            def __init__(self, **kwargs: object) -> None:
                init_calls.append(kwargs)

            def submit(self, _func, *args, **_kwargs):  # noqa: ANN001
                # Scheduler signature:
                # ``executor.submit(_worker_wrapper, inner_func, inner_args,
                #                   inner_kwargs, cuda_dev, label)``
                inner_args = args[1] if len(args) >= 2 else ()
                return_value = inner_args[0] if inner_args else None
                label = args[4] if len(args) >= 5 else "spy"
                fut: Future = Future()
                fut.set_result({
                    "status": "ok",
                    "label": label,
                    "result": return_value,
                    "elapsed_s": 0.0,
                })
                return fut

            def shutdown(self, **_kwargs: object) -> None:
                return None

        return _FakeExecutor, init_calls

    def test_multi_gpu_sets_max_tasks_per_child_one(self) -> None:
        FakeExecutor, init_calls = self._spy_executor()

        fake_res = SystemResources(
            cpu_count=32,
            load_avg_1m=0.0,
            available_cpus=32.0,
            gpus=(
                GPUInfo(index=0, total_memory_mb=24_000, free_memory_mb=20_000),
                GPUInfo(index=1, total_memory_mb=24_000, free_memory_mb=20_000),
            ),
        )
        tasks = [TaskSpec(func=_return_value, args=(i,), label=f"t{i}") for i in range(2)]

        with (
            patch("src.parallel.get_system_resources", return_value=fake_res),
            patch("src.parallel._probe_gpu_free_memory",
                  return_value={0: 20_000.0, 1: 20_000.0}),
            patch("src.parallel.ProcessPoolExecutor", FakeExecutor),
        ):
            run_parallel_tasks(tasks, device_strategy="auto")

        self.assertEqual(len(init_calls), 1)
        self.assertEqual(init_calls[0].get("max_tasks_per_child"), 1)

    def test_single_gpu_allows_worker_reuse(self) -> None:
        FakeExecutor, init_calls = self._spy_executor()

        fake_res = SystemResources(
            cpu_count=32,
            load_avg_1m=0.0,
            available_cpus=32.0,
            gpus=(GPUInfo(index=0, total_memory_mb=24_000, free_memory_mb=20_000),),
        )
        tasks = [TaskSpec(func=_return_value, args=(i,), label=f"t{i}") for i in range(2)]

        with (
            patch("src.parallel.get_system_resources", return_value=fake_res),
            patch("src.parallel._probe_gpu_free_memory", return_value={0: 20_000.0}),
            patch("src.parallel.ProcessPoolExecutor", FakeExecutor),
        ):
            run_parallel_tasks(tasks, device_strategy="auto")

        self.assertEqual(len(init_calls), 1)
        # Single GPU: no worker-reuse risk, so no per-task restart penalty.
        self.assertIsNone(init_calls[0].get("max_tasks_per_child"))

    def test_cpu_only_allows_worker_reuse(self) -> None:
        FakeExecutor, init_calls = self._spy_executor()

        fake_res = SystemResources(
            cpu_count=8, load_avg_1m=0.0, available_cpus=8.0, gpus=(),
        )
        tasks = [TaskSpec(func=_return_value, args=(i,), label=f"t{i}") for i in range(2)]

        with (
            patch("src.parallel.get_system_resources", return_value=fake_res),
            patch("src.parallel.ProcessPoolExecutor", FakeExecutor),
        ):
            run_parallel_tasks(tasks, device_strategy="cpu")

        self.assertEqual(len(init_calls), 1)
        self.assertIsNone(init_calls[0].get("max_tasks_per_child"))


class TestGpuReprobe(unittest.TestCase):
    """Live-probe resampling for shared-server correctness."""

    def test_cotenant_releases_vram_unblocks_waiting_task(self) -> None:
        """A task rejected while a co-tenant held VRAM gets admitted once they release.

        First N probes show only 2 GB free — below the 5 GB default threshold.
        After that, the probe shows 20 GB free, so the pending task is admitted.
        """
        probe_call_count = 0

        def fake_probe(indices: list[int]) -> dict[int, float]:
            nonlocal probe_call_count
            probe_call_count += 1
            if probe_call_count <= 2:
                return {0: 2_000.0}  # co-tenant still holding 22 GB on a 24 GB card
            return {0: 20_000.0}     # co-tenant released → scheduler admits

        fake_res = SystemResources(
            cpu_count=32,
            load_avg_1m=0.0,
            available_cpus=32.0,
            gpus=(GPUInfo(index=0, total_memory_mb=24_000, free_memory_mb=2_000),),
        )

        tasks = [TaskSpec(func=_return_value, args=(0,), label="t0")]

        with (
            patch("src.parallel.get_system_resources", return_value=fake_res),
            patch("src.parallel._probe_gpu_free_memory", side_effect=fake_probe),
            patch("src.parallel.time.sleep"),  # don't block the test waiting for backoff
        ):
            results = run_parallel_tasks(tasks, device_strategy="auto")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "ok")
        # Must have called probe at least 3 times (2 rejections + 1 admit).
        self.assertGreaterEqual(probe_call_count, 3)

    def test_cotenant_drop_blocks_admission_until_recovery(self) -> None:
        """When the live probe dips below the task's need, admission is deferred.

        Probe sequence: first two ticks return 2 GB (co-tenant holding VRAM),
        later ticks return 20 GB (released). The single task is held off for
        the low-VRAM window, then admitted once VRAM returns.
        """
        probe_state = {"calls": 0}

        def fake_probe(indices: list[int]) -> dict[int, float]:
            probe_state["calls"] += 1
            if probe_state["calls"] <= 2:
                return {0: 2_000.0}  # below 5 GB threshold → defer
            return {0: 20_000.0}

        fake_res = SystemResources(
            cpu_count=32,
            load_avg_1m=0.0,
            available_cpus=32.0,
            gpus=(GPUInfo(index=0, total_memory_mb=24_000, free_memory_mb=20_000),),
        )

        tasks = [TaskSpec(func=_return_value, args=(0,), label="t0")]

        with (
            patch("src.parallel.get_system_resources", return_value=fake_res),
            patch("src.parallel._probe_gpu_free_memory", side_effect=fake_probe),
            patch("src.parallel.time.sleep"),
        ):
            results = run_parallel_tasks(tasks, device_strategy="auto")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "ok")
        # Probe must have run during the low-VRAM ticks AND once more to see
        # the recovery. ``probe_state["calls"] >= 3`` is guaranteed by the
        # wait-and-retry loop in the scheduler.
        self.assertGreaterEqual(probe_state["calls"], 3)

    def test_probe_failure_skips_gpu_this_tick_and_retries(self) -> None:
        """If one GPU's probe fails a tick, the scheduler skips it and retries next tick.

        The fake probe returns only GPU 1 on the first tick (GPU 0 "probe
        failed" — omitted from the dict). Next tick both GPUs return. Both
        tasks still complete — GPU 0's transient failure doesn't abort.
        """
        tick = {"n": 0}

        def fake_probe(indices: list[int]) -> dict[int, float]:
            tick["n"] += 1
            if tick["n"] == 1:
                return {1: 20_000.0}  # GPU 0 probe "failed"
            return {0: 20_000.0, 1: 20_000.0}

        fake_res = SystemResources(
            cpu_count=32,
            load_avg_1m=0.0,
            available_cpus=32.0,
            gpus=(
                GPUInfo(index=0, total_memory_mb=24_000, free_memory_mb=20_000),
                GPUInfo(index=1, total_memory_mb=24_000, free_memory_mb=20_000),
            ),
        )

        tasks = [
            TaskSpec(func=_return_value, args=(i,), label=f"t{i}")
            for i in range(2)
        ]

        with (
            patch("src.parallel.get_system_resources", return_value=fake_res),
            patch("src.parallel._probe_gpu_free_memory", side_effect=fake_probe),
        ):
            results = run_parallel_tasks(tasks, device_strategy="auto")

        self.assertEqual(len(results), 2)
        for r in results:
            self.assertEqual(r["status"], "ok")

    def test_grace_window_subtracts_recent_reserves_from_probe(self) -> None:
        """A just-submitted task's reserve is subtracted from the live probe.

        Without grace-window accounting, two quick submissions on the same
        GPU would both see "20 GB free" (the driver hasn't observed task 0's
        allocation yet) and both admit when only 20 GB is actually available.
        The scheduler must instead see 20 - 15 = 5 GB effective after task 0
        is queued, below the 10 GB threshold, and refuse task 1 for this tick.
        """
        fake_res = SystemResources(
            cpu_count=32,
            load_avg_1m=0.0,
            available_cpus=32.0,
            gpus=(GPUInfo(index=0, total_memory_mb=24_000, free_memory_mb=20_000),),
        )

        task_duration = 0.2
        tasks = [
            TaskSpec(
                func=_record_timing,
                args=(task_duration,),
                label=f"t{i}",
                vram_needed_mb=10_000,
                vram_reserve_mb=15_000,  # aggressive reserve — only one fits at a time
            )
            for i in range(2)
        ]

        # Driver probe always reports 20 GB — simulating the pre-allocation
        # window before the worker has called .to('cuda'). Without grace
        # accounting, both tasks would be admitted simultaneously. With it,
        # task 1 must wait for task 0 to complete and release its reserve.
        with (
            patch("src.parallel.get_system_resources", return_value=fake_res),
            patch("src.parallel._probe_gpu_free_memory", return_value={0: 20_000.0}),
        ):
            results = run_parallel_tasks(tasks, device_strategy="auto")

        self.assertEqual(len(results), 2)
        for r in results:
            self.assertEqual(r["status"], "ok")

        t0 = results[0]["result"]
        t1 = results[1]["result"]
        self.assertGreaterEqual(
            t1["start"], t0["end"] - 0.05,
            "Grace-window reserve should serialize tasks on the same GPU",
        )


class TestCpuOnlyTaskBypassesGpuGate(unittest.TestCase):
    """``vram_needed_mb <= 0`` marks a pure-CPU task (e.g. DBAS): it must run
    on a GPU box even when no GPU has free VRAM, never blocking on / reserving
    memory it doesn't use."""

    def test_zero_vram_task_runs_when_gpu_full(self) -> None:
        fake_res = SystemResources(
            cpu_count=8,
            load_avg_1m=0.0,
            available_cpus=8.0,
            # A GPU is present but completely full.
            gpus=(GPUInfo(index=0, total_memory_mb=24_000, free_memory_mb=0),),
        )
        tasks = [
            TaskSpec(func=_return_value, args=(i,), label=f"t{i}",
                     vram_needed_mb=0.0, vram_reserve_mb=0.0)
            for i in range(3)
        ]
        with (
            patch("src.parallel.get_system_resources", return_value=fake_res),
            patch("src.parallel._probe_gpu_free_memory", return_value={0: 0.0}),
        ):
            results = run_parallel_tasks(tasks, device_strategy="auto")

        # Without the bypass these would wait forever on a GPU that never frees.
        self.assertEqual(len(results), 3)
        for i, r in enumerate(results):
            self.assertEqual(r["status"], "ok")
            self.assertEqual(r["result"], i)


class TestBrokenPool(unittest.TestCase):
    """A worker killed mid-run breaks the pool; the scheduler must surface that
    as task failures (via ParallelTaskErrors), not let BrokenProcessPool escape
    and crash the whole run."""

    def test_broken_pool_surfaces_as_failures(self) -> None:
        # max_workers=1 forces strict serialization: t0 kills its worker, then
        # the submit of t1 hits the now-broken pool and the guard records the
        # unsubmitted remainder.
        tasks = [
            TaskSpec(func=_hard_exit if i == 0 else _return_value,
                     args=(i,), label=f"t{i}")
            for i in range(3)
        ]
        with self.assertRaises(ParallelTaskErrors) as ctx:
            run_parallel_tasks(tasks, device_strategy="cpu", max_workers=1)

        # Every task is accounted for as a failure — none silently vanish.
        failures = ctx.exception.failures
        self.assertEqual({f["label"] for f in failures}, {"t0", "t1", "t2"})
        # t1/t2 were never submitted once the pool broke.
        not_submitted = [f for f in failures if "not submitted" in f["error"]]
        self.assertTrue(not_submitted)


if __name__ == "__main__":
    unittest.main()
