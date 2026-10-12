"""Report operational training cost from current per-run timing records.

Exactness label: implementation convenience only.

Accept schema-v2 training_record.json sidecars matching the colocated config
fingerprint. Scheduled training may include CPU contention; isolated learning
curves are reported separately by report_convergence.py. Report budgets and
seconds per 100000 physical environment steps alongside elapsed duration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path

DEFAULT_ARTIFACTS_ROOT = Path("artifacts/revision_2026_v5")
OUTPUT_DIR = Path("outputs/revision_2026_v5/tables")
TRAIN_FINGERPRINT_FILE = "config_fingerprint.txt"

# run_name convention from build_sweep_config: "{algo}_s{seed}{_suffix}".
_VEHICLE_SUFFIX = re.compile(r"_v(\d+)$")


def load_current_runs(
    artifacts_root: Path
) -> list[dict]:
    """Load timing records keyed by artifact identity, not mutable run name.

    Accept only schema-v2 sidecars with matching configuration fingerprints.
    """
    by_identity: dict[tuple[str, str], dict] = {}
    for sidecar in sorted(artifacts_root.rglob("training_record.json")):
        if not (sidecar.parent / "summary.json").exists():
            continue
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("training record must be an object")
            fingerprint_path = sidecar.parent / TRAIN_FINGERPRINT_FILE
            fingerprint = (
                fingerprint_path.read_text(encoding="utf-8").strip()
                if fingerprint_path.exists() else ""
            )
            config_digest = str(payload.get("config_digest") or "")
            if (
                payload.get("schema_version") != 2
                or not fingerprint
                or config_digest != fingerprint
            ):
                continue
            elapsed_s = float(payload["elapsed_s"])
            if not math.isfinite(elapsed_s) or elapsed_s <= 0.0:
                raise ValueError("elapsed_s must be finite and positive")
            run_name = str(payload["run_name"])
            match = _VEHICLE_SUFFIX.search(run_name)
            record = {
                "run_name": run_name,
                "algorithm": str(payload["algorithm"]),
                "train_seed": payload.get("train_seed"),
                "purpose": payload.get("purpose"),
                "num_vehicles": payload.get(
                    "num_vehicles", int(match.group(1)) if match else None
                ),
                "num_tasks": payload.get("num_tasks"),
                "total_timesteps": payload.get("total_timesteps"),
                "elapsed_s": elapsed_s,
                "timing_source": "training_record",
                "run_id": str(payload.get("run_id", "")),
                "config_digest": config_digest,
                "hardware_id": payload.get("hardware_id"),
                "execution_mode": payload.get("execution_mode"),
                "artifact_path": str(sidecar.parent),
            }
            artifact_identity = str(sidecar.parent.resolve())
            by_identity[(artifact_identity, config_digest)] = record
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            print(f"warning: ignoring invalid timing sidecar {sidecar}: {exc}")
    return sorted(by_identity.values(), key=lambda row: (row["run_name"], row.get("artifact_path", "")))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    runs = load_current_runs(args.artifacts_root)
    if not runs:
        print(f"error: no completed training timing records found under "
              f"{args.artifacts_root}")
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- per-algorithm summary -------------------------------------------
    # Only like-for-like records may share a statistical row.  In particular,
    # Distinct shapes, purposes, budgets, hardware, and scheduler modes never mix.
    by_algo: dict[tuple[object, ...], list[dict]] = {}
    for run in runs:
        key = (
            run["algorithm"], run.get("num_vehicles"), run.get("num_tasks"),
            run.get("purpose"), run.get("total_timesteps"),
            run.get("hardware_id"), run.get("execution_mode"),
        )
        by_algo.setdefault(key, []).append(run)

    summary: list[dict] = []
    for key, group in sorted(by_algo.items(), key=lambda item: statistics.mean(
        float(row["elapsed_s"]) for row in item[1]
    )):
        algorithm, vehicles, tasks, purpose, steps, hardware_id, execution_mode = key
        values = [float(run["elapsed_s"]) for run in group]
        median = statistics.median(values)
        summary.append({
            "algorithm": algorithm,
            "num_vehicles": vehicles if vehicles is not None else "",
            "num_tasks": tasks if tasks is not None else "",
            "purpose": purpose or "",
            "hardware_id": hardware_id or "",
            "execution_mode": execution_mode or "",
            "runs": len(values),
            "total_timesteps": steps or "",
            "median_s": round(median, 1),
            "median_min": round(median / 60, 2),
            "mean_s": round(statistics.mean(values), 1),
            "max_s": round(max(values), 1),
            "total_hours": round(sum(values) / 3600, 2),
            "s_per_100k_steps": (
                round(median / (int(steps) / 100_000), 1) if steps else ""
            ),
        })

    summary_path = args.output_dir / "training_cost.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    # ---- per-(algorithm, V) scaling table ---------------------------------
    by_algo_v: dict[tuple[str, int, int | None, object, object, object, object], list[float]] = {}
    for run in runs:
        if run["num_vehicles"] is None:
            continue
        by_algo_v.setdefault((
            run["algorithm"], run["num_vehicles"], run.get("num_tasks"),
            run.get("purpose"), run.get("total_timesteps"),
            run.get("hardware_id"), run.get("execution_mode"),
        ), []).append(
            run["elapsed_s"]
        )

    scaling: list[dict] = []
    for key, values in sorted(by_algo_v.items(), key=lambda item: tuple(
        "" if value is None else str(value) for value in item[0]
    )):
        algorithm, vehicles, tasks, purpose, steps, hardware_id, execution_mode = key
        scaling.append({
            "algorithm": algorithm,
            "num_vehicles": vehicles,
            "num_tasks": tasks if tasks is not None else "",
            "purpose": purpose or "",
            "total_timesteps": steps or "",
            "hardware_id": hardware_id or "",
            "execution_mode": execution_mode or "",
            "runs": len(values),
            "median_s": round(statistics.median(values), 1),
            "median_min": round(statistics.median(values) / 60, 2),
        })

    scaling_path = args.output_dir / "training_cost_vs_vehicles.csv"
    if scaling:
        with scaling_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(scaling[0].keys()))
            writer.writeheader()
            writer.writerows(scaling)

    # ---- console report ---------------------------------------------------
    total_hours = sum(r["elapsed_s"] for r in runs) / 3600
    print(f"\nSource: {args.artifacts_root}  ({len(runs)} current runs)\n")
    print(f"  {'algorithm':<10}{'V':>5}{'K':>5}{'runs':>5}{'steps':>9}{'median min':>12}"
          f"{'total h':>9}{'s/100k steps':>14}")
    for record in summary:
        steps = record["total_timesteps"]
        steps_str = f"{steps // 1000}k" if steps else "n/a"
        norm = record["s_per_100k_steps"]
        print(f"  {record['algorithm']:<10}{record['num_vehicles']:>5}{record['num_tasks']:>5}"
              f"{record['runs']:>5}{steps_str:>9}"
              f"{record['median_min']:>12.1f}{record['total_hours']:>9.1f}"
              f"{norm if norm else 'n/a':>14}")
        print(f"    purpose={record['purpose'] or 'n/a'}; "
              f"hardware={record['hardware_id'] or 'n/a'}; "
              f"mode={record['execution_mode'] or 'n/a'}")
    print(f"\n  Aggregate run wall time: {total_hours:.1f} h across {len(runs)} runs")

    algorithm_seconds: dict[str, float] = {}
    for run in runs:
        algorithm_seconds[run["algorithm"]] = (
            algorithm_seconds.get(run["algorithm"], 0.0) + run["elapsed_s"]
        )
    heaviest = max(algorithm_seconds, key=algorithm_seconds.__getitem__)
    heaviest_hours = algorithm_seconds[heaviest] / 3600
    share = 100 * heaviest_hours / total_hours
    print(f"  {heaviest} accounts for "
          f"{heaviest_hours:.1f} h ({share:.0f}% of the total)")

    print("\n  Scaling cells (median minutes per run):")
    print(f"    {'algorithm':<10}{'V':>5}{'K':>5}{'runs':>7}{'median min':>13}")
    for record in scaling:
        print(
            f"    {record['algorithm']:<10}{record['num_vehicles']:>5}"
            f"{str(record['num_tasks']):>5}{record['runs']:>7}"
            f"{record['median_min']:>13.1f}"
        )

    print(f"\n  -> {summary_path}")
    if scaling:
        print(f"  -> {scaling_path}")
    print("\n  Formal headline/scalability budgets must all be 600k; "
          "quote s/100k steps for throughput and total hours for study cost.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
