"""Seed-level summary statistics and paired significance tests.

Exactness label: implementation convenience only.

Average evaluation scenarios within each training seed, then report the mean,
standard deviation, and confidence interval across independently trained
models. Significance uses an exact two-sided paired sign-flip test on matched
training-seed means. Scenario spread is reported separately.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from pathlib import Path

import numpy as np

from src.evaluation.provenance import sha256_file
from src.evaluation.sweep import ALGORITHM_LABELS, aggregate_sweep_rows

# The three headline metrics the paper reports in its comparison figures.
METRICS: tuple[tuple[str, str], ...] = (
    ("social_welfare", "Social Welfare"),
    ("raw_rsu_reward", "RSU Utility"),
    ("perception_quality_total", "Effective Sensing Quality"),
)

DEFAULT_CSV = Path("outputs/revision_2026_v5/headline/episodes.csv")
DEFAULT_SWEEP_VALUE = None
OUTPUT_DIR = Path("outputs/revision_2026_v5/tables")

# Algorithm order for the table: proposed first, then the baselines in the
# order used by the maintained evaluation section.
ALGORITHM_ORDER: tuple[str, ...] = (
    "proposed",
    "jdls",
    "mappo",
    "ebas",
    "spis",
    "rdus",
    "dbas",
)


def per_seed_means(rows: list[dict], algorithm: str, metric: str) -> list[float]:
    """One value per training seed: that seed's mean over its eval bank."""
    by_seed: dict[str, list[float]] = {}
    for row in rows:
        if row["algorithm"] != algorithm:
            continue
        raw = row.get(metric, "")
        if raw in ("", None):
            continue
        value = float(raw)
        if math.isnan(value):
            continue
        by_seed.setdefault(str(row["train_seed"]), []).append(value)
    return [float(np.mean(v)) for v in by_seed.values() if v]


def per_seed_mean_map(rows: list[dict], algorithm: str, metric: str) -> dict[str, float]:
    """One validation/test-bank mean keyed by independent training seed."""
    by_seed: dict[str, list[float]] = {}
    for row in rows:
        if row["algorithm"] != algorithm:
            continue
        raw = row.get(metric, "")
        if raw in ("", None):
            continue
        value = float(raw)
        if math.isfinite(value):
            by_seed.setdefault(str(row["train_seed"]), []).append(value)
    return {seed: float(np.mean(values)) for seed, values in by_seed.items() if values}


def paired_signflip_p(a: dict[str, float], b: dict[str, float]) -> tuple[float, int]:
    """Exact two-sided paired randomization p-value on shared seed identities."""
    shared = sorted(set(a) & set(b))
    if len(shared) < 2:
        return float("nan"), len(shared)
    deltas = np.asarray([a[seed] - b[seed] for seed in shared], dtype=np.float64)
    observed = abs(float(np.mean(deltas)))
    extreme = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(deltas)):
        statistic = abs(float(np.mean(deltas * np.asarray(signs))))
        if statistic >= observed - 1e-12:
            extreme += 1
    return extreme / (2 ** len(deltas)), len(shared)


def paired_improvement_summary(
    proposed: dict[str, float], baseline: dict[str, float]
) -> tuple[float, float, float]:
    """Mean/std/CI of matched per-seed percentage improvements."""
    values = [
        100.0 * (proposed[seed] - baseline[seed]) / abs(baseline[seed])
        for seed in sorted(set(proposed) & set(baseline))
        if baseline[seed] != 0.0
    ]
    if not values:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, float("nan"), float("nan")
    std = float(np.std(values, ddof=1))
    return mean, std, 1.96 * std / math.sqrt(len(values))


def cohens_d(a: list[float], b: list[float]) -> float:
    """Pooled-SD standardized mean difference."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    pooled_sd = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled_sd <= 0.0:
        return float("nan")
    return (float(np.mean(a)) - float(np.mean(b))) / pooled_sd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"per-episode sweep CSV (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--sweep-value",
        type=float,
        default=DEFAULT_SWEEP_VALUE,
        help="optional sweep grid point to slice, e.g. 20.0 vehicles",
    )
    parser.add_argument(
        "--baseline", default="ebas", help="algorithm to test the proposal against (default: ebas)"
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)

    if not args.csv.exists():
        print(f"error: {args.csv} not found; run the evaluate stage first.")
        return 1

    with args.csv.open(newline="", encoding="utf-8") as handle:
        all_rows = list(csv.DictReader(handle))
    rows = (
        all_rows
        if args.sweep_value is None
        else [
            r
            for r in all_rows
            if r.get("sweep_value") and float(r["sweep_value"]) == args.sweep_value
        ]
    )
    if not rows:
        print(f"error: no rows at sweep_value == {args.sweep_value} in {args.csv}")
        return 1

    # Scenario spread comes from the shared aggregator so the two definitions
    # cannot drift apart; seed spread is recomputed here per algorithm.
    scenario_agg = {
        r["algorithm"]: r for r in aggregate_sweep_rows(rows, group_fields=("algorithm",))
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    present = [a for a in ALGORITHM_ORDER if a in scenario_agg]
    source_csv_sha256 = sha256_file(args.csv)

    summary: list[dict] = []
    for algorithm in present:
        record: dict = {
            "algorithm": algorithm,
            "algorithm_label": ALGORITHM_LABELS.get(algorithm, algorithm),
            "source_csv_sha256": source_csv_sha256,
        }
        for metric, _ in METRICS:
            seeds = per_seed_means(rows, algorithm, metric)
            record[f"{metric}_mean"] = float(np.mean(seeds)) if seeds else float("nan")
            record[f"{metric}_seed_std"] = (
                float(np.std(seeds, ddof=1)) if len(seeds) > 1 else float("nan")
            )
            record[f"{metric}_n_seeds"] = len(seeds)
            record[f"{metric}_seed_ci95"] = (
                1.96 * float(np.std(seeds, ddof=1)) / math.sqrt(len(seeds))
                if len(seeds) > 1
                else float("nan")
            )
            record[f"{metric}_scenario_std"] = scenario_agg[algorithm].get(
                f"{metric}_std", float("nan")
            )
        summary.append(record)

    summary_path = args.output_dir / "summary_stats.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    # ---- significance: proposed vs every baseline -------------------------
    significance: list[dict] = []
    for metric, _ in METRICS:
        proposed = per_seed_means(rows, "proposed", metric)
        for algorithm in present:
            if algorithm == "proposed":
                continue
            other = per_seed_means(rows, algorithm, metric)
            proposed_map = per_seed_mean_map(rows, "proposed", metric)
            other_map = per_seed_mean_map(rows, algorithm, metric)
            paired_p, paired_n = paired_signflip_p(proposed_map, other_map)
            complete_pairing = len(proposed_map) >= 2 and set(proposed_map) == set(other_map)
            improvement_mean, improvement_std, improvement_ci = paired_improvement_summary(
                proposed_map, other_map
            )
            mean_p = float(np.mean(proposed)) if proposed else float("nan")
            mean_o = float(np.mean(other)) if other else float("nan")
            significance.append(
                {
                    "metric": metric,
                    "baseline": algorithm,
                    "proposed_mean": mean_p,
                    "baseline_mean": mean_o,
                    "difference": mean_p - mean_o,
                    "improvement_pct": (
                        100.0 * (mean_p - mean_o) / abs(mean_o)
                        if mean_o not in (0.0,) and not math.isnan(mean_o)
                        else float("nan")
                    ),
                    "paired_improvement_pct_mean": improvement_mean,
                    "paired_improvement_pct_std": improvement_std,
                    "paired_improvement_pct_ci95": improvement_ci,
                    "n_proposed": len(proposed),
                    "n_baseline": len(other),
                    "paired_signflip_p": paired_p,
                    "paired_seed_count": paired_n,
                    "paired_seed_complete": complete_pairing,
                    "primary_test": (
                        "exact_two_sided_paired_sign_flip" if complete_pairing else ""
                    ),
                    "primary_test_status": (
                        "complete_paper"
                        if complete_pairing
                        else "diagnostic_only_non_final_insufficient_pairs"
                    ),
                    "primary_p_value": paired_p if complete_pairing else "",
                    "primary_pair_count": paired_n,
                    "cohens_d": cohens_d(proposed, other),
                    "source_csv_sha256": source_csv_sha256,
                }
            )

    selected = [row for row in significance if row["baseline"] == args.baseline]
    if not selected:
        print(f"error: baseline {args.baseline!r} is absent from {args.csv}")
        return 1
    incomplete = [row["metric"] for row in selected if not row["paired_seed_complete"]]
    if incomplete:
        metrics = ", ".join(map(str, incomplete))
        print(
            f"error: baseline {args.baseline!r} lacks complete matched training-seed "
            f"pairs for: {metrics}"
        )
        return 1

    significance_path = args.output_dir / "significance.csv"
    with significance_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(significance[0].keys()))
        writer.writeheader()
        writer.writerows(significance)

    # ---- console report ---------------------------------------------------
    print(f"\nSource: {args.csv}  (sweep_value = {args.sweep_value})\n")
    for metric, label in METRICS:
        print(f"{label}")
        print(f"  {'algorithm':<20}{'mean':>10}{'seed std':>10}{'n':>4}{'scenario std':>14}")
        for record in summary:
            n = record[f"{metric}_n_seeds"]
            sd = record[f"{metric}_seed_std"]
            sd_str = "  n/a" if math.isnan(sd) else f"{sd:.3f}"
            print(
                f"  {record['algorithm_label']:<20}"
                f"{record[f'{metric}_mean']:>10.3f}{sd_str:>10}{n:>4}"
                f"{record[f'{metric}_scenario_std']:>14.3f}"
            )
        print()

    print(f"Significance: TE-PPO vs {args.baseline} (exact two-sided paired sign-flip test)\n")
    for row in significance:
        if row["baseline"] != args.baseline:
            continue
        print(
            f"  {row['metric']:<26} "
            f"{row['proposed_mean']:.3f} vs {row['baseline_mean']:.3f}  "
            f"({row['paired_improvement_pct_mean']:+.2f}% "
            f"± {row['paired_improvement_pct_ci95']:.2f}% CI95)  "
            f"pairs={row['primary_pair_count']}  "
            f"p={row['primary_p_value']:.4f} (exact)"
        )

    print(f"\n  -> {summary_path}")
    print(f"  -> {significance_path}")
    if any(r["n_proposed"] < 5 for r in significance):
        print(
            "\n  NOTE: with fewer than 5 seeds the smallest attainable p-value "
            "is bounded\n        by the number of distinct splits; complete the "
            "headline seed bank declared in configs/revision.yaml."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
