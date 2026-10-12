"""Shared seed-level aggregation for experiment evaluation tables.

Exactness label: implementation convenience only.  The functions preserve the
preregistered training-seed sampling unit without changing simulator metrics.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable

import numpy as np

from src.evaluation.sweep import aggregate_sweep_rows


def _mean_finite(values: Iterable[object]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else math.nan


def summarize_by_seed(
    rows: list[dict[str, object]],
    *,
    group_fields: tuple[str, ...],
    metrics: tuple[str, ...],
) -> list[dict[str, object]]:
    """Average profiles within scenario, then scenarios within training seed."""

    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in group_fields) + (row["train_seed"],)
        grouped[key].append(row)
    result: list[dict[str, object]] = []
    for key, group in grouped.items():
        record = {field: key[index] for index, field in enumerate(group_fields)}
        record["train_seed"] = int(key[-1])
        by_eval: dict[int, list[dict[str, object]]] = defaultdict(list)
        for row in group:
            by_eval[int(row["eval_seed"])].append(row)
        record["evaluation_scenario_count"] = len(by_eval)
        record["source_row_count"] = len(group)
        for metric in metrics:
            per_scenario = [
                _mean_finite(row[metric] for row in scenario_rows)
                for scenario_rows in by_eval.values()
            ]
            record[metric] = _mean_finite(per_scenario)
        result.append(record)
    return sorted(
        result,
        key=lambda row: tuple(str(row[field]) for field in group_fields)
        + (int(row["train_seed"]),),
    )


def summarize_across_training_seeds(
    seed_rows: list[dict[str, object]],
    *,
    group_fields: tuple[str, ...],
    metrics: tuple[str, ...],
) -> list[dict[str, object]]:
    """Form paper statistics from one equally weighted row per training seed."""

    return aggregate_sweep_rows(
        seed_rows,
        group_fields=group_fields,
        metrics=metrics,
        uncertainty="seed",
    )
