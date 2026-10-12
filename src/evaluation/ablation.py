"""Controlled configurations for the four-component ablation.

The full model and all variants share the same scenario, seed bank, training
budget, PPO network width, and test scenarios.  An arm changes only the
decision mechanism required to remove its named component; where that changes
the action space (fixed price/equal bandwidth/joint follower learning), the
engine change is explicit in :data:`ABLATION_ENGINES`.

Exactness label: implementation convenience only (experimental controls).
"""

from __future__ import annotations

import copy
import csv
import itertools
import math
from pathlib import Path
from typing import Any

from src.evaluation.revision import (
    formal_algorithm_config,
    formal_provenance_overrides,
    load_revision_config,
    revision_roots,
)

ABLATION_ARMS = (
    "full",
    "no_dynamic_pricing",
    "no_bandwidth_allocation",
    "no_analytical_best_response",
    "no_candidate_pruning",
)

ABLATION_ENGINES = {
    "full": "proposed",
    "no_dynamic_pricing": "spis",
    "no_bandwidth_allocation": "ebas",
    "no_analytical_best_response": "jdls",
    "no_candidate_pruning": "proposed",
}

ABLATION_UNCONDITIONAL_METRICS = (
    "social_welfare",
    "raw_rsu_reward",
    "perception_quality_total",
    "energy_total_sum",
    "participation_rate",
    "selected_vehicle_count",
    "task_success_rate",
    "candidate_pruned_pair_ratio",
)
ABLATION_CONDITIONAL_RATIO_METRIC = "energy_per_participating_vehicle_j"


def _matched_seed_rows(
    rows: list[dict[str, object]],
    *,
    arm: str,
    train_seed: int,
    eval_seeds: tuple[int, ...],
    metric_label: str,
) -> list[dict[str, object]]:
    matching = [
        row for row in rows
        if row.get("ablation_arm") == arm
        and int(row.get("train_seed", -1)) == train_seed
    ]
    actual_eval_seeds = [int(row.get("eval_seed", -1)) for row in matching]
    if (
        set(actual_eval_seeds) != set(eval_seeds)
        or len(actual_eval_seeds) != len(set(actual_eval_seeds))
    ):
        raise ValueError(
            f"{arm}/{metric_label}/seed={train_seed} lacks the complete "
            "unique test-seed bank"
        )
    return matching


def _delta_statistics(deltas: list[float]) -> dict[str, float | int]:
    mean = sum(deltas) / len(deltas)
    if len(deltas) > 1:
        variance = sum((value - mean) ** 2 for value in deltas) / (len(deltas) - 1)
        std = math.sqrt(variance)
        ci95 = 1.96 * std / math.sqrt(len(deltas))
    else:
        std = float("nan")
        ci95 = float("nan")

    observed = abs(mean)
    extreme = 0
    total = 2 ** len(deltas)
    for signs in itertools.product((-1.0, 1.0), repeat=len(deltas)):
        permuted = abs(
            sum(
                sign * value
                for sign, value in zip(signs, deltas, strict=True)
            )
            / len(deltas)
        )
        if permuted >= observed - 1e-12:
            extreme += 1

    return {
        "mean": mean,
        "std": std,
        "ci95": ci95,
        "paired_permutation_p": extreme / total,
        "pairs": len(deltas),
    }


def paired_ablation_delta_summary(
    rows: list[dict[str, object]],
    *,
    arm: str,
    metric: str,
    train_seeds: tuple[int, ...],
    eval_seeds: tuple[int, ...],
) -> dict[str, float | int]:
    """Summarize matched-seed percentage deltas against the full arm.

    Each independently trained checkpoint first contributes one mean over the
    complete held-out scenario bank.  Exactness label: implementation
    convenience only (the maintained definition is in the revision protocol).
    """

    seed_means: dict[tuple[str, int], float] = {}
    for candidate_arm in ("full", arm):
        for seed in train_seeds:
            matching = _matched_seed_rows(
                rows,
                arm=candidate_arm,
                train_seed=seed,
                eval_seeds=eval_seeds,
                metric_label=metric,
            )
            values = [float(row.get(metric, float("nan"))) for row in matching]
            if not values or not all(math.isfinite(value) for value in values):
                raise ValueError(
                    f"{candidate_arm}/{metric}/seed={seed} has non-finite values"
                )
            seed_means[(candidate_arm, seed)] = sum(values) / len(values)

    deltas: list[float] = []
    for seed in train_seeds:
        baseline = seed_means[("full", seed)]
        if baseline == 0.0:
            raise ValueError(f"full/{metric}/seed={seed} has a zero denominator")
        value = seed_means[(arm, seed)]
        deltas.append(100.0 * (value - baseline) / abs(baseline))

    return _delta_statistics(deltas)


def conditional_ratio_summary(
    rows: list[dict[str, object]],
    *,
    arm: str,
    numerator: str,
    denominator: str,
    train_seeds: tuple[int, ...],
    eval_seeds: tuple[int, ...],
) -> dict[str, float | int | str]:
    """Summarize a ratio of sums that is conditional on a positive denominator."""

    ratios: list[float] = []
    valid_episode_count = 0
    denominator_total = 0.0
    for seed in train_seeds:
        matching = _matched_seed_rows(
            rows,
            arm=arm,
            train_seed=seed,
            eval_seeds=eval_seeds,
            metric_label=f"{numerator}/{denominator}",
        )
        numerators = [float(row.get(numerator, float("nan"))) for row in matching]
        denominators = [float(row.get(denominator, float("nan"))) for row in matching]
        if not all(math.isfinite(value) for value in (*numerators, *denominators)):
            raise ValueError(
                f"{arm}/{numerator}/{denominator}/seed={seed} has non-finite values"
            )
        if any(value < 0.0 for value in denominators):
            raise ValueError(
                f"{arm}/{denominator}/seed={seed} has a negative denominator"
            )
        seed_denominator = math.fsum(denominators)
        denominator_total += seed_denominator
        valid_episode_count += sum(value > 0.0 for value in denominators)
        if seed_denominator > 0.0:
            ratios.append(math.fsum(numerators) / seed_denominator)

    if not ratios:
        mean = std = ci95 = float("nan")
        status = "not_estimable_no_participants"
    else:
        mean = sum(ratios) / len(ratios)
        if len(ratios) > 1:
            std = math.sqrt(
                sum((value - mean) ** 2 for value in ratios) / (len(ratios) - 1)
            )
            ci95 = 1.96 * std / math.sqrt(len(ratios))
        else:
            std = ci95 = float("nan")
        status = (
            "estimable"
            if len(ratios) == len(train_seeds)
            else "partially_estimable"
        )

    return {
        "mean": mean,
        "std": std,
        "ci95": ci95,
        "seed_mean": mean,
        "seed_std": std,
        "seed_ci95": ci95,
        "replicates": len(ratios),
        "seed_replicates": len(ratios),
        "valid_count": valid_episode_count,
        "denominator_total": denominator_total,
        "status": status,
    }


def paired_ablation_ratio_delta_summary(
    rows: list[dict[str, object]],
    *,
    arm: str,
    numerator: str,
    denominator: str,
    train_seeds: tuple[int, ...],
    eval_seeds: tuple[int, ...],
) -> dict[str, float | int | str]:
    """Return paired deltas for a conditional ratio when every pair exists."""

    seed_ratios: dict[tuple[str, int], float | None] = {}
    for candidate_arm in ("full", arm):
        for seed in train_seeds:
            matching = _matched_seed_rows(
                rows,
                arm=candidate_arm,
                train_seed=seed,
                eval_seeds=eval_seeds,
                metric_label=f"{numerator}/{denominator}",
            )
            numerators = [float(row.get(numerator, float("nan"))) for row in matching]
            denominators = [float(row.get(denominator, float("nan"))) for row in matching]
            if not all(math.isfinite(value) for value in (*numerators, *denominators)):
                raise ValueError(
                    f"{candidate_arm}/{numerator}/{denominator}/seed={seed} "
                    "has non-finite values"
                )
            if any(value < 0.0 for value in denominators):
                raise ValueError(
                    f"{candidate_arm}/{denominator}/seed={seed} "
                    "has a negative denominator"
                )
            denominator_sum = math.fsum(denominators)
            seed_ratios[(candidate_arm, seed)] = (
                math.fsum(numerators) / denominator_sum
                if denominator_sum > 0.0
                else None
            )

    deltas: list[float] = []
    missing_participation = False
    zero_reference = False
    for seed in train_seeds:
        baseline = seed_ratios[("full", seed)]
        value = seed_ratios[(arm, seed)]
        if baseline is None or value is None:
            missing_participation = True
            continue
        if baseline == 0.0:
            zero_reference = True
            continue
        deltas.append(100.0 * (value - baseline) / abs(baseline))

    if len(deltas) != len(train_seeds):
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "ci95": float("nan"),
            "paired_permutation_p": float("nan"),
            "pairs": len(deltas),
            "status": (
                "not_estimable_no_participants"
                if missing_participation and not deltas
                else "not_estimable_zero_full_reference"
                if zero_reference and not deltas
                else "not_estimable_incomplete_pairs"
            ),
        }
    return {**_delta_statistics(deltas), "status": "estimable"}


def calibrated_price_from_run(run_dir: str | Path) -> float:
    """Mean validation-set price emitted by one full-model training run."""
    path = Path(run_dir) / "eval" / "selected_validation_rows.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"fixed-price ablation requires full-model validation rows: {path}"
        )
    values: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            raw = row.get("average_price")
            if raw not in (None, ""):
                values.append(float(raw))
    if not values:
        raise ValueError(f"no average_price values in {path}")
    return sum(values) / len(values)


def build_ablation_config(
    arm: str,
    seed: int,
    *,
    fixed_price: float | None = None,
    revision_config: dict[str, Any] | None = None,
    base_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Materialize one ablation arm from the frozen full-method config.

    Formal callers use the materialized experiment configuration.  Tests may
    inject ``base_config`` explicitly so they do not depend on formal output
    artifacts.
    """
    if arm not in ABLATION_ARMS:
        raise ValueError(f"unknown ablation arm: {arm!r}")
    cfg = revision_config or load_revision_config()
    base = copy.deepcopy(
        base_config
        if base_config is not None
        else formal_algorithm_config("proposed", cfg)
    )
    base["seed"] = int(seed)
    for dotted, value in formal_provenance_overrides(cfg).items():
        node = base
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    base.setdefault("evaluation", {})["seed_list"] = list(cfg["seeds"]["validation"])
    base.setdefault("artifacts", {})["run_name"] = f"{arm}_s{int(seed)}"

    algorithm = base["algorithm"]
    full_rollout_batch = int(algorithm["n_envs"]) * int(algorithm["n_steps"])
    full_minibatch = int(algorithm["batch_size"])
    full_epochs = int(algorithm["n_epochs"])
    if arm == "no_dynamic_pricing":
        if fixed_price is None:
            raise ValueError("no_dynamic_pricing requires a calibrated fixed_price")
        algorithm["name"] = "spis"
        algorithm["fixed_price"] = float(fixed_price)
    elif arm == "no_bandwidth_allocation":
        algorithm["name"] = "ebas"
    elif arm == "no_analytical_best_response":
        # Joint learning uses the declared non-analytical shaping terms.
        # JDLS uses its own custom PPO loop and does not create an SB3
        # VecNormalize state. Override the full arm's normalization block so
        # the materialized config describes the path that is actually run.
        base["normalization"] = {
            "norm_obs": False,
            "norm_reward": False,
        }
        base["algorithm"] = {
            "name": "jdls",
            "implementation_version": str(
                cfg["ablation"]["learned_follower_implementation_version"]
            ),
            "joint_action_parameterization": str(
                cfg["ablation"]["learned_follower_action_parameterization"]
            ),
            "total_timesteps": int(algorithm["total_timesteps"]),
            "rollout_steps": full_rollout_batch,
            "n_epochs": full_epochs,
            "minibatch_size": full_minibatch,
            "learning_rate": float(algorithm["learning_rate"]),
            "target_kl": algorithm.get("target_kl"),
            "clip_range": float(algorithm["clip_range"]),
            "vf_coef": float(algorithm["vf_coef"]),
            "ent_coef": float(algorithm["ent_coef"]),
            "max_grad_norm": float(algorithm["max_grad_norm"]),
            "hidden_sizes": list(cfg["fairness"]["hidden_sizes"]),
            "device": algorithm.get("device", "auto"),
        }
        learned_reward = cfg["ablation"]["learned_follower_reward"]
        base["reward"] = {
            **learned_reward,
            "omega_fail": float(base["reward"]["omega_fail"]),
            "omega_lat": float(base["reward"]["omega_lat"]),
            "omega_part": float(base["reward"]["omega_part"]),
            "min_participants": int(base["reward"]["min_participants"]),
        }
    elif arm == "no_candidate_pruning":
        base["scenario"]["enable_candidate_pruning"] = False

    # Evaluation cadence is measured in physical environment steps.
    interval = int(cfg["ablation"]["evaluation_interval_env_steps"])
    engine_algorithm = base["algorithm"]
    rollout_quantum = int(engine_algorithm.get("n_envs", 1))
    if "rollout_steps" in engine_algorithm:
        rollout_quantum = int(engine_algorithm["rollout_steps"])
    if interval % rollout_quantum:
        raise ValueError("ablation evaluation interval must align with its engine")
    base["evaluation"]["interval_env_steps"] = interval
    base["evaluation"].pop("eval_freq", None)
    return base


def ablation_run_dir(arm: str, seed: int,
                     revision_config: dict[str, Any] | None = None) -> Path:
    cfg = revision_config or load_revision_config()
    artifacts_root, _ = revision_roots(cfg)
    return artifacts_root / "ablation" / f"{arm}_s{int(seed)}"


def validate_controlled_ablation(configs: dict[str, dict[str, Any]]) -> None:
    """Fail when shared controls drift across the five materialized arms."""
    missing = set(ABLATION_ARMS) - configs.keys()
    if missing:
        raise ValueError(f"missing ablation configs: {sorted(missing)}")
    full = configs["full"]
    shared_scenario = dict(full["scenario"])
    full_algorithm = full["algorithm"]
    expected_rollout = int(full_algorithm["n_envs"]) * int(full_algorithm["n_steps"])
    expected_minibatch = int(full_algorithm["batch_size"])
    expected_epochs = int(full_algorithm["n_epochs"])
    expected_eval_steps = int(full["evaluation"]["interval_env_steps"])
    expected_hidden = list(
        full_algorithm.get(
            "hidden_sizes", full.get("policy", {}).get("actor", {}).get("hidden_sizes")
        )
    )
    for arm, config in configs.items():
        scenario = dict(config["scenario"])
        if arm == "no_candidate_pruning":
            scenario["enable_candidate_pruning"] = shared_scenario["enable_candidate_pruning"]
        if scenario != shared_scenario:
            raise ValueError(f"ablation arm {arm} changes uncontrolled scenario fields")
        if int(config["algorithm"]["total_timesteps"]) != int(
            full["algorithm"]["total_timesteps"]
        ):
            raise ValueError(f"ablation arm {arm} changes the training budget")
        algorithm = config["algorithm"]
        rollout = (
            int(algorithm["n_envs"]) * int(algorithm["n_steps"])
            if "n_envs" in algorithm else int(algorithm["rollout_steps"])
        )
        minibatch = int(
            algorithm.get("batch_size", algorithm.get("minibatch_size", 0))
        )
        epochs = int(algorithm["n_epochs"])
        eval_steps = int(config["evaluation"]["interval_env_steps"])
        if rollout != expected_rollout:
            raise ValueError(f"ablation arm {arm} changes effective rollout batch")
        if minibatch != expected_minibatch:
            raise ValueError(f"ablation arm {arm} changes minibatch size")
        if epochs != expected_epochs:
            raise ValueError(f"ablation arm {arm} changes update epochs")
        if math.ceil(rollout / minibatch) != math.ceil(
            expected_rollout / expected_minibatch
        ):
            raise ValueError(f"ablation arm {arm} changes minibatches per epoch")
        if eval_steps != expected_eval_steps:
            raise ValueError(f"ablation arm {arm} changes evaluation cadence")
        if list(config["evaluation"]["seed_list"]) != list(
            full["evaluation"]["seed_list"]
        ):
            raise ValueError(f"ablation arm {arm} changes validation seeds")
        hidden = algorithm.get(
            "hidden_sizes", config.get("policy", {}).get("actor", {}).get("hidden_sizes")
        )
        if list(hidden) != expected_hidden:
            raise ValueError(f"ablation arm {arm} changes network width")
        if arm == "no_analytical_best_response":
            normalization = config.get("normalization", {})
            if normalization.get("norm_obs", False) or normalization.get(
                "norm_reward", False
            ):
                raise ValueError(
                    "no_analytical_best_response uses the custom JDLS loop, "
                    "which does not support observation or reward normalization"
                )
