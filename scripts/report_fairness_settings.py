"""Export the configured shared training and evaluation settings.

Exactness label: implementation convenience only.  This report describes the
configured experiment contract; it performs no training or model selection.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from src.evaluation.revision import (
    ALL_ALGORITHMS,
    STOCHASTIC_ALGORITHMS,
    formal_algorithm_config,
    load_revision_config,
)


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _layout(algorithm: str, config: dict[str, Any]) -> tuple[object, ...]:
    if algorithm not in STOCHASTIC_ALGORITHMS:
        return ("N/A", "N/A", "N/A", "N/A")
    algo = config["algorithm"]
    if algorithm in ("jdls", "mappo"):
        return (
            algo["rollout_steps"],
            algo["minibatch_size"],
            algo["n_epochs"],
            str(algo["hidden_sizes"]),
        )
    return (
        int(algo["n_envs"]) * int(algo["n_steps"]),
        algo["batch_size"],
        algo["n_epochs"],
        str(config["policy"]["actor"]["hidden_sizes"]),
    )


def fairness_rows(cfg: dict[str, Any]) -> list[dict[str, object]]:
    method_notes = {
        "proposed": "analytical follower responses; candidate pruning",
        "jdls": "joint-action policy; cold start; C2/C3-feasible action map",
        "mappo": "shared vehicle actor; cold start; C2/C3-feasible action map",
        "dbas": "validation-only fixed-price calibration",
        "ebas": "equal-bandwidth action constraint",
        "spis": "fixed-price action constraint",
        "rdus": "raw upload (eta=1) action constraint",
    }
    rows: list[dict[str, object]] = []
    for algorithm in ALL_ALGORITHMS:
        config = formal_algorithm_config(
            algorithm,
            cfg,
            require_dbas_calibration=False,
        )
        rollout, minibatch, epochs, hidden = _layout(algorithm, config)
        stochastic = algorithm in STOCHASTIC_ALGORITHMS
        rows.append({
            "algorithm": algorithm,
            "protocol": cfg["fairness"]["protocol"],
            "network_hidden_sizes": hidden,
            "learning_rate": config["algorithm"].get("learning_rate", "N/A"),
            "clip_range": config["algorithm"].get("clip_range", "N/A"),
            "entropy_coefficient": config["algorithm"].get("ent_coef", "N/A"),
            "total_timesteps": config["algorithm"].get("total_timesteps", "N/A"),
            "headline_train_seeds": (
                len(cfg["seeds"]["headline_train"]) if stochastic else "N/A"
            ),
            "discount_handling": (
                "one-step return; gamma invariant"
                if algorithm in ("jdls", "mappo")
                else f"gamma={float(config['algorithm']['gamma']):g}"
                if stochastic
                else "N/A"
            ),
            "rollout_samples_per_update": rollout,
            "minibatch_size": minibatch,
            "epochs_per_update": epochs,
            "checkpoint_selection": (
                "maximum validation mean social welfare; earliest tie"
                if stochastic else "N/A"
            ),
            "initialization": "cold start" if algorithm in ("jdls", "mappo")
            else "standard initialization",
            "joint_action_parameterization": config["algorithm"].get(
                "joint_action_parameterization", "N/A"
            ),
            "collection_microbatch_size": config["algorithm"].get(
                "collection_microbatch_size", "N/A"
            ),
            "method_specific_setting": (
                f"{method_notes[algorithm]}; "
                f"price={float(config['algorithm']['fixed_price']):g}"
                if algorithm == "spis"
                else method_notes[algorithm]
            ),
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/revision_2026_v5/tables"),
    )
    args = parser.parse_args(argv)
    cfg = load_revision_config()
    target = args.output_dir / "fairness_settings.csv"
    _write(target, fairness_rows(cfg))
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
