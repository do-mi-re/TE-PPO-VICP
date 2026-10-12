"""Quantify theory-embedded action-space reduction.

The report distinguishes physical decision variables, sampled latent variables,
and network output parameters so the comparison does not hide the bandwidth-
utilization latent or count categorical logits as scalar decisions.

Exactness label: implementation convenience only (architecture reporting).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.evaluation.revision import load_revision_config


def dimension_row(num_vehicles: int, num_tasks: int) -> dict[str, int]:
    v, k = int(num_vehicles), int(num_tasks)
    return {
        "num_vehicles": v,
        "num_tasks": k,
        # Prices + bandwidth shares. The analytical follower response supplies
        # task assignment, sensing duration, and compression ratio.
        "te_physical_decisions": k + v,
        "te_sampled_latents": k + 1 + v,
        # Leader decisions plus tau, eta, and one task assignment per vehicle.
        "jdls_physical_decisions": k + 4 * v,
        "jdls_sampled_variables": k + 1 + 4 * v,
        # Continuous mean plus V*K categorical logits; log-std parameters are
        # kept separate because they are distribution parameters, not actions.
        "jdls_actor_output_parameters": k + 1 + 3 * v + v * k,
        "joint_to_te_physical_ratio_x1000": round(
            1000 * (k + 4 * v) / (k + v)
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/revision_2026_v5/tables/action_dimensions.csv"),
    )
    args = parser.parse_args(argv)
    cfg = load_revision_config()
    rows = [
        dimension_row(v, k)
        for k in cfg["scalability"]["task_grid"]
        for v in cfg["scalability"]["vehicle_grid"]
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    default = dimension_row(20, 3)
    print(
        "V=20,K=3: TE-PPO physical/latent="
        f"{default['te_physical_decisions']}/{default['te_sampled_latents']}; "
        "JDLS physical/sampled/output="
        f"{default['jdls_physical_decisions']}/"
        f"{default['jdls_sampled_variables']}/"
        f"{default['jdls_actor_output_parameters']}"
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
