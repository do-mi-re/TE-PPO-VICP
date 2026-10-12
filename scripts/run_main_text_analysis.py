"""Evaluate selected-policy pricing and compression mechanisms.

Exactness labels:
- Experiment orchestration: implementation convenience only.
- Compression energy decomposition: paper-exact deterministic equations under
  the documented component counterfactual.

Pricing holds learned bandwidth fixed and recomputes analytical responses.
Compression counterfactuals need not be feasible reoptimized policies.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation import orchestration as orch
from src.evaluation.provenance import (
    validate_run_artifact,
    write_evaluation_contract,
)
from src.evaluation.revision import (
    headline_run_dir,
    headline_specs,
    load_revision_config,
    revision_roots,
    train_seeds_for,
)
from src.evaluation.seed_aggregation import (
    summarize_across_training_seeds,
    summarize_by_seed,
)
from src.evaluation.sweep import (
    ALGORITHM_LABELS,
    write_sweep_csv,
)
from src.models.communication import transmission_energy, transmission_latency, uplink_rate
from src.models.computation import computation_energy, computation_latency, processed_data_size
from src.models.constraints import total_latency
from src.models.sensing_quality import raw_data_volume, sensing_energy
from src.rl.checkpoint_selection import SELECTED_VECNORMALIZE, selected_model_name
from src.rl.eval import episode_metrics_from_info
from src.rl.train import ALGORITHM_OPTION_KEYS, load_training_config
from src.sim.env import RSULeaderEnv
from src.sim.scenario import ScenarioConfig
from src.sim.state import LeaderAction

PRICE_METRICS = ("raw_rsu_reward", "task_success_rate", "price_cap_fraction")
COMPRESSION_METRICS = (
    "sensing_energy_j",
    "computation_energy_j",
    "transmission_energy_j",
    "total_energy_j",
    "counterfactual_service_feasible",
)


def _headline_expected_configs(cfg: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    validation_seeds = tuple(cfg["seeds"]["validation"])
    return {
        (spec.algorithm, spec.train_seed): orch.materialize_config(
            spec.to_worker_dict(), validation_seeds
        )
        for spec in headline_specs(cfg)
        if spec.algorithm != "dbas"
    }


def _algorithm_seeds(algorithm: str, cfg: dict[str, Any]) -> tuple[int, ...]:
    if algorithm == "dbas":
        return (int(cfg["seeds"]["headline_train"][0]),)
    return train_seeds_for(algorithm, config=cfg)


def _write_experiment(
    target: Path,
    *,
    rows: list[dict[str, object]],
    summary: list[dict[str, object]],
    seed_summary: list[dict[str, object]],
    contract: dict[str, Any],
) -> None:
    target.mkdir(parents=True, exist_ok=True)
    write_sweep_csv(target / "episodes.csv", rows)
    write_sweep_csv(target / "seed_summary.csv", seed_summary)
    write_sweep_csv(target / "summary.csv", summary)
    write_evaluation_contract(target, contract)


def _load_proposed_policy(
    run_dir: Path,
    expected_config: dict[str, Any],
):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize

    from src.rl.ppo import NormalizationConfig, RewardConfig, make_vectorized_eval_env

    provenance = validate_run_artifact(run_dir, expected_config=expected_config)
    config = load_training_config(run_dir / "config_snapshot.yaml")
    scenario_config = ScenarioConfig(**config.get("scenario", {}))
    reward_config = RewardConfig(**config.get("reward", {}))
    normalization_config = NormalizationConfig(**config.get("normalization", {}))
    action_logit_bound = float(config.get("algorithm", {}).get("action_logit_bound", 8.0))
    env = make_vectorized_eval_env(
        scenario_config=scenario_config,
        reward_config=reward_config,
        normalization_config=normalization_config,
        action_logit_bound=action_logit_bound,
        algorithm_name="proposed",
        algorithm_options={
            key: value
            for key, value in config.get("algorithm", {}).items()
            if key not in ALGORITHM_OPTION_KEYS
        },
    )
    vec_path = run_dir / SELECTED_VECNORMALIZE
    if isinstance(env, VecNormalize):
        if not vec_path.exists():
            env.close()
            raise FileNotFoundError(f"selected normalization state missing under {run_dir}")
        base_env = env.venv
        env = VecNormalize.load(str(vec_path), base_env)
        env.training = False
        env.norm_reward = False
    model = PPO.load(
        str(run_dir / selected_model_name("proposed")),
        env=env,
        device=str(config.get("algorithm", {}).get("device", "cpu")),
    )
    return model, env, scenario_config, provenance.as_dict()


def _scaled_prices(prices: np.ndarray, theta: float, cap: float) -> tuple[tuple[float, ...], float]:
    targets = np.asarray(prices, dtype=np.float64) * float(theta)
    clipped = np.clip(targets, 0.0, cap)
    fraction = float(np.mean(np.abs(clipped - targets) > 1e-12))
    return tuple(float(value) for value in clipped), fraction


def _compression_rows_from_info(
    info: dict[str, Any],
    *,
    train_seed: int,
    eval_seed: int,
    eta_grid: tuple[float, ...],
    scenario_config: ScenarioConfig,
    provenance: dict[str, Any],
) -> list[dict[str, object]]:
    snapshot = info["observation_snapshot"]
    selected_tasks = np.asarray(info["selected_tasks"], dtype=np.int64)
    sensing_times = np.asarray(info["follower_sensing_durations_s"], dtype=np.float64)
    selected_eta = np.asarray(info["follower_compression_ratios"], dtype=np.float64)
    bandwidth = np.asarray(info["leader_action"]["bandwidth"], dtype=np.float64)
    channel_gains = np.asarray(snapshot["channels"], dtype=np.float64)
    cpu = np.asarray(snapshot["vehicles"]["cpu_cycles_per_s"], dtype=np.float64)
    sensor_power = np.asarray(snapshot["vehicles"]["sensor_power_w"], dtype=np.float64)
    deadlines = np.asarray(snapshot["tasks"]["tau_max_s"], dtype=np.float64)
    rows: list[dict[str, object]] = []
    for vehicle_index, task_id in enumerate(selected_tasks):
        tau_s = float(sensing_times[vehicle_index])
        weight = float(bandwidth[vehicle_index])
        if int(task_id) < 0 or tau_s <= 0.0 or weight <= 0.0:
            continue
        rate = uplink_rate(
            bandwidth_fraction=weight,
            total_bandwidth_hz=scenario_config.system_bandwidth_hz,
            tx_power_w=scenario_config.tx_power_w,
            channel_gain_value=float(channel_gains[vehicle_index]),
            noise_power_w=scenario_config.noise_power_w,
        )
        raw_bits = raw_data_volume(
            raw_data_rate_bps=scenario_config.raw_data_rate_bps,
            sensing_duration_s=tau_s,
        )
        sense_j = sensing_energy(float(sensor_power[vehicle_index]), tau_s)
        for eta in eta_grid:
            comp_j = computation_energy(
                raw_data_rate_bps=scenario_config.raw_data_rate_bps,
                sensing_duration_s=tau_s,
                compression_ratio=eta,
                baseline_complexity_cycles_per_bit=scenario_config.baseline_complexity_cycles_per_bit,
                scaling_complexity_cycles_per_bit=scenario_config.scaling_complexity_cycles_per_bit,
                complexity_exponent=scenario_config.complexity_exponent,
                switched_capacitance_coeff=scenario_config.switched_capacitance_coeff,
                cpu_cycles_per_s=float(cpu[vehicle_index]),
            )
            comp_s = computation_latency(
                raw_data_rate_bps=scenario_config.raw_data_rate_bps,
                sensing_duration_s=tau_s,
                compression_ratio=eta,
                baseline_complexity_cycles_per_bit=scenario_config.baseline_complexity_cycles_per_bit,
                scaling_complexity_cycles_per_bit=scenario_config.scaling_complexity_cycles_per_bit,
                complexity_exponent=scenario_config.complexity_exponent,
                cpu_cycles_per_s=float(cpu[vehicle_index]),
            )
            tx_s = transmission_latency(processed_data_size(raw_bits, eta), rate)
            tx_j = transmission_energy(scenario_config.tx_power_w, tx_s)
            latency = total_latency(tau_s, comp_s, tx_s)
            row: dict[str, object] = {
                "algorithm": "proposed",
                "algorithm_label": ALGORITHM_LABELS["proposed"],
                "train_seed": train_seed,
                "eval_seed": eval_seed,
                "vehicle_index": vehicle_index,
                "selected_task_id": int(task_id),
                "profile_id": f"{train_seed}:{eval_seed}:{vehicle_index}",
                "sweep_param": "compression_ratio",
                "sweep_value": eta,
                "compression_ratio": eta,
                "selected_compression_ratio": float(selected_eta[vehicle_index]),
                "fixed_sensing_duration_s": tau_s,
                "fixed_bandwidth_fraction": weight,
                "uplink_rate_bps": rate,
                "sensing_energy_j": sense_j,
                "computation_energy_j": comp_j,
                "transmission_energy_j": tx_j,
                "total_energy_j": sense_j + comp_j + tx_j,
                "computation_latency_s": comp_s,
                "transmission_latency_s": tx_s,
                "total_latency_s": latency,
                "deadline_s": float(deadlines[int(task_id)]),
                "counterfactual_service_feasible": int(
                    latency <= float(deadlines[int(task_id)]) + scenario_config.bisection_tol
                ),
                "analysis_kind": "component_counterfactual_not_policy_evaluation",
                "exactness_label": "paper-exact energy equations",
            }
            row.update(provenance)
            rows.append(row)
    return rows


def run_mechanism_analysis(cfg: dict[str, Any]) -> None:
    analysis = cfg["main_text_analysis"]
    seed_bank = str(analysis["evaluation_seed_bank"])
    eval_seeds = tuple(cfg["seeds"][seed_bank])
    theta_grid = tuple(analysis["price_scale_grid"])
    eta_grid = tuple(analysis["compression_ratio_grid"])
    expected = _headline_expected_configs(cfg)
    price_rows: list[dict[str, object]] = []
    compression_rows: list[dict[str, object]] = []
    max_theta_one_gap = 0.0
    for train_seed in tuple(cfg["seeds"]["headline_train"]):
        run_dir = headline_run_dir("proposed", train_seed, cfg)
        print(f"evaluating main-text mechanisms from {run_dir.name}")
        model, policy_env, scenario_config, provenance = _load_proposed_policy(
            run_dir, expected[("proposed", train_seed)]
        )
        raw_env = RSULeaderEnv(scenario_config=scenario_config)
        try:
            for episode_index, eval_seed in enumerate(eval_seeds):
                policy_env.seed(int(eval_seed))
                observation = policy_env.reset()
                action, _ = model.predict(observation, deterministic=True)
                _, reward_vector, done, infos = policy_env.step(action)
                if not bool(done[0]):
                    raise RuntimeError("main-text mechanism evaluation expects one-step episodes")
                info = dict(infos[0])
                learned = info["leader_action"]
                learned_prices = np.asarray(learned["prices"], dtype=np.float64)
                learned_bandwidth = tuple(float(value) for value in learned["bandwidth"])
                original_raw_reward = float(info.get("raw_rsu_reward", reward_vector[0]))
                compression_rows.extend(
                    _compression_rows_from_info(
                        info,
                        train_seed=train_seed,
                        eval_seed=int(eval_seed),
                        eta_grid=eta_grid,
                        scenario_config=scenario_config,
                        provenance=provenance,
                    )
                )
                for theta in theta_grid:
                    raw_env.reset(seed=int(eval_seed))
                    prices, cap_fraction = _scaled_prices(
                        learned_prices, theta, scenario_config.pricing_budget_max
                    )
                    scaled_action = LeaderAction(prices=prices, bandwidth=learned_bandwidth)
                    _, reward, _terminated, _truncated, scaled_info = raw_env.step(scaled_action)
                    row: dict[str, object] = {
                        "episode_index": episode_index,
                        "eval_seed": int(eval_seed),
                        "algorithm": "proposed",
                        "algorithm_label": ALGORITHM_LABELS["proposed"],
                        "train_seed": train_seed,
                        "sweep_param": "price_scale",
                        "sweep_value": theta,
                        "price_scale": theta,
                        "price_cap_fraction": cap_fraction,
                        "theta_one_reference_raw_rsu_reward": original_raw_reward,
                        "analysis_kind": "fixed_bandwidth_price_counterfactual",
                    }
                    row.update(episode_metrics_from_info(dict(scaled_info), episode_reward=float(reward)))
                    row.update(provenance)
                    price_rows.append(row)
                    if theta == 1.0:
                        max_theta_one_gap = max(
                            max_theta_one_gap, abs(float(reward) - original_raw_reward)
                        )
        finally:
            policy_env.close()
            raw_env.close()
    if max_theta_one_gap > 1e-6:
        raise ValueError(
            f"theta=1 failed to reproduce raw RSU utility; max gap={max_theta_one_gap:g}"
        )
    if not compression_rows:
        raise ValueError("no TE-PPO participating-vehicle profiles were available")

    common_contract = {
        "revision_id": cfg["revision_id"],
        "revision_schema_version": int(cfg["schema_version"]),
        "local_contract_version": int(analysis["contract_version"]),
        "evaluation_seed_bank": seed_bank,
        "evaluation_seeds": list(eval_seeds),
        "training_seeds": list(cfg["seeds"]["headline_train"]),
        "algorithm": "proposed",
        "aggregation_unit": "training_seed_mean_over_evaluation_scenarios",
    }
    price_seed_summary = summarize_by_seed(
        price_rows,
        group_fields=("algorithm", "sweep_value"),
        metrics=PRICE_METRICS,
    )
    price_summary = summarize_across_training_seeds(
        price_seed_summary,
        group_fields=("algorithm", "sweep_value"),
        metrics=PRICE_METRICS,
    )
    compression_seed_summary = summarize_by_seed(
        compression_rows,
        group_fields=("algorithm", "sweep_value"),
        metrics=COMPRESSION_METRICS,
    )
    compression_summary = summarize_across_training_seeds(
        compression_seed_summary,
        group_fields=("algorithm", "sweep_value"),
        metrics=COMPRESSION_METRICS,
    )
    root = revision_roots(cfg)[1] / "main_text_analysis"
    _write_experiment(
        root / "pricing",
        rows=price_rows,
        summary=price_summary,
        seed_summary=price_seed_summary,
        contract={
            **common_contract,
            "experiment": "main_text_pricing_sensitivity",
            "price_scale_grid": list(theta_grid),
            "price_transform": "clip(theta * learned_price, 0, pricing_budget_max)",
            "fixed_quantity": "learned_bandwidth_and_sampled_scenario",
            "theta_one_max_abs_rsu_utility_gap": max_theta_one_gap,
        },
    )
    _write_experiment(
        root / "compression",
        rows=compression_rows,
        summary=compression_summary,
        seed_summary=compression_seed_summary,
        contract={
            **common_contract,
            "experiment": "main_text_compression_energy",
            "compression_ratio_grid": list(eta_grid),
            "analysis_semantics": "selected_vehicle_component_counterfactual",
            "fixed_quantities": [
                "sampled_vehicle_and_channel_state",
                "learned_bandwidth",
                "selected_sensing_duration",
            ],
            "deadline_policy": "retain_and_label_infeasible_counterfactual_points",
        },
    )


def _missing_headline_artifacts(cfg: dict[str, Any]) -> list[Path]:
    algorithms = tuple(cfg["main_text_analysis"]["algorithms"])
    return [
        headline_run_dir(algorithm, seed, cfg)
        for algorithm in algorithms
        if algorithm != "dbas"
        for seed in _algorithm_seeds(algorithm, cfg)
        if not (headline_run_dir(algorithm, seed, cfg) / "summary.json").exists()
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "scope", nargs="?", choices=("all", "mechanisms"), default="all"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_revision_config()
    missing = _missing_headline_artifacts(cfg)
    if args.dry_run:
        analysis = cfg["main_text_analysis"]
        print(
            f"main-text contract v{analysis['contract_version']}; "
            f"CPU cells={len(analysis['cpu_capacity_ranges_cycles_per_s'])}; "
            f"price cells={len(analysis['price_scale_grid'])}; "
            f"compression cells={len(analysis['compression_ratio_grid'])}; "
            f"evaluation seeds={len(cfg['seeds'][analysis['evaluation_seed_bank']])}; "
            f"missing headline artifacts={len(missing)}"
        )
        return 0
    if missing:
        print("error: formal headline artifacts are missing", file=sys.stderr)
        for path in missing[:10]:
            print(f"  {path}", file=sys.stderr)
        return 1
    try:
        if args.scope in ("all", "mechanisms"):
            run_mechanism_analysis(cfg)
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote formal main-text analysis under {revision_roots(cfg)[1] / 'main_text_analysis'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
