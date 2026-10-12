"""Regression tests for MAPPO's implementation-only batching layer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

import src.sim.env as env_module
from src.rl.joint_action import CONSTRAINT_AWARE_JOINT_ACTION_PARAMETERIZATION, decode_joint_action
from src.rl.mappo import (
    GaussianActor,
    MAPPOLearnedPolicy,
    VehicleHybridActor,
    _vehicle_rows_by_step,
    decode_leader_latent,
    flatten_global_observation,
    flatten_vehicle_observation,
    train_from_config,
    vehicle_observation_matrix,
)
from src.sim.env import RSULeaderEnv
from src.sim.scenario import ScenarioConfig


def _env(num_vehicles: int = 4) -> RSULeaderEnv:
    return RSULeaderEnv(
        scenario_config=ScenarioConfig(
            num_vehicles=num_vehicles,
            pricing_budget_max=0.3,
        )
    )


def test_vehicle_observation_matrix_matches_scalar_contract() -> None:
    env = _env()
    env.reset(seed=1201)
    prices = np.asarray([0.05, 0.10, 0.15])
    bandwidth = np.asarray([0.1, 0.2, 0.3, 0.4])

    expected = np.stack(
        [
            flatten_vehicle_observation(
                env.vehicle_observation(vehicle_idx, prices, bandwidth)
            )
            for vehicle_idx in range(4)
        ]
    )

    np.testing.assert_array_equal(
        vehicle_observation_matrix(env, prices, bandwidth), expected
    )
    env.close()


def test_joint_step_scores_only_the_assigned_task(monkeypatch) -> None:
    env = _env(num_vehicles=4)
    env.reset(seed=1202)
    real_evaluator = env_module.evaluate_assigned_task
    calls = 0

    def counted_evaluator(**kwargs):
        nonlocal calls
        calls += 1
        return real_evaluator(**kwargs)

    monkeypatch.setattr(env_module, "evaluate_assigned_task", counted_evaluator)
    joint = decode_joint_action(
        leader_latent=np.zeros(env.scenario_config.num_tasks + 1 + env.scenario_config.num_vehicles),
        vehicle_cont=np.zeros((env.scenario_config.num_vehicles, 2)),
        assignment=np.zeros(env.scenario_config.num_vehicles, dtype=int),
        env=env, parameterization=CONSTRAINT_AWARE_JOINT_ACTION_PARAMETERIZATION,
    )
    _, _, _, _, info = env.step_joint(joint)

    assert calls == env.scenario_config.num_vehicles
    assert all(
        len(response.task_evaluations) == env.scenario_config.num_tasks
        for response in info["follower_responses"]
    )
    env.close()


def test_batched_vehicle_log_prob_matches_scalar_evaluation() -> None:
    torch.manual_seed(1202)
    actor = VehicleHybridActor(obs_dim=11, num_tasks=3, hidden_sizes=(16, 16))
    observations = torch.randn(7, 11)
    continuous_actions = torch.randn(7, 2)
    categorical_actions = torch.tensor([0, 1, 2, 0, 2, 1, 0])

    batched_logp, batched_mean = actor.evaluate_actions(
        observations, continuous_actions, categorical_actions
    )
    scalar_results = [
        actor.evaluate_actions(
            observations[index : index + 1],
            continuous_actions[index : index + 1],
            categorical_actions[index : index + 1],
        )
        for index in range(observations.shape[0])
    ]
    scalar_logp = torch.cat([result[0] for result in scalar_results])
    scalar_mean = torch.cat([result[1] for result in scalar_results])

    torch.testing.assert_close(batched_logp, scalar_logp, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(batched_mean, scalar_mean, rtol=1e-6, atol=1e-6)


def test_batched_vehicle_entropy_matches_scalar_evaluation() -> None:
    torch.manual_seed(1203)
    actor = VehicleHybridActor(obs_dim=11, num_tasks=3, hidden_sizes=(16, 16))
    observations = torch.randn(7, 11)
    continuous_actions = torch.randn(7, 2)
    categorical_actions = torch.tensor([0, 1, 2, 0, 2, 1, 0])

    batched = actor.evaluate_actions_with_entropy(
        observations, continuous_actions, categorical_actions
    )
    scalar = [
        actor.evaluate_actions_with_entropy(
            observations[index : index + 1],
            continuous_actions[index : index + 1],
            categorical_actions[index : index + 1],
        )
        for index in range(observations.shape[0])
    ]

    for position in (0, 1, 2, 3):
        torch.testing.assert_close(
            batched[position],
            torch.cat([result[position] for result in scalar]),
            rtol=1e-6,
            atol=1e-6,
        )


def test_vehicle_actor_candidate_mask_applies_to_sample_and_log_prob() -> None:
    torch.manual_seed(1204)
    actor = VehicleHybridActor(obs_dim=11, num_tasks=3, hidden_sizes=(16, 16))
    observations = torch.randn(4, 11)
    mask = torch.zeros((4, 3), dtype=torch.bool)
    mask[:, 2] = True
    mask[3, :] = False

    continuous, categorical, sampled_logp = actor.sample(
        observations, assignment_mask=mask
    )
    assert torch.all(categorical[:3] == 2)
    assert int(categorical[3]) == 0
    recomputed = actor.log_prob(
        observations,
        continuous,
        categorical,
        assignment_mask=mask,
    )
    torch.testing.assert_close(sampled_logp, recomputed, rtol=1e-6, atol=1e-6)


def test_vehicle_row_mapping_preserves_step_major_order() -> None:
    lookup = _vehicle_rows_by_step(num_steps=3, num_vehicles=4)
    rows = lookup[torch.tensor([2, 0])].reshape(-1)
    torch.testing.assert_close(rows, torch.tensor([8, 9, 10, 11, 0, 1, 2, 3]))


def test_mappo_applies_and_records_entropy_configuration(tmp_path: Path) -> None:
    config = yaml.safe_load(Path("configs/algorithms/mappo.yaml").read_text())
    config["scenario"]["num_vehicles"] = 4
    config["algorithm"].update({
        "total_timesteps": 2,
        "rollout_steps": 2,
        "collection_microbatch_size": 1,
        "n_epochs": 1,
        "minibatch_size": 2,
        "hidden_sizes": [16, 16],
        "device": "cpu",
        "ent_coef": 0.01,
    })
    config["evaluation"] = {"interval_env_steps": 1000, "seed_list": [91203]}
    config["artifacts"]["run_name"] = "entropy_enabled"

    summary = train_from_config(config, output_root=tmp_path)

    assert summary["performance"]["ent_coef"] == 0.01
    assert np.isfinite(summary["history"][0]["entropy_loss"])
    assert summary["history"][0]["vehicle_assignment_entropy"] >= 0.0

    zero_entropy_config = yaml.safe_load(yaml.safe_dump(config))
    zero_entropy_config["algorithm"]["ent_coef"] = 0.0
    zero_entropy_config["artifacts"]["run_name"] = "entropy_disabled"
    zero_summary = train_from_config(zero_entropy_config, output_root=tmp_path)
    with_entropy = torch.load(
        summary["last_model_path"], map_location="cpu", weights_only=True
    )
    without_entropy = torch.load(
        zero_summary["last_model_path"], map_location="cpu", weights_only=True
    )
    assert not torch.equal(
        with_entropy["vehicle_actor"]["assign_logits.weight"],
        without_entropy["vehicle_actor"]["assign_logits.weight"],
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_batched_vehicle_actor_cuda_smoke() -> None:
    actor = VehicleHybridActor(obs_dim=11, num_tasks=3, hidden_sizes=(16, 16)).cuda()
    observations = torch.randn(32 * 20, 11, device="cuda")

    continuous, categorical, log_prob = actor.sample(observations)

    assert continuous.shape == (640, 2)
    assert categorical.shape == (640,)
    assert log_prob.shape == (640,)
    assert torch.isfinite(log_prob).all()


@torch.no_grad()
def test_batched_deterministic_policy_matches_scalar_reference() -> None:
    torch.manual_seed(1203)
    env = _env()
    observation, _ = env.reset(seed=1203)
    scenario = env.scenario_config
    global_dim = flatten_global_observation(observation).shape[0]
    zeros_p = np.zeros(scenario.num_tasks)
    zeros_w = np.zeros(scenario.num_vehicles)
    vehicle_dim = flatten_vehicle_observation(
        env.vehicle_observation(0, zeros_p, zeros_w)
    ).shape[0]
    leader = GaussianActor(
        global_dim,
        scenario.num_tasks + 1 + scenario.num_vehicles,
        (16, 16),
    )
    vehicle = VehicleHybridActor(vehicle_dim, scenario.num_tasks, (16, 16))
    policy = MAPPOLearnedPolicy(leader, vehicle, env, torch.device("cpu"))

    batched = policy.act(observation)

    global_t = torch.from_numpy(flatten_global_observation(observation)).unsqueeze(0)
    leader_latent = leader.distribution(global_t).mean.squeeze(0).numpy()
    prices, bandwidth = decode_leader_latent(
        leader_latent,
        num_tasks=scenario.num_tasks,
        pricing_budget_max=scenario.pricing_budget_max,
    )
    continuous = []
    assignment = []
    for vehicle_idx in range(scenario.num_vehicles):
        vehicle_obs = flatten_vehicle_observation(
            env.vehicle_observation(vehicle_idx, prices, bandwidth)
        )
        mean, logits = vehicle.action_parameters(torch.from_numpy(vehicle_obs).unsqueeze(0))
        continuous.append(mean.squeeze(0).numpy())
        assignment.append(int(torch.argmax(logits.squeeze(0)).item()))
    scalar = decode_joint_action(
        leader_latent=leader_latent, vehicle_cont=np.asarray(continuous),
        assignment=np.asarray(assignment), env=env,
        parameterization=CONSTRAINT_AWARE_JOINT_ACTION_PARAMETERIZATION,
    )

    np.testing.assert_allclose(batched.prices, scalar.prices, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(batched.bandwidth, scalar.bandwidth, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(batched.tau_s, scalar.tau_s, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(batched.eta, scalar.eta, rtol=1e-6, atol=1e-6)
    assert batched.assignment == scalar.assignment
    env.close()


def test_training_handles_partial_final_rollout_and_records_performance(tmp_path) -> None:
    config = {
        "seed": 1204,
        "scenario": {"num_vehicles": 3, "pricing_budget_max": 0.3},
        "algorithm": {
            "name": "mappo",
            "implementation_version": "mappo_constraint_aware_cold_start_v2",
            "total_timesteps": 19,
            "rollout_steps": 8,
            "collection_microbatch_size": 3,
            "n_epochs": 1,
            "minibatch_size": 8,
            "hidden_sizes": [16, 16],
            "device": "cpu",
        },
        "evaluation": {"interval_env_steps": 1000, "seed_list": [91204]},
        "artifacts": {"run_name": "mappo_partial_rollout"},
    }

    summary = train_from_config(config, output_root=tmp_path)

    assert summary["total_timesteps"] == 19
    assert [row["timesteps"] for row in summary["history"]] == [8.0, 16.0, 19.0]
    assert summary["implementation_version"] == "mappo_constraint_aware_cold_start_v2"
    assert summary["joint_action_parameterization"] == "candidate_masked_feasible_v1"
    performance = summary["performance"]
    assert performance["implementation_version"] == "mappo_constraint_aware_cold_start_v2"
    assert performance["joint_action_parameterization"] == "candidate_masked_feasible_v1"
    assert performance["rollout_samples_per_update"] == 8
    assert performance["collection_microbatch_size"] == 3
    assert performance["training_loop_wall_time_s"] > 0.0
    assert performance["collection_wall_time_s"] > 0.0
    assert performance["update_wall_time_s"] > 0.0
    assert performance["steps_per_second"] > 0.0

    snapshot = yaml.safe_load(
        (tmp_path / "mappo_partial_rollout" / "config_snapshot.yaml").read_text()
    )
    assert snapshot["algorithm"]["implementation_version"] == (
        "mappo_constraint_aware_cold_start_v2"
    )


def test_formal_mappo_config_declares_cold_start_implementation() -> None:
    config = yaml.safe_load(
        Path("configs/algorithms/mappo.yaml").read_text(encoding="utf-8")
    )
    assert config["algorithm"]["implementation_version"] == (
        "mappo_constraint_aware_cold_start_v2"
    )
    assert config["algorithm"]["joint_action_parameterization"] == (
        "candidate_masked_feasible_v1"
    )
    assert config["algorithm"]["rollout_steps"] == 1024
    assert config["algorithm"]["collection_microbatch_size"] == 32
    assert config["algorithm"]["ent_coef"] == 0.01
