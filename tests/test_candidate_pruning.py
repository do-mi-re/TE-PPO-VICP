"""Candidate admission, solver filtering, and bandwidth-mask checks."""

import numpy as np
import pytest

from src.rl.joint_action import CONSTRAINT_AWARE_JOINT_ACTION_PARAMETERIZATION, decode_joint_action
from src.sim.env import RSULeaderEnv
from src.sim.pruning import candidate_masks
from src.sim.scenario import ScenarioConfig, sample_scenario


def _config(**overrides) -> ScenarioConfig:
    base = {
        "num_vehicles": 20,
        "spatial_potential_max": 50.0,
        "pricing_budget_max": 0.3,
        # Use the rectangular-RoI distance model from the experiment contract.
        "roi_distance_model": "rectangle",
    }
    base.update(overrides)
    return ScenarioConfig(**base)


def _step(config: ScenarioConfig, seed: int):
    env = RSULeaderEnv(scenario_config=config)
    env.reset(seed=seed)
    action = {
        "prices": np.full(config.num_tasks, 0.04),
        "bandwidth": np.full(config.num_vehicles, 1.0 / config.num_vehicles),
    }
    _, reward, _, _, info = env.step(action)
    return reward, info


@pytest.mark.parametrize("num_vehicles", [10, 20, 40])
@pytest.mark.parametrize("seed", [40_000, 40_001])
def test_disabled_pruning_leaves_the_step_bit_identical(num_vehicles, seed):
    default_reward, default_info = _step(_config(num_vehicles=num_vehicles), seed)
    explicit_reward, explicit_info = _step(
        _config(num_vehicles=num_vehicles, enable_candidate_pruning=False), seed
    )
    assert default_reward == explicit_reward
    assert default_info["social_welfare"] == explicit_info["social_welfare"]


@pytest.mark.parametrize("num_vehicles", [10, 20, 40])
def test_pruning_caps_participants_per_task(num_vehicles):
    """With pruning on, no task may exceed its own N_max^(k)."""
    config = _config(num_vehicles=num_vehicles, enable_candidate_pruning=True)
    _, info = _step(config, seed=40_000)

    state = sample_scenario(np.random.default_rng(40_000), config)
    selected = np.asarray(info["selected_tasks"])
    for task in state.tasks:
        chosen = int((selected == task.task_id).sum())
        assert chosen <= task.candidate_limit, (
            f"task {task.task_id} admitted {chosen} vehicles, "
            f"N_max = {task.candidate_limit}"
        )


def test_admission_set_uses_bounded_potential_with_raw_saturation_ties():
    """Ranking follows bounded A, then resolves saturation ties.

    spatial_potential_max saturates a large share of vehicles at the cap. The
    raw secondary key prevents those paper-defined bounded ties from
    degenerating into an arbitrary vehicle-index ordering.
    """
    from src.models.sensing_quality import spatial_potential

    config = _config(enable_candidate_pruning=True)
    state = sample_scenario(np.random.default_rng(40_000), config)
    masks = candidate_masks(state, config)

    for task in state.tasks:
        uncapped = [
            spatial_potential(
                resolution_coeff=vehicle.resolution_coeff,
                occlusion_factor=state.occlusion_factors[idx][task.task_id],
                distance_to_roi_m=state.distance_to_roi_m[idx][task.task_id],
                sensing_range_m=config.sensing_range_m,
                sensing_decay_exponent=config.sensing_decay_exponent,
                max_value=None,
            )
            for idx, vehicle in enumerate(state.vehicles)
        ]
        bounded = [min(value, config.spatial_potential_max) for value in uncapped]
        expected = {
            idx for idx in sorted(
                range(len(uncapped)),
                key=lambda i: (-bounded[i], -uncapped[i], i),
            )[: task.candidate_limit]
        }
        actual = {idx for idx, tasks in enumerate(masks) if task.task_id in tasks}
        assert actual == expected


def test_bounded_ranking_with_raw_ties_matches_monotone_order():
    """Monotone clipping with raw tie-breaking preserves raw-potential ordering."""
    from src.models.sensing_quality import spatial_potential

    config = _config(enable_candidate_pruning=True)
    state = sample_scenario(np.random.default_rng(40_001), config)
    masks = candidate_masks(state, config)
    for task in state.tasks:
        raw = [
            spatial_potential(
                resolution_coeff=vehicle.resolution_coeff,
                occlusion_factor=state.occlusion_factors[idx][task.task_id],
                distance_to_roi_m=state.distance_to_roi_m[idx][task.task_id],
                sensing_range_m=config.sensing_range_m,
                sensing_decay_exponent=config.sensing_decay_exponent,
                max_value=None,
            )
            for idx, vehicle in enumerate(state.vehicles)
        ]
        raw_order = set(sorted(range(len(raw)), key=lambda i: (-raw[i], i))[
            :task.candidate_limit
        ])
        actual = {idx for idx, tasks in enumerate(masks) if task.task_id in tasks}
        assert actual == raw_order


def test_pruning_without_saturation_cap_uses_raw_potential():
    """The generic ``None`` cap means uncapped ranking, not an invalid config."""
    from src.models.sensing_quality import spatial_potential

    config = ScenarioConfig(enable_candidate_pruning=True)
    state = sample_scenario(np.random.default_rng(40_000), config)
    masks = candidate_masks(state, config)

    for task in state.tasks:
        raw = [
            spatial_potential(
                resolution_coeff=vehicle.resolution_coeff,
                occlusion_factor=state.occlusion_factors[idx][task.task_id],
                distance_to_roi_m=state.distance_to_roi_m[idx][task.task_id],
                sensing_range_m=config.sensing_range_m,
                sensing_decay_exponent=config.sensing_decay_exponent,
                max_value=None,
            )
            for idx, vehicle in enumerate(state.vehicles)
        ]
        expected = set(sorted(range(len(raw)), key=lambda i: (-raw[i], i))[
            :task.candidate_limit
        ])
        actual = {idx for idx, tasks in enumerate(masks) if task.task_id in tasks}
        assert actual == expected


def test_pruned_pairs_are_omitted_before_follower_optimization():
    """Only admitted pairs should reach the expensive task solver."""
    config = _config(enable_candidate_pruning=True)
    env = RSULeaderEnv(scenario_config=config)
    env.reset(seed=40_000)
    action = {
        "prices": np.full(config.num_tasks, 0.04),
        "bandwidth": np.full(config.num_vehicles, 1.0 / config.num_vehicles),
    }
    _, reward, _, _, info = env.step(action)

    assert np.isfinite(reward)
    admitted = int(info["candidate_pruning"]["admitted_pair_count"])
    evaluated = sum(
        len(response.task_evaluations) for response in info["follower_responses"]
    )
    assert evaluated == admitted
    assert admitted < config.num_vehicles * config.num_tasks
    assert not info["violated_constraints"]["selected_latency"]


def test_public_candidate_masks_match_solver_admission_and_are_copies():
    config = _config(enable_candidate_pruning=True)
    env = RSULeaderEnv(scenario_config=config)
    env.reset(seed=40_000)

    pair_mask = env.candidate_pair_mask()
    vehicle_mask = env.candidate_vehicle_mask()

    assert pair_mask.shape == (config.num_vehicles, config.num_tasks)
    np.testing.assert_array_equal(vehicle_mask, np.any(pair_mask, axis=1))
    assert int(np.sum(pair_mask)) == sum(len(tasks) for tasks in env._candidate_masks)
    pair_mask[:] = False
    assert np.any(env.candidate_pair_mask())


def test_latent_action_allocates_no_bandwidth_to_non_candidates():
    from src.rl.ppo import LeaderLatentActionWrapper

    config = _config(enable_candidate_pruning=True)
    base_env = RSULeaderEnv(scenario_config=config)
    env = LeaderLatentActionWrapper(
        base_env,
        num_tasks=config.num_tasks,
        num_vehicles=config.num_vehicles,
        pricing_budget_max=config.pricing_budget_max,
    )
    env.reset(seed=40_000)
    latent = np.linspace(-1.0, 1.0, env.action_space.shape[0])

    _, _, _, _, info = env.step(latent)

    eligible = base_env.candidate_vehicle_mask()
    bandwidth = np.asarray(info["leader_action"]["bandwidth"])
    assert np.all(bandwidth[~eligible] == 0.0)
    diagnostics = info["bandwidth_allocation"]
    assert diagnostics["nonadmitted"] == 0.0
    assert diagnostics["nonadmitted_ratio"] == 0.0
    assert diagnostics["total"] == pytest.approx(diagnostics["admitted"])


def test_joint_policy_path_enforces_pruning_for_controlled_ablation():
    config = _config(enable_candidate_pruning=True)
    env = RSULeaderEnv(scenario_config=config)
    env.reset(seed=40_000)
    joint = decode_joint_action(
        leader_latent=np.zeros(env.scenario_config.num_tasks + 1 + env.scenario_config.num_vehicles),
        vehicle_cont=np.zeros((env.scenario_config.num_vehicles, 2)),
        assignment=np.zeros(env.scenario_config.num_vehicles, dtype=int),
        env=env, parameterization=CONSTRAINT_AWARE_JOINT_ACTION_PARAMETERIZATION,
    )

    _, _, _, _, info = env.step_joint(joint)

    admitted = int(info["candidate_pruning"]["admitted_pair_count"])
    evaluated = sum(
        len(response.task_evaluations) for response in info["follower_responses"]
    )
    assert evaluated == admitted


def test_distance_and_occlusion_prefilters_shrink_the_admission_set():
    config = _config(enable_candidate_pruning=True)
    state = sample_scenario(np.random.default_rng(40_000), config)
    baseline = sum(len(tasks) for tasks in candidate_masks(state, config))

    filtered = candidate_masks(
        state,
        _config(
            enable_candidate_pruning=True,
            candidate_pruning_radius_m=30.0,
            pruning_occlusion_max=0.3,
        ),
    )
    assert sum(len(tasks) for tasks in filtered) <= baseline


def test_candidate_limit_range_decouples_the_candidate_cap():
    """candidate_limit_range_m must move N_max without touching quality.

    r_0 normalizes (d/r_0)^-zeta AND sets the candidate cap, and the two pull
    in opposite directions. The override exists so a sensitivity sweep can
    separate them; it must therefore change candidate_limit and leave the
    solver's sensing_range_m alone.
    """
    from src.sim.scenario import make_solver_params

    coupled = _config()
    # Eight metres yields multiple 2*r0 coverage spans and therefore exercises
    # the candidate-cap role without changing sensing-quality normalization.
    decoupled = _config(candidate_limit_range_m=8.0)

    base_limits = [t.candidate_limit for t in
                   sample_scenario(np.random.default_rng(7), coupled).tasks]
    moved_limits = [t.candidate_limit for t in
                    sample_scenario(np.random.default_rng(7), decoupled).tasks]

    assert moved_limits != base_limits
    # Quality normalization is untouched.
    assert make_solver_params(decoupled).sensing_range_m == coupled.sensing_range_m


def test_pruning_flag_survives_the_training_config_round_trip():
    """The flag must reach the env through build_sweep_config's snapshot.

    Training writes config_snapshot.yaml and evaluation rebuilds ScenarioConfig
    from it, so a flag that is dropped anywhere along that path would train a
    pruned policy and then score it unpruned (or the reverse) without failing.
    """
    from src.evaluation.sweep import (
        EVAL_SEEDS,
        build_sweep_config,
        load_algorithm_config,
    )

    for enabled in (False, True):
        config = build_sweep_config(
            "proposed",
            base_config=load_algorithm_config("proposed"),
            scenario_overrides={
                "num_vehicles": 20,
                "enable_candidate_pruning": enabled,
            },
            train_seed=123,
            eval_seeds=EVAL_SEEDS,
            run_name_suffix="prune" if enabled else "noprune",
        )
        scenario = ScenarioConfig(**config["scenario"])
        assert scenario.enable_candidate_pruning is enabled

        env = RSULeaderEnv(scenario_config=scenario)
        env.reset(seed=40_000)
        masks_built = env._candidate_masks is not None
        assert masks_built is enabled


def test_pruning_reduces_participation_at_a_fixed_leader_action():
    """Sanity-check the mechanism end to end at a realistic price.

    Not a claim about trained performance -- a trained leader suppresses
    redundant participation through pricing too -- just confirmation that the
    admission stage actually binds when several vehicles would otherwise crowd
    the same task.
    """
    baseline = _step(_config(num_vehicles=40), seed=40_000)[1]
    pruned = _step(
        _config(num_vehicles=40, enable_candidate_pruning=True), seed=40_000
    )[1]

    selected_baseline = int((np.asarray(baseline["selected_tasks"]) >= 0).sum())
    selected_pruned = int((np.asarray(pruned["selected_tasks"]) >= 0).sum())
    assert selected_pruned < selected_baseline


@pytest.mark.parametrize("num_vehicles", [10, 20, 40, 60, 100])
def test_admitted_pairs_are_bounded_independently_of_density(num_vehicles):
    """Admission is capped by the tasks, not by the fleet size.

    Sum_k N_max^(k) bounds the admitted (vehicle, task) pairs regardless of V,
    so the follower solve stops scaling with vehicle count once pruning is on.
    This is the basis for the scalability argument in the paper: the leader's
    per-cycle work becomes O(sum_k N_max^(k)) rather than O(V*K).
    """
    config = _config(num_vehicles=num_vehicles, enable_candidate_pruning=True)
    state = sample_scenario(np.random.default_rng(0), config)
    masks = candidate_masks(state, config)

    admitted = sum(len(tasks) for tasks in masks)
    cap = sum(task.candidate_limit for task in state.tasks)
    assert admitted <= cap
    # The cap does not grow with the fleet, so the share pruned rises with V.
    assert cap < num_vehicles * config.num_tasks or num_vehicles <= 2


def test_pruning_share_grows_with_vehicle_density():
    """A denser fleet is pruned harder, which is why a V=20 result about
    pruning must not be extrapolated to V=100 without measuring it."""
    shares = []
    for num_vehicles in (10, 40, 100):
        config = _config(num_vehicles=num_vehicles, enable_candidate_pruning=True)
        state = sample_scenario(np.random.default_rng(0), config)
        admitted = sum(len(t) for t in candidate_masks(state, config))
        shares.append(1.0 - admitted / (num_vehicles * config.num_tasks))

    assert shares == sorted(shares), "pruning share should be monotone in V"
    assert shares[-1] > shares[0]


@pytest.mark.parametrize(
    ("rho", "expected"),
    [(1.0, 2), (2.0, 2), (3.0, 3), (4.0, 4)],
)
def test_segment_first_candidate_limit_activates_diversity_factor(rho, expected):
    from src.sim.scenario import candidate_limit

    assert candidate_limit(35.0, 80.0, rho) == expected


def test_candidate_limit_counts_long_roi_coverage_spans_before_diversity():
    from src.sim.scenario import candidate_limit

    # ceil(350 / 160) = 3 spans; rho=2 therefore retains six candidates.
    assert candidate_limit(350.0, 80.0, 2.0) == 6


@pytest.mark.parametrize("field", ["task_length_m", "sensing_range_m", "diversity_factor"])
def test_candidate_limit_rejects_non_positive_inputs(field):
    from src.sim.scenario import candidate_limit

    values = {"task_length_m": 35.0, "sensing_range_m": 80.0, "diversity_factor": 2.0}
    values[field] = 0.0
    with pytest.raises(ValueError, match="positive"):
        candidate_limit(**values)
