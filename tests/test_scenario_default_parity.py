"""Optional scenario defaults preserve deterministic sampling and actions."""

import numpy as np

from src.sim.env import RSULeaderEnv
from src.sim.scenario import ScenarioConfig, sample_scenario


def _step(config: ScenarioConfig, seed: int):
    env = RSULeaderEnv(scenario_config=config)
    env.reset(seed=seed)
    action = {
        "prices": np.full(config.num_tasks, 0.04),
        "bandwidth": np.full(config.num_vehicles, 1.0 / config.num_vehicles),
    }
    _, reward, _, _, info = env.step(action)
    return reward, info


def test_explicit_defaults_match_omitting_the_field_entirely():
    """Passing a field its own default must equal not passing it at all.

    Guards the case where a field's declared default and its effective default
    diverge -- e.g. a None that is later replaced by something else downstream.
    """
    from dataclasses import asdict

    defaults = asdict(ScenarioConfig())
    neutral = {
        f: defaults[f]
        for f in ("enable_candidate_pruning", "candidate_pruning_radius_m",
                  "candidate_limit_range_m", "task_region_span_m",
                  "pruning_occlusion_max", "cpu_cycles_per_s_regimes")
        if f in defaults
    }
    assert neutral, "no optional fields are declared"

    base = ScenarioConfig(num_vehicles=20, spatial_potential_max=50.0)
    explicit = ScenarioConfig(num_vehicles=20, spatial_potential_max=50.0, **neutral)

    base_state = sample_scenario(np.random.default_rng(40_000), base)
    explicit_state = sample_scenario(np.random.default_rng(40_000), explicit)

    assert [t.roi_center_m for t in base_state.tasks] == [
        t.roi_center_m for t in explicit_state.tasks
    ]
    assert [t.candidate_limit for t in base_state.tasks] == [
        t.candidate_limit for t in explicit_state.tasks
    ]
    assert [v.position_m for v in base_state.vehicles] == [
        v.position_m for v in explicit_state.vehicles
    ]
    assert _step(base, 40_000)[0] == _step(explicit, 40_000)[0]


def test_pruning_defaults_leave_the_admission_stage_off():
    config = ScenarioConfig(num_vehicles=20, spatial_potential_max=50.0)
    assert config.enable_candidate_pruning is False
    assert config.pruning_occlusion_max is None
    assert config.candidate_pruning_radius_m is None
    assert config.candidate_limit_range_m is None
    assert config.task_region_span_m is None

    env = RSULeaderEnv(scenario_config=config)
    env.reset(seed=40_000)
    assert env._candidate_masks is None
