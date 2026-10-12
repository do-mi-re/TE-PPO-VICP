"""Episode-level CPU regime sampling for the revision robustness contract."""

from dataclasses import asdict

import numpy as np
import pytest

from src.evaluation.provenance import config_digest
from src.sim.scenario import ScenarioConfig, sample_scenario

RANGES = (
    (2.5e9, 7.5e9),
    (7.5e9, 12.5e9),
    (12.5e9, 17.5e9),
    (17.5e9, 22.5e9),
    (22.5e9, 27.5e9),
)


def test_default_sampler_is_bit_identical_when_optional_fields_are_explicit():
    implicit = ScenarioConfig(num_vehicles=20, spatial_potential_max=50.0)
    explicit = ScenarioConfig(
        num_vehicles=20,
        spatial_potential_max=50.0,
        cpu_cycles_per_s_regimes=None,
        cpu_regime_probabilities=None,
    )
    left = sample_scenario(np.random.default_rng(40_000), implicit)
    right = sample_scenario(np.random.default_rng(40_000), explicit)
    assert left == right


def test_regime_is_deterministic_and_shared_by_all_vehicles():
    config = ScenarioConfig(
        num_vehicles=30,
        cpu_cycles_per_s_regimes=RANGES,
        cpu_regime_probabilities=(0.2,) * 5,
    )
    first = sample_scenario(np.random.default_rng(123), config)
    second = sample_scenario(np.random.default_rng(123), config)
    assert first == second
    cpus = [vehicle.cpu_cycles_per_s for vehicle in first.vehicles]
    matching = [
        (low, high) for low, high in RANGES
        if all(low <= value <= high for value in cpus)
    ]
    assert len(matching) == 1


@pytest.mark.parametrize(
    "kwargs",
    (
        {"cpu_regime_probabilities": (1.0,)},
        {"cpu_cycles_per_s_regimes": ()},
        {"cpu_cycles_per_s_regimes": ((1.0, 1.0),)},
        {
            "cpu_cycles_per_s_regimes": ((1.0, 2.0), (2.0, 3.0)),
            "cpu_regime_probabilities": (1.0,),
        },
        {
            "cpu_cycles_per_s_regimes": ((1.0, 2.0), (2.0, 3.0)),
            "cpu_regime_probabilities": (0.2, 0.2),
        },
    ),
)
def test_invalid_regime_contract_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ScenarioConfig(**kwargs)


def test_optional_defaults_are_digest_neutral_but_enabled_sampler_is_not():
    base = config_digest({"scenario": {"num_vehicles": 20}})
    explicit = config_digest({
        "scenario": {
            "num_vehicles": 20,
            "cpu_cycles_per_s_regimes": None,
            "cpu_regime_probabilities": None,
        }
    })
    enabled = config_digest({
        "scenario": {
            "num_vehicles": 20,
            "cpu_cycles_per_s_regimes": RANGES,
            "cpu_regime_probabilities": (0.2,) * 5,
        }
    })
    assert base == explicit
    assert enabled != base
    assert "cpu_cycles_per_s_regimes" in asdict(ScenarioConfig())
