import json

import numpy as np
import pytest

from src.evaluation.provenance import (
    TRAIN_FINGERPRINT_FILE,
    config_digest,
    validate_run_artifact,
)
from src.evaluation.sweep import load_algorithm_config
from src.rl.checkpoint_selection import CPU_MAXIMIN_SELECTION_RULE
from src.rl.train import train_from_config
from src.rl.training_config import resolve_effective_training_config


@pytest.mark.parametrize("algorithm", ("proposed", "jdls", "mappo"))
def test_training_engine_emits_maximin_selection_artifact(tmp_path, algorithm):
    config = load_algorithm_config(algorithm)
    config["seed"] = 123
    config["scenario"].update({
        "num_vehicles": 4,
        "cpu_cycles_per_s_regimes": ((2.5e9, 7.5e9), (22.5e9, 27.5e9)),
        "cpu_regime_probabilities": (0.5, 0.5),
    })
    config["evaluation"].update({
        "seed_list": [30000, 30001],
        "interval_env_steps": 8,
        "checkpoint_selection": CPU_MAXIMIN_SELECTION_RULE,
        "cpu_capacity_ranges_cycles_per_s": (
            (2.5e9, 7.5e9), (22.5e9, 27.5e9),
        ),
    })
    config["artifacts"]["run_name"] = f"{algorithm}_cpu_smoke"
    config["algorithm"].update({
        "total_timesteps": 16,
        "n_epochs": 1,
        "target_kl": None,
        "device": "cpu",
    })
    if algorithm == "proposed":
        config["algorithm"].update({
            "n_envs": 1,
            "n_steps": 8,
            "batch_size": 8,
        })
        config["artifacts"]["checkpoint_freq"] = 8
    else:
        config["algorithm"].update({
            "rollout_steps": 8,
            "minibatch_size": 8,
        })

    effective_config = resolve_effective_training_config(config)
    summary = train_from_config(effective_config, output_root=tmp_path)
    run_dir = tmp_path / config["artifacts"]["run_name"]
    (run_dir / TRAIN_FINGERPRINT_FILE).write_text(
        f"{config_digest(effective_config)}\n",
        encoding="utf-8",
    )
    selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    assert selection["schema_version"] == 2
    assert selection["selection_rule"] == CPU_MAXIMIN_SELECTION_RULE
    assert len(selection["selected_cell_social_welfare_means"]) == 2
    with np.load(run_dir / "eval" / "evaluations.npz", allow_pickle=False) as data:
        assert data["social_welfare"].shape[1] == 4
        assert data["validation_cell_index"].shape == data["social_welfare"].shape
        assert data["raw_rsu_reward"].shape == data["social_welfare"].shape
        assert data["common_evaluation_reward"].shape == data["social_welfare"].shape
        assert str(data["reward_schema"].item()) == "common_evaluation_reward_v1"
        if algorithm == "proposed":
            np.testing.assert_allclose(
                data["common_evaluation_reward"], data["results"]
            )
        else:
            np.testing.assert_allclose(data["raw_rsu_reward"], data["results"])
        assert np.all(
            data["common_evaluation_reward"] <= data["raw_rsu_reward"] + 1e-12
        )
    validate_run_artifact(run_dir, expected_config=effective_config)
    assert summary["total_timesteps"] == 16
