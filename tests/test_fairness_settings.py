from scripts.report_fairness_settings import fairness_rows
from src.evaluation.revision import (
    ALL_ALGORITHMS,
    STOCHASTIC_ALGORITHMS,
    load_revision_config,
)


def test_fairness_report_has_one_row_per_algorithm_and_fixed_core() -> None:
    cfg = load_revision_config()
    rows = fairness_rows(cfg)

    assert [row["algorithm"] for row in rows] == list(ALL_ALGORITHMS)
    stochastic = [
        row for row in rows if row["algorithm"] in STOCHASTIC_ALGORITHMS
    ]
    assert {row["network_hidden_sizes"] for row in stochastic} == {"[256, 256]"}
    assert {float(row["learning_rate"]) for row in stochastic} == {0.0003}
    assert {float(row["clip_range"]) for row in stochastic} == {0.2}
    assert {float(row["entropy_coefficient"]) for row in stochastic} == {0.01}
    assert {int(row["total_timesteps"]) for row in stochastic} == {600_000}
    assert {int(row["headline_train_seeds"]) for row in stochastic} == {8}


def test_fairness_report_discloses_engine_specific_update_layouts() -> None:
    rows = {
        row["algorithm"]: row for row in fairness_rows(load_revision_config())
    }

    assert rows["proposed"]["rollout_samples_per_update"] == 1024
    assert rows["proposed"]["epochs_per_update"] == 10
    assert rows["jdls"]["rollout_samples_per_update"] == 1024
    assert rows["jdls"]["epochs_per_update"] == 10
    assert rows["mappo"]["rollout_samples_per_update"] == 1024
    assert rows["mappo"]["epochs_per_update"] == 4
    assert rows["mappo"]["collection_microbatch_size"] == 32
    assert rows["jdls"]["initialization"] == "cold start"
    assert rows["mappo"]["initialization"] == "cold start"
