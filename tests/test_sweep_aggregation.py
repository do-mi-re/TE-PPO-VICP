import math

from src.evaluation.sweep import aggregate_sweep_rows
from src.rl.train import load_training_config


def _row(eval_seed, train_seed, value, *, metric="raw_rsu_reward"):
    return {
        "algorithm": "proposed",
        "sweep_value": 10.0,
        "eval_seed": eval_seed,
        "train_seed": train_seed,
        metric: value,
    }


def test_aggregate_sweep_rows_std_is_spread_across_scenarios() -> None:
    # Two scenarios (eval seeds), each averaged across two train seeds:
    #   eval 40000: mean(0, 4) = 2.0
    #   eval 40001: mean(10, 6) = 8.0
    # scenario means [2, 8] → point mean 5.0, population std 3.0.
    rows = [
        _row(40000, 101, 0.0), _row(40000, 202, 4.0),
        _row(40001, 101, 10.0), _row(40001, 202, 6.0),
    ]

    aggregates = aggregate_sweep_rows(rows, metrics=("raw_rsu_reward",))

    assert len(aggregates) == 1
    record = aggregates[0]
    assert record["replicate_count"] == 2  # distinct train seeds
    assert record["raw_rsu_reward_mean"] == 5.0
    assert math.isclose(record["raw_rsu_reward_std"], 3.0)
    assert record["raw_rsu_reward_replicates"] == 2  # valid scenarios


def test_aggregate_sweep_rows_single_train_seed_still_gets_a_band() -> None:
    # DBAS-style: one train seed, but multiple eval scenarios → real std.
    rows = [_row(40000, 101, 2.0), _row(40001, 101, 8.0)]

    aggregates = aggregate_sweep_rows(rows, metrics=("raw_rsu_reward",))

    record = aggregates[0]
    assert record["replicate_count"] == 1
    assert math.isclose(record["raw_rsu_reward_mean"], 5.0)
    assert math.isclose(record["raw_rsu_reward_std"], 3.0)  # non-zero, from 2 scenarios


def test_aggregate_sweep_rows_single_scenario_has_nan_std() -> None:
    # A single valid scenario → std undefined → NaN (not a misleading 0.0).
    rows = [_row(40000, 101, 7.0)]

    record = aggregate_sweep_rows(rows, metrics=("raw_rsu_reward",))[0]
    assert math.isclose(record["raw_rsu_reward_mean"], 7.0)
    assert math.isnan(record["raw_rsu_reward_std"])
    assert record["raw_rsu_reward_replicates"] == 1


def test_aggregate_sweep_rows_reports_zero_replicates_for_all_nan_metric() -> None:
    rows = [
        _row(40000, 101, math.nan, metric="mean_selected_latency_s"),
        _row(40001, 202, math.nan, metric="mean_selected_latency_s"),
    ]

    aggregates = aggregate_sweep_rows(rows, metrics=("mean_selected_latency_s",))

    record = aggregates[0]
    assert record["replicate_count"] == 2
    assert record["mean_selected_latency_s_replicates"] == 0
    assert math.isnan(record["mean_selected_latency_s_mean"])
    assert math.isnan(record["mean_selected_latency_s_std"])


def test_default_training_config_uses_cpu_device() -> None:
    config = load_training_config("configs/default.yaml")

    assert config["algorithm"]["device"] == "cpu"


def test_joint_learning_configs_use_common_physical_validation_interval() -> None:
    mappo = load_training_config("configs/algorithms/mappo.yaml")
    jdls = load_training_config("configs/algorithms/jdls.yaml")

    assert mappo["evaluation"]["interval_env_steps"] == 3072
    assert jdls["evaluation"]["interval_env_steps"] == 3072


def test_aggregate_sweep_rows_seed_std_is_spread_across_train_seeds() -> None:
    # Same four rows as the scenario-spread test above, read the other way:
    #   train 101: mean(0, 10) = 5.0
    #   train 202: mean(4, 6)  = 5.0
    # Per-seed means [5, 5] → seed std 0.0, while the scenario std is 3.0.
    # The two definitions measure different sources of variability:
    # the seed one), and this fixture separates them cleanly.
    rows = [
        _row(40000, 101, 0.0), _row(40000, 202, 4.0),
        _row(40001, 101, 10.0), _row(40001, 202, 6.0),
    ]

    aggregates = aggregate_sweep_rows(rows, metrics=("raw_rsu_reward",))

    record = aggregates[0]
    assert record["raw_rsu_reward_seed_mean"] == 5.0
    assert math.isclose(record["raw_rsu_reward_seed_std"], 0.0, abs_tol=1e-12)
    assert record["raw_rsu_reward_seed_replicates"] == 2
    # Scenario spread is untouched and still reports the scenario definition.
    assert math.isclose(record["raw_rsu_reward_std"], 3.0)


def test_aggregate_sweep_rows_seed_std_uses_sample_ddof() -> None:
    # Per-seed means [2, 4, 9] → sample std (ddof=1) = 3.60555, not the
    # population std 2.94392. Train seeds are a sample of the seed
    # population, so ddof=1 is the right estimator.
    rows = [
        _row(40000, 101, 2.0), _row(40000, 202, 4.0), _row(40000, 303, 9.0),
    ]

    record = aggregate_sweep_rows(rows, metrics=("raw_rsu_reward",))[0]

    assert math.isclose(record["raw_rsu_reward_seed_std"], 3.605551275463989)
    assert math.isclose(
        record["raw_rsu_reward_seed_ci95"], 1.96 * 3.605551275463989 / math.sqrt(3)
    )


def test_aggregate_sweep_rows_single_train_seed_has_nan_seed_std() -> None:
    # DBAS-style: one train seed. A scenario band is still meaningful (see
    # test_aggregate_sweep_rows_single_train_seed_still_gets_a_band), but
    # there is no retraining variability to measure, so the seed std is NaN
    # rather than a zero-width band that would read as false certainty.
    rows = [_row(40000, 101, 2.0), _row(40001, 101, 8.0)]

    record = aggregate_sweep_rows(rows, metrics=("raw_rsu_reward",))[0]

    assert record["raw_rsu_reward_seed_replicates"] == 1
    assert math.isnan(record["raw_rsu_reward_seed_std"])
    assert math.isnan(record["raw_rsu_reward_seed_ci95"])
    assert math.isclose(record["raw_rsu_reward_std"], 3.0)  # scenario band survives


def test_aggregate_sweep_rows_uncertainty_seed_repoints_std() -> None:
    # uncertainty="seed" is what the plot layer uses to pick up the seed
    # definition without teaching it a second key name.
    rows = [
        _row(40000, 101, 0.0), _row(40000, 202, 4.0),
        _row(40001, 101, 10.0), _row(40001, 202, 6.0),
    ]

    record = aggregate_sweep_rows(
        rows, metrics=("raw_rsu_reward",), uncertainty="seed"
    )[0]

    # Seed uncertainty is 0.0 for these equal seed-level means.
    assert math.isclose(record["raw_rsu_reward_std"], 0.0, abs_tol=1e-12)
    assert math.isclose(record["raw_rsu_reward_seed_std"], 0.0, abs_tol=1e-12)
    # The mean is a property of the data, not of the uncertainty definition.
    assert record["raw_rsu_reward_mean"] == 5.0


def test_aggregate_sweep_rows_rejects_unknown_uncertainty() -> None:
    rows = [_row(40000, 101, 2.0)]

    try:
        aggregate_sweep_rows(rows, metrics=("raw_rsu_reward",), uncertainty="bogus")
    except ValueError as exc:
        assert "uncertainty" in str(exc)
    else:  # pragma: no cover - the call above must raise
        raise AssertionError("expected ValueError for an unknown uncertainty mode")
