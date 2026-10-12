from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass

import numpy as np
import pytest

from scripts.measure_inference_latency import render_figure, summarize_ablation_timing
from scripts.report_ablation import ARM_LABELS, missing_ablation_arms
from scripts.report_action_dimensions import dimension_row
from scripts.report_convergence import (
    isolated_wall_time,
    load_common_reward_curve,
    normalized_aulc,
    progress_target,
    steps_to_threshold,
    validate_curve_coverage,
)
from scripts.report_summary_stats import (
    ALGORITHM_ORDER,
    paired_improvement_summary,
    paired_signflip_p,
    per_seed_mean_map,
)
from scripts.report_summary_stats import main as report_summary_main
from scripts.run_ablation_experiment import summarize_ablation_rows
from scripts.validate_revision_outputs import _validate_significance_report
from src.evaluation.ablation import (
    conditional_ratio_summary,
    paired_ablation_delta_summary,
    paired_ablation_ratio_delta_summary,
)
from src.plotting.revision_style import FIGSIZE_SINGLE_COLUMN
from src.rl.eval import SWEEP_METRIC_NAMES


def test_ablation_report_completeness_check_accepts_mapping_keys():
    complete = {arm: {} for arm in ARM_LABELS}
    assert missing_ablation_arms(complete) == set()

    incomplete = dict(complete)
    incomplete.pop("no_candidate_pruning")
    assert missing_ablation_arms(incomplete) == {"no_candidate_pruning"}


def test_ablation_timing_aggregates_checkpoint_medians_across_seeds():
    rows = []
    for arm in ARM_LABELS:
        for seed, total in ((1, 4.0), (2, 6.0)):
            rows.append({
                "ablation_arm": arm,
                "algorithm": "jdls" if arm == "no_analytical_best_response" else "proposed",
                "train_seed": seed,
                "total_cycle_ms": total,
                "cycle_p95_ms": total + 1.0,
                "inference_wall_ms": total - 0.5,
            })
    summary = summarize_ablation_timing(rows)
    assert len(summary) == len(ARM_LABELS)
    assert all(row["seed_count"] == 2 for row in summary)
    assert all(row["total_cycle_ms_mean"] == 5.0 for row in summary)
    assert all(row["total_cycle_ms_ci95"] > 0.0 for row in summary)


def test_latency_figure_uses_single_column_size_and_distinct_series_styles(
    tmp_path, monkeypatch
):
    import matplotlib.axes

    calls: list[dict[str, object]] = []
    original_errorbar = matplotlib.axes.Axes.errorbar

    def capture_errorbar(self, *args, **kwargs):
        calls.append({"args": args, **kwargs})
        return original_errorbar(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "errorbar", capture_errorbar)
    rows = [
        {
            "num_vehicles": vehicles,
            "num_tasks": tasks,
            "total_cycle_ms": 1.0 + vehicles / 10 + tasks / 10,
            "cycle_p95_ms": 1.2 + vehicles / 10 + tasks / 10,
        }
        for tasks in (3, 6, 9, 12)
        for vehicles in (10, 20, 30, 40, 50, 60)
    ]

    stem = tmp_path / "figR_latency"
    render_figure(rows, stem)

    assert stem.with_suffix(".png").stat().st_size > 0
    pdf_path = stem.with_suffix(".pdf")
    assert pdf_path.stat().st_size > 0
    media_box = re.search(
        rb"/MediaBox\s*\[\s*0\s+0\s+([0-9.]+)\s+([0-9.]+)\s*\]",
        pdf_path.read_bytes(),
    )
    assert media_box is not None
    width_points, height_points = (float(value) for value in media_box.groups())
    assert width_points / 72 == pytest.approx(FIGSIZE_SINGLE_COLUMN[0])
    assert height_points / 72 == pytest.approx(FIGSIZE_SINGLE_COLUMN[1])
    assert len(calls) == 4
    assert len({call["marker"] for call in calls}) == 4
    assert len({call["linestyle"] for call in calls}) == 4
    assert all(len(call["args"][0]) == 6 for call in calls)
    assert all(call["barsabove"] is True for call in calls)
    assert all(call["capsize"] == 3.5 for call in calls)
    assert all(call["capthick"] == 1.0 for call in calls)
    assert all(call["elinewidth"] == 1.0 for call in calls)


def test_per_seed_map_averages_eval_scenarios_before_statistics():
    rows = [
        {"algorithm": "proposed", "train_seed": 1, "score": 1.0},
        {"algorithm": "proposed", "train_seed": 1, "score": 3.0},
        {"algorithm": "proposed", "train_seed": 2, "score": 5.0},
    ]
    assert per_seed_mean_map(rows, "proposed", "score") == {"1": 2.0, "2": 5.0}


def test_paired_signflip_uses_only_shared_training_seed_identities():
    proposed = {"1": 4.0, "2": 5.0, "3": 100.0}
    baseline = {"1": 1.0, "2": 2.0, "4": -100.0}
    p, n = paired_signflip_p(proposed, baseline)
    assert n == 2
    assert p == pytest.approx(0.5)


def test_paired_improvement_interval_is_computed_across_train_seeds():
    mean, std, ci = paired_improvement_summary({"1": 12.0, "2": 24.0}, {"1": 10.0, "2": 20.0})
    assert mean == pytest.approx(20.0)
    assert std == pytest.approx(0.0)
    assert ci == pytest.approx(0.0)


def test_summary_report_uses_paired_signflip_as_primary_evidence(tmp_path, capsys):
    outputs = tmp_path / "outputs"
    headline = outputs / "headline" / "episodes.csv"
    headline.parent.mkdir(parents=True)
    rows = []
    for algorithm_index, algorithm in enumerate(ALGORITHM_ORDER):
        train_seeds = (1,) if algorithm == "dbas" else (1, 2, 3)
        for train_seed in train_seeds:
            row = {metric: 0.0 for metric in SWEEP_METRIC_NAMES}
            comparison = 20.0 + train_seed - algorithm_index
            value = comparison + 1.0 if algorithm == "proposed" else comparison
            row.update(
                {
                    "algorithm": algorithm,
                    "train_seed": train_seed,
                    "eval_seed": 100,
                    "social_welfare": value,
                    "raw_rsu_reward": value,
                    "perception_quality_total": value,
                }
            )
            rows.append(row)
    with headline.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    assert (
        report_summary_main(["--csv", str(headline), "--output-dir", str(outputs / "tables")]) == 0
    )

    console = capsys.readouterr().out
    assert "exact two-sided paired sign-flip test" in console
    assert "p=0.2500 (exact)" in console
    significance_path = outputs / "tables" / "significance.csv"
    with significance_path.open(newline="", encoding="utf-8") as handle:
        significance = list(csv.DictReader(handle))
    ebas = [row for row in significance if row["baseline"] == "ebas"]
    assert len(ebas) == 3
    assert all(row["primary_test"] == "exact_two_sided_paired_sign_flip" for row in ebas)
    assert all(float(row["primary_p_value"]) == pytest.approx(0.25) for row in ebas)
    dbas = [row for row in significance if row["baseline"] == "dbas"]
    assert all(row["primary_p_value"] == "" for row in dbas)

    issues: list[str] = []
    _validate_significance_report(outputs, issues)
    assert issues == []

    ebas[0]["primary_p_value"] = "0.5"
    with significance_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(significance[0]))
        writer.writeheader()
        writer.writerows(significance)
    issues = []
    _validate_significance_report(outputs, issues)
    assert any("wrong primary p-value" in issue for issue in issues)


def test_ablation_percentage_delta_is_formed_within_matched_seeds():
    rows = []
    for arm, values in {"full": (10.0, 20.0), "variant": (8.0, 10.0)}.items():
        for train_seed, value in zip((1, 2), values, strict=True):
            for eval_seed in (10, 11):
                rows.append(
                    {
                        "ablation_arm": arm,
                        "train_seed": train_seed,
                        "eval_seed": eval_seed,
                        "score": value,
                    }
                )

    result = paired_ablation_delta_summary(
        rows,
        arm="variant",
        metric="score",
        train_seeds=(1, 2),
        eval_seeds=(10, 11),
    )

    assert result["mean"] == pytest.approx(-35.0)
    assert result["std"] == pytest.approx(21.2132034356)
    assert result["ci95"] == pytest.approx(29.4)
    assert result["paired_permutation_p"] == pytest.approx(0.5)
    assert result["pairs"] == 2


def test_ablation_delta_requires_complete_matched_test_bank():
    rows = [
        {"ablation_arm": "full", "train_seed": 1, "eval_seed": 10, "score": 1.0},
        {"ablation_arm": "variant", "train_seed": 1, "eval_seed": 10, "score": 2.0},
    ]
    with pytest.raises(ValueError, match="complete unique test-seed bank"):
        paired_ablation_delta_summary(
            rows,
            arm="variant",
            metric="score",
            train_seeds=(1,),
            eval_seeds=(10, 11),
        )


def _conditional_ablation_rows(
    variant_participants: tuple[tuple[float, float], tuple[float, float]],
):
    rows = []
    for arm, energies in (("full", (2.0, 4.0)), ("variant", (1.0, 3.0))):
        for seed_index, train_seed in enumerate((1, 2)):
            participants = (
                (1.0, 2.0)
                if arm == "full"
                else variant_participants[seed_index]
            )
            for eval_seed, energy, selected in zip(
                (10, 11), energies, participants, strict=True
            ):
                rows.append(
                    {
                        "ablation_arm": arm,
                        "train_seed": train_seed,
                        "eval_seed": eval_seed,
                        "energy_total_sum": energy if selected > 0.0 else 0.0,
                        "selected_vehicle_count": selected,
                    }
                )
    return rows


def test_conditional_energy_uses_ratio_of_sums_per_training_seed():
    rows = _conditional_ablation_rows(((1.0, 3.0), (1.0, 3.0)))

    result = conditional_ratio_summary(
        rows,
        arm="variant",
        numerator="energy_total_sum",
        denominator="selected_vehicle_count",
        train_seeds=(1, 2),
        eval_seeds=(10, 11),
    )
    delta = paired_ablation_ratio_delta_summary(
        rows,
        arm="variant",
        numerator="energy_total_sum",
        denominator="selected_vehicle_count",
        train_seeds=(1, 2),
        eval_seeds=(10, 11),
    )

    assert result["mean"] == pytest.approx(1.0)
    assert result["seed_replicates"] == 2
    assert result["valid_count"] == 4
    assert result["status"] == "estimable"
    assert delta["mean"] == pytest.approx(-50.0)
    assert delta["pairs"] == 2
    assert delta["status"] == "estimable"


def test_conditional_energy_is_not_coerced_to_zero_without_participants():
    rows = _conditional_ablation_rows(((0.0, 0.0), (0.0, 0.0)))

    result = conditional_ratio_summary(
        rows,
        arm="variant",
        numerator="energy_total_sum",
        denominator="selected_vehicle_count",
        train_seeds=(1, 2),
        eval_seeds=(10, 11),
    )
    delta = paired_ablation_ratio_delta_summary(
        rows,
        arm="variant",
        numerator="energy_total_sum",
        denominator="selected_vehicle_count",
        train_seeds=(1, 2),
        eval_seeds=(10, 11),
    )

    assert np.isnan(result["mean"])
    assert result["status"] == "not_estimable_no_participants"
    assert result["valid_count"] == 0
    assert np.isnan(delta["mean"])
    assert delta["pairs"] == 0
    assert delta["status"] == "not_estimable_no_participants"


def test_conditional_delta_requires_every_matched_training_seed():
    rows = _conditional_ablation_rows(((1.0, 1.0), (0.0, 0.0)))

    delta = paired_ablation_ratio_delta_summary(
        rows,
        arm="variant",
        numerator="energy_total_sum",
        denominator="selected_vehicle_count",
        train_seeds=(1, 2),
        eval_seeds=(10, 11),
    )

    assert np.isnan(delta["mean"])
    assert delta["pairs"] == 1
    assert delta["status"] == "not_estimable_incomplete_pairs"


def test_ablation_summary_accepts_structural_no_participation():
    rows = []
    for arm, engine, selected in (
        ("full", "proposed", 2.0),
        ("no_analytical_best_response", "jdls", 0.0),
    ):
        for train_seed in (1, 2):
            for eval_seed in (10, 11):
                row = {
                    "ablation_arm": arm,
                    "ablation_engine": engine,
                    "train_seed": train_seed,
                    "eval_seed": eval_seed,
                    "energy_total_sum": selected,
                    "energy_per_participating_vehicle_j": (
                        1.0 if selected > 0.0 else float("nan")
                    ),
                    "participation_rate": float(selected > 0.0),
                    "selected_vehicle_count": selected,
                    "task_success_rate": float(selected > 0.0),
                }
                for metric in (
                    "social_welfare",
                    "raw_rsu_reward",
                    "perception_quality_total",
                    "candidate_pruned_pair_ratio",
                    "inference_wall_time_ms",
                    "follower_solve_time_ms",
                ):
                    row[metric] = 1.0 if selected > 0.0 else 0.0
                rows.append(row)

    summary = summarize_ablation_rows(
        rows, train_seeds=(1, 2), eval_seeds=(10, 11)
    )
    no_response = next(
        row for row in summary
        if row["ablation_arm"] == "no_analytical_best_response"
    )

    assert np.isnan(no_response["energy_per_participating_vehicle_j_mean"])
    assert (
        no_response["energy_per_participating_vehicle_j_status"]
        == "not_estimable_no_participants"
    )
    assert no_response["energy_per_participating_vehicle_j_valid_count"] == 0
    assert no_response["selected_vehicle_count_change_vs_full_pct"] == -100.0
    assert no_response["task_success_rate_change_vs_full_pct"] == -100.0


def test_convergence_threshold_ignores_transient_spike():
    timesteps = np.array([10, 20, 30, 40])
    curve = np.array([0.0, 10.0, 1.0, 10.0])
    assert steps_to_threshold(timesteps, curve, 5.0) == 40


def test_normalized_aulc_is_step_span_normalized():
    timesteps = np.array([0, 10, 20])
    curve = np.array([0.0, 1.0, 2.0])
    assert normalized_aulc(timesteps, curve) == pytest.approx(1.0)


def test_convergence_coverage_rejects_truncated_curve(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config_snapshot.yaml").write_text(
        "algorithm:\n  name: proposed\n  total_timesteps: 100\n  n_envs: 2\n"
        "evaluation:\n  interval_env_steps: 5\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="truncated"):
        validate_curve_coverage(np.array([10, 20, 30]), np.array([1.0, 2.0, 3.0]), None, run_dir)


def test_convergence_progress_target_handles_signed_welfare():
    assert progress_target(-10.0, -2.0, 0.8) == pytest.approx(-3.6)
    assert progress_target(-2.0, -3.0, 0.8) is None


def test_convergence_wall_time_accepts_only_isolated_record(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    wall = np.array([1.0, 2.0])
    (run_dir / "training_record.json").write_text(
        json.dumps({"execution_mode": "scheduled"}), encoding="utf-8"
    )
    assert isolated_wall_time(run_dir, wall) is None
    (run_dir / "training_record.json").write_text(
        json.dumps({"execution_mode": "isolated"}), encoding="utf-8"
    )
    assert isolated_wall_time(run_dir, wall) is wall


def test_common_reward_curve_checks_explicit_schema(tmp_path):
    path = tmp_path / "evaluations.npz"
    np.savez(
        path,
        timesteps=np.array([10]),
        common_evaluation_reward=np.array([[5.0, 7.0]]),
        reward_schema=np.asarray("common_evaluation_reward_v1"),
    )
    _, values, source = load_common_reward_curve(
        path,
        algorithm="jdls",
        expected_schema="common_evaluation_reward_v1",
    )
    np.testing.assert_allclose(values, [6.0])
    assert source == "common_evaluation_reward"


def test_common_reward_curve_rejects_mismatched_evaluation_grid(tmp_path):
    path = tmp_path / "evaluations.npz"
    np.savez(
        path,
        timesteps=np.array([10]),
        common_evaluation_reward=np.array([[5.0, 7.0]]),
        social_welfare=np.array([[1.0]]),
        reward_schema=np.asarray("common_evaluation_reward_v1"),
    )
    with pytest.raises(ValueError, match="evaluation grids differ"):
        load_common_reward_curve(
            path,
            algorithm="jdls",
            expected_schema="common_evaluation_reward_v1",
            )


def test_convergence_report_rejects_stale_artifact_before_curve_load(tmp_path, monkeypatch):
    import scripts.report_convergence as report

    @dataclass
    class Spec:
        algorithm: str = "proposed"
        train_seed: int = 1

        def to_worker_dict(self):
            return {"algorithm": self.algorithm, "train_seed": self.train_seed}

    run_dir = tmp_path / "artifacts" / "proposed_s1_headline" / "eval"
    run_dir.mkdir(parents=True)
    (run_dir / "evaluations.npz").write_bytes(b"not-read")
    monkeypatch.setattr(
        report,
        "load_revision_config",
        lambda: {
            "seeds": {"validation": (30_000,)},
            "common_evaluation_reward": {
                "schema": "common_evaluation_reward_v1",
                "plotted_algorithms": ["proposed"],
                "figure_algorithms": ["proposed"],
            },
        },
    )
    monkeypatch.setattr(report, "headline_specs", lambda _cfg: [Spec()])
    monkeypatch.setattr(report.orch, "materialize_config", lambda *_args: {})
    monkeypatch.setattr(
        report,
        "validate_run_artifact",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("stale artifact")),
    )
    monkeypatch.setattr(
        report,
        "load_curve",
        lambda _path: pytest.fail("curve must not load before provenance validation"),
    )

    code = report.main(
        [
            "--artifacts-root",
            str(tmp_path / "artifacts"),
            "--algorithms",
            "proposed",
            "--seeds",
            "1",
            "--output-dir",
            str(tmp_path / "outputs"),
            "--no-figure",
        ]
    )

    assert code == 1


def test_action_dimension_report_distinguishes_decisions_and_logits():
    row = dimension_row(20, 3)
    assert row["te_physical_decisions"] == 23
    assert row["te_sampled_latents"] == 24
    assert row["jdls_physical_decisions"] == 83
    assert row["jdls_sampled_variables"] == 84
    assert row["jdls_actor_output_parameters"] == 124


def test_common_reward_curve_requires_the_declared_metric(tmp_path):
    path = tmp_path / "evaluations.npz"
    np.savez(path, timesteps=np.array([10]), results=np.array([[1.0, 3.0]]))
    with pytest.raises(KeyError, match="lacks common_evaluation_reward_v1"):
        load_common_reward_curve(path, algorithm="proposed",
                                expected_schema="common_evaluation_reward_v1")
