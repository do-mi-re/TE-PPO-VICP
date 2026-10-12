#!/usr/bin/env bash
# Safe convenience dispatcher for the maintained experiment workflow.
# Default is read-only. Formal multi-day stages require an explicit phase.
set -euo pipefail

# Formal-server defaults. Experiment-level parallelism is controlled by the
# Python scheduler; one native thread per process prevents 32 concurrent tiny
# RL models from each creating a 48-thread BLAS/OpenMP pool.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TEPPO_STRICT_VECENV="${TEPPO_STRICT_VECENV:-1}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export TEPPO_CPU_CAPACITY="${TEPPO_CPU_CAPACITY:-48}"
server_cache_root="${TMPDIR:-/tmp}/copsens_boyi_cache_${UID:-user}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${server_cache_root}/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${server_cache_root}/xdg}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

if [[ -x .venv/bin/python ]]; then
  python_cmd=(.venv/bin/python)
else
  python_cmd=(uv run python)
fi
run_python() { PYTHONPATH=. "${python_cmd[@]}" "$@"; }

# Build the shared font cache once in the parent. Otherwise dozens of spawned
# workers may each pay the same first-import cost (or contend on cache files).
run_python -c "import matplotlib"

phase="${1:---check}"
if [[ $# -gt 0 ]]; then
  shift
fi
phase_args=("$@")
case "$phase" in
  --check)
    run_python scripts/validate_revision_outputs.py --preflight
    run_python scripts/run_revision_pipeline.py train-headline --dry-run
    run_python scripts/run_ablation_experiment.py plan
    run_python scripts/run_scalability_experiment.py plan --axis vehicles
    run_python scripts/run_scalability_experiment.py plan --axis tasks
    run_python scripts/run_scalability_experiment.py plan --axis timing
    run_python scripts/run_pruning_sensitivity.py plan
    run_python scripts/run_dynamic_robustness.py --dry-run
    run_python scripts/run_main_text_analysis.py --dry-run
    run_python scripts/run_cpu_robustness_experiment.py plan
    run_python scripts/measure_inference_latency.py --dry-run
    ;;
  calibrate-dbas)
    run_python scripts/tune_dbas_price.py
    ;;
  headline)
    run_python scripts/tune_dbas_price.py
    run_python scripts/run_revision_pipeline.py train-headline "${phase_args[@]}"
    run_python scripts/run_revision_pipeline.py evaluate-headline
    ;;
  cpu-robustness)
    run_python scripts/run_cpu_robustness_experiment.py calibrate
    run_python scripts/run_cpu_robustness_experiment.py train "${phase_args[@]}"
    run_python scripts/run_cpu_robustness_experiment.py evaluate
    ;;
  eval-only)
    run_python scripts/run_scalability_experiment.py evaluate --axis vehicles
    run_python scripts/run_scalability_experiment.py evaluate --axis tasks
    run_python scripts/run_pruning_sensitivity.py evaluate
    run_python scripts/run_dynamic_robustness.py
    run_python scripts/run_cpu_robustness_experiment.py evaluate
    run_python scripts/run_main_text_analysis.py mechanisms
    ;;
  pruning-sensitivity)
    run_python scripts/run_pruning_sensitivity.py train "${phase_args[@]}"
    run_python scripts/run_pruning_sensitivity.py evaluate
    ;;
  ablation)
    run_python scripts/run_ablation_experiment.py train "${phase_args[@]}"
    run_python scripts/run_ablation_experiment.py evaluate
    ;;
  scalability)
    run_python scripts/run_scalability_experiment.py train --axis vehicles "${phase_args[@]}"
    run_python scripts/run_scalability_experiment.py evaluate --axis vehicles
    run_python scripts/run_scalability_experiment.py train --axis tasks "${phase_args[@]}"
    run_python scripts/run_scalability_experiment.py evaluate --axis tasks
    run_python scripts/run_scalability_experiment.py train --axis timing "${phase_args[@]}"
    ;;
  timing)
    run_python scripts/measure_inference_latency.py --scope all "${phase_args[@]}"
    ;;
  reports)
    run_python scripts/run_pruning_sensitivity.py plot
    run_python scripts/report_summary_stats.py
    run_python scripts/report_convergence.py
    run_python scripts/report_action_dimensions.py
    run_python scripts/report_fairness_settings.py
    run_python scripts/measure_inference_latency.py --scope all
    run_python scripts/report_ablation.py
    run_python scripts/report_scalability.py
    run_python scripts/report_main_text_figures.py
    run_python scripts/report_manuscript_figures.py
    run_python scripts/report_training_cost.py
    run_python scripts/validate_revision_outputs.py
    ;;
  *)
    echo "usage: $0 [--check|calibrate-dbas|headline|cpu-robustness|eval-only|pruning-sensitivity|ablation|scalability|timing|reports]" >&2
    exit 2
    ;;
esac
