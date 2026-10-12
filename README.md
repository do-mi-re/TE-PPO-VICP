# Cooperative Perception Stackelberg Experiments

Research code for **Incentive-Aware Resource Allocation for Energy-Efficient
Cooperative Perception: A Stackelberg Game Approach**.

The repository implements a two-stage scheduling cycle: the RSU selects task
prices and vehicle bandwidth shares, and vehicles compute their task,
compression, and sensing-effort responses. TE-PPO learns the leader policy;
JDLS, MAPPO (labelled MAPS in figures), DBAS, EBAS, SPIS, and RDUS provide the
comparison methods.

## Installation

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync --dev --frozen
```

The locked dependencies include PyTorch and Stable-Baselines3. CPU execution is
supported. GPU execution requires a compatible NVIDIA driver and PyTorch CUDA
runtime. Experiment configurations use `device: auto`; set it to `cpu` for CPU
runs. The dependency index is declared in `pyproject.toml`.

## Quick start

Inspect a deterministic scheduling cycle without training:

```bash
uv run python -m src.examples.single_step_explain
```

The example uses the maintained sensing and price bounds on a manually
specified three-vehicle, two-task geometry. It illustrates the model and is
separate from the configured experiment grids and seed banks.

Check the installation and experiment configuration:

```bash
uv run ruff check .
uv run pytest -q
PYTHONPATH=. uv run python scripts/validate_revision_outputs.py --preflight
bash scripts/run_revision_pipeline.sh --check
```

Run a small installation smoke test, separate from paper experiments:

```bash
PYTHONPATH=. uv run python -m src.rl.train --config configs/train_small.yaml
```

Its output is `artifacts/ppo/train_small/`; it does not qualify as a paper
checkpoint. Run all script commands from the repository root.

## Reproducing the experiments

`configs/revision.yaml` defines the experiment grids, seed banks, training
budgets, and output paths. Algorithm YAMLs specify each method's implementation
and optimizer settings. The [experiment protocol](docs/revision_experiment_protocol.md)
describes statistical units, checkpoint selection, and timing.

Execute the following phases in order:

```bash
bash scripts/run_revision_pipeline.sh headline
bash scripts/run_revision_pipeline.sh pruning-sensitivity
bash scripts/run_revision_pipeline.sh cpu-robustness
bash scripts/run_revision_pipeline.sh ablation
bash scripts/run_revision_pipeline.sh scalability
bash scripts/run_revision_pipeline.sh eval-only
bash scripts/run_revision_pipeline.sh timing
bash scripts/run_revision_pipeline.sh reports
```

The headline phase calibrates DBAS on validation scenarios, trains missing
checkpoints, and evaluates them on the held-out test bank. The CPU phase
calibrates a separate DBAS price across the CPU-domain grid. Reports require
complete inputs; missing or mismatched artifacts produce an error.

Formal checkpoints and reports use `artifacts/revision_2026_v5/` and
`outputs/revision_2026_v5/`. Existing matching runs are reused. Incompatible or
partial runs block training until explicitly replaced with `--retrain-stale`;
the `headline`, `cpu-robustness`, and `ablation` phases also accept `--force`
to replace all requested runs. Back up any outputs that you intend to retain
before requesting replacement.

Learning experiments can take substantial time. Use the plan commands under
`--check` to inspect job counts first. Training commands accept `--max-workers`.
The launcher limits native numerical-library threads, configures a writable
plot cache, and reserves CPU/GPU resources. The designated timing seed runs
alone. Progress can be controlled with `TEPPO_PROGRESS=bar`, `log`, or `off`.
For a smaller machine, limit concurrent jobs, for example:

```bash
TEPPO_CPU_CAPACITY=4 bash scripts/run_revision_pipeline.sh headline --max-workers 1
```

The launcher's CPU admission budget is the smaller of the detected capacity
and the configured cap (48 by default). Admission slots are scheduling
reservations, not a limit on process count: an SB3 job owns eight environment
workers plus its trainer. A job larger than the admission budget can run alone.
This cap and `--max-workers` do not change models or training budgets.

## Repository layout

- `src/models/`, `src/game/`, `src/sim/`: mathematical models, follower solver,
  scenario sampler, and environment.
- `src/rl/`, `src/baselines/`: training adapters, policies, and comparison methods.
- `src/evaluation/`, `scripts/`: experiment planning, evaluation, statistics,
  plotting, and artifact validation.
- `configs/`: algorithm and experiment settings.
- `tests/`: mathematical, training, and reporting checks.
- `docs/`: model and experiment specifications.

The implementation boundaries and artifact layout are described in
[ENGINEERING.md](ENGINEERING.md). [baseline.md](baseline.md) explains method
interfaces and differences. Mathematical definitions are maintained in
[docs/manuscript_revision_snippet.tex](docs/manuscript_revision_snippet.tex).

Each episode represents one scheduling cycle, with synthetic geometry,
occlusion, channel, and compute-capacity realizations. Continuous sensing
investment and one-step episodic learning are documented approximations.
This repository does not model continuous mobility or multi-RSU coordination.
Independent runs use fixed seeds; exact accelerator bitwise reproducibility
also depends on the numerical runtime and device.
