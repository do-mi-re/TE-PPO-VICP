# Implementation contracts

Experiment settings are defined by `configs/revision.yaml` and
`docs/revision_experiment_protocol.md`. Model definitions are maintained in
`docs/manuscript_revision_snippet.tex` and deterministic module docstrings.

## Environment and learning

`RSULeaderEnv.step` executes leader prices and bandwidth, analytical follower
responses, and raw RSU utility. `step_joint` executes the joint decisions learned
by JDLS/MAPPO and the learned-follower ablation. Both methods use the same
physical model and evaluation metrics.

Candidate masks are episode-static. Bandwidth decoding assigns exactly zero
to vehicles excluded from every task, and the analytical solver receives only
admitted vehicle-task pairs. The joint decoder maps sensing effort into the
known latency-feasible interval; it does not optimize follower utility.

TE-PPO's adapter adds the configured failure, latency, and participation
penalties to raw RSU utility. Joint-learning methods use the shaping terms
specified in their YAMLs and `src/baselines/joint_learning_reward.py`.
Evaluation reports raw RSU utility and social welfare separately from shaping.
Training uses lean information dictionaries; full evaluation provides utility,
constraint, and task-level details with the same physical rewards.

## Artifacts

Formal runs use the roots declared in the experiment YAML. A training directory
contains `config_snapshot.yaml`, `config_fingerprint.txt`, `selection.json`,
`selected_model.zip` or `selected_model.pt`, evaluation arrays, summary,
training-cost records, and runtime metadata. SB3 policies additionally require
the same-step `selected_vecnormalize.pkl`.

Evaluation checks the expected resolved configuration, snapshot, fingerprint,
checkpoint hashes, normalization hashes, and selection rule. Matching outputs
may be reused; incompatible or partial outputs are rejected. Evaluation
contracts and source manifests bind derived reports to their inputs.

The selected checkpoint maximizes validation social welfare with earliest-tie
selection. CPU-domain policies instead use the declared maximin validation rule.
Last-step weights and periodic training checkpoints are operational outputs;
reported results use selected checkpoints only.

## Execution

The scheduler accounts for CPU admission slots and GPU memory; native thread
pools are limited by the launcher. The specified timing seed is isolated.
Hardware records contain CPU, RAM, GPU, runtime, and configuration identities.
Use `PYTHONPATH=.` for scripts and run commands from the repository root.
`run.py` provides shared orchestration helpers; experiment commands are exposed
by `scripts/run_revision_pipeline.sh` and the individual scripts.

## Exactness and limitations

Deterministic model formulas implement the maintained equations. The follower
multiplier uses numerical bisection of the stationary residual followed by a
feasible sensing-duration projection. Scenario distributions, continuous
sensing investment, and one-cycle episodes are documented approximations.
Scheduling, caching, statistics export, and plotting have no model semantics.

The canonical experiment uses fixed hyperparameters and seed banks. Changing
model or training behavior requires a training-semantics version decision and
new matching artifacts. Statistical and numerical checks do not establish
full Stackelberg-equilibrium existence or policy optimality.
