# Project authority and scope

Implement and maintain a mathematically traceable research codebase for
"Incentive-Aware Resource Allocation for Energy-Efficient Cooperative
Perception: A Stackelberg Game Approach".

The maintained sources of truth are:

- `configs/revision.yaml` for the experiment contract;
- `docs/revision_experiment_protocol.md` for statistical and orchestration rules;
- `docs/manuscript_revision_snippet.tex` for the bounded sensing model;
- the tested deterministic modules under `src/models/`, `src/game/`, and `src/sim/`.

Only the maintained sources above are implementation authority.

## Core principles

- Prioritize the maintained mathematical specification over convenience.
- Do not silently invent missing mathematical details.
- Record unresolved model questions in
  `docs/revision_experiment_protocol.md` or a focused document under `docs/`.
- Separate deterministic mathematical logic from RL/training logic.
- Preserve the two-stage environment semantics: the leader sets price and
  bandwidth, then followers solve their own decision problem.
- Every nontrivial formula must be traceable to a maintained equation or an
  explicit assumption in its docstring/comment.
- Prefer small pure functions, config-driven experiments, reproducible seeds,
  and tests for every nontrivial module.

## Exactness labels

Every major module must state one of:

- `paper-exact`: implements a maintained manuscript equation directly;
- `paper-consistent approximation`: fills an explicitly documented gap;
- `implementation convenience only`: orchestration or tooling with no model semantics.

## Before substantial semantic changes

1. Restate the equations or definitions affected.
2. List assumptions and any missing details.
3. Define verification tests.
4. Confirm whether the change increments `training_semantics_version` and
   invalidates trained checkpoints.

Do not proceed through an unresolved mathematical ambiguity. Document it,
propose one or two options with downstream effects, and ask the user.

## Implementation and testing order

For a new mathematical feature, update the maintained specification first,
then deterministic math, follower solver, scenario/environment, RL wrapper,
training, baselines, and finally experiment/reporting code. Each changed layer
must have passing basic tests before work proceeds to the next layer.

## Deliverables

For completed semantic work, report files changed, formulas implemented,
assumptions, tests, and remaining ambiguities. For code-quality-only work,
state explicitly that mathematical and training semantics were unchanged.

## Coding style

- Python with practical type hints and clear module boundaries.
- No hidden global mutable state.
- Prefer readability over framework cleverness; avoid overengineering.
