# Experiment protocol

The experiment authority is `configs/revision.yaml`, algorithm YAMLs, the
maintained model specification, and deterministic model tests. All quantities
use SI units unless a plotted axis explicitly rescales them.

## Model and assumptions

Each episode is one scheduling cycle. Synthetic scenario geometry, sensor
resolution, occlusion, channel power, and CPU capacity are sampled using the
configured seed. Vehicle-to-task distance is the shortest distance to the
rectangular RoI with a 1 m floor. Sensing potential saturates at 50; the sensing
reference range is 80 m. Occlusion enters through `(1-O)`.

The single-step example uses the same rectangular-RoI distance, sensing bound,
price bound, and candidate-cap definition. Its manually specified three-vehicle,
two-task geometry illustrates one scheduling cycle; it is not an experiment
replicate. Sanity sweeps use this illustrative geometry within the same bounds.

Continuous sensing effort has a zero lower bound for the canonical model and
an upper bound determined by the task deadline, computation, and transmission.
Compression lies in `[1,50]`. RDUS declares a minimum raw service-data constraint
in its own configuration. Each vehicle chooses at most one task or does not
participate. The leader commits prices and bandwidth before analytical
follower responses. Joint-learning baselines execute their own decisions
without the online best-response optimizer.

The follower latency multiplier is a numerical approximation: bisection uses
the unclipped stationary sensing-duration residual, and the resulting duration
is projected into the feasible interval. An unbracketable multiplier yields an
infeasible response. Exhausted bisection iterations return the midpoint with a
warning. This numerical procedure is not a proof of equilibrium existence.

## Candidate selection

`sensing_range_m` sets sensing-quality normalization;
`candidate_pruning_radius_m` sets admission distance;
`candidate_limit_range_m` sets the candidate-cap coverage scale.

For task span `a_k`,
`S_k=max(1,ceil(a_k/(2*r0)))` and `N_max(k)=max(2,ceil(rho*S_k))`.
Admitted candidates rank by descending bounded potential, then raw potential,
then ascending vehicle ID. Raw potential is required for saturation ties.
The full method uses 80 m and `rho=4`. Only admitted tasks are passed to the
analytical follower solver. Leader bandwidth normalization uses the union of
candidate vehicles; excluded vehicles receive zero bandwidth.

Pruning sensitivity varies admission and cap scales while keeping sensing
quality at 80 m. Each distinct effective cell has eight independently trained
policies at 600k steps. The canonical cell reuses headline policies. Under the
configured geometry, `rho=1` and `rho=2` both yield cap 2 and share a training
identity. All reported cells remain in the table; the plot groups those exact
equivalents. No interpolation or smoothing is used.

## Seeds, budgets, and checkpoint selection

An independent replicate is an independently trained policy, not an individual
evaluation episode. Training, validation, and all held-out scenario banks are
explicit in the YAML and mutually separated where declared. Validation selects
checkpoints and calibrates DBAS. Held-out scenarios never train or calibrate a
policy. The headline and CPU experiments use eight training seeds; ablation and
shape-changing scalability use five. The configured held-out banks determine
the complete Cartesian products required for each report.

All stochastic methods use 600000 physical environment steps, hidden widths
`[256,256]`, learning rate `3e-4`, and clip range `0.2`. PPO engines retain the
explicit engine-specific epochs, minibatches, and rollout sizes in their YAMLs.
Validation runs every 3072 physical environment steps and at the terminal step.
The selected checkpoint maximizes mean validation social welfare; exact ties
retain the earlier checkpoint. Selected SB3 weights and normalization state
must come from the same step. Common price bound is `p_max=0.3`.

DBAS uses validation-only fixed-price calibration over the configured price
grid. Headline candidates must have mean task-success rate at least 0.75 and
positive mean selected-vehicle count over the calibration scenarios. Among
eligible candidates, selection maximizes mean social welfare, then mean raw
RSU utility, then prefers the lower price. If no candidate is eligible,
calibration fails. These are calibration constraints, independent of the
held-out comparison between methods. SPIS uses the fixed configured price.
Calibration outputs, configurations, and checkpoint hashes are checked before
evaluation.
Calibration contracts bind the exact validation seeds, price grid, resolved
DBAS base configuration, and calibration implementation. CPU calibration also
binds its CPU bands and experiment settings. A changed contract requires
recalibration, including when only source comments change the recorded source
digest. This affects DBAS calibration artifacts, not learned checkpoints.

## Metrics and statistics

Raw RSU utility, social welfare, quality, energy, participation, and task success
are separate metrics. Completion is `I(Q_k>0)` and is reported overall and by
high/mid/low urgency. Low-urgency noncompletion remains a reported outcome.
Training shaping is not social welfare. Common evaluation reward is
`R_common=U_RSU-P_fail-P_lat-P_part`, with weights `(2,3,3)` and minimum one
participant. It is recorded under an explicit schema by every training engine;
it does not replace social welfare for checkpoint selection.

For replicate-level means `x_s`,
`std_seed=sqrt(sum((x_s-mean(x))**2)/(n-1))` and
`CI95=1.96*std_seed/sqrt(n)`. Curves use mean +/- seed standard deviation;
point figures use the declared 95% interval. Scenario observations are first
averaged within each training seed. Percentage deltas are formed within matched
training seeds, then aggregated. Significance uses an exact two-sided paired
sign-flip test. A zero denominator or missing pair prevents the corresponding
percentage statistic; insufficient independent replicates cannot support a
significance claim.

Energy per participating vehicle is
`sum_episode(E_total)/sum_episode(N_selected)` within each training seed.
A zero participant count is non-estimable, not zero energy. Unconditional
energy, service success, and participant counts remain reportable.

Convergence thresholds use
`T_alpha=W_ref+alpha*(W_final-W_ref)` for alpha 0.8, 0.9, and 0.95, with the
final 10% of evaluations defining `W_final`. No positive progress yields no
threshold crossing and is disclosed through the contributing-seed count.

## Robustness and mechanism analyses

Shape-changing V/K cells train matching observation/action shapes with five
common seeds; evaluation averages over 32 scalability scenarios within each
seed. Dynamics replay uses five canonical policies and 32 separate scenarios.

CPU policies train on a uniform mixture of five configured capacity intervals.
Each learning method has one policy per training seed, evaluated at all fixed
CPU intervals. Validation selects by maximin CPU-band social welfare, with the
declared mean/timestep tie-breaking. DBAS uses one calibrated price across the
entire CPU grid. The CPU policy is not retrained separately for each point.
CPU DBAS candidates must meet the same task-success and participation thresholds
in every validation CPU band. Selection maximizes minimum band social welfare,
then macro mean social welfare, then macro mean raw RSU utility, then prefers
the lower price. Calibration fails if no candidate satisfies all bands.

Main-text analyses use selected policies and the configured independent scenario
bank. Price sensitivity holds learned bandwidth fixed, rescales prices, clips
them to `[0,p_max]`, recomputes analytical responses, and reports clipping.
Compression-energy analysis holds selected vehicles, sensing durations,
bandwidth, channels, and compute capacity fixed while varying compression.
It is a component counterfactual, not necessarily a feasible reoptimized policy.

## Timing and reporting

Inference measures a complete scheduling cycle, with the matching shape's
selected checkpoint and normalization. Use 20 warmups and 200 repetitions,
accelerator synchronization, the configured seed, and isolated execution.
Record hardware, runtime, configuration, checkpoint, and normalization
identities. Contended training duration is operational cost, not isolated
inference latency. Report the current training records only.
Training-cost tables accept schema-2 records with matching fingerprints and
finite positive elapsed durations. Their total hours sum per-run wall times;
they are neither CPU consumption nor the elapsed wall time of concurrent jobs.
Duration statistics group only runs with the same algorithm, vehicle/task
counts, purpose, budget, hardware identity, and execution mode. The algorithm's
share of aggregate run time sums all of its groups before computing the share.

Formal figures retain source CSVs and digest manifests. The paper subset and
full comparison set are explicit in the YAML. Do not smooth, monotonicize,
interpolate pseudo-observations, or substitute shaped reward for welfare.

## Execution and acceptance

Run preflight, DBAS calibration and headline training/evaluation, matched
pruning sensitivity, CPU training/evaluation, ablation, scalability, dynamic
and mechanism evaluation, isolated timing, then reports and final validation.
The launcher exposes these stages. No separate decision file is required to
start a final experiment.

Validation checks complete grids and seed sets, matching configurations,
calibration records, selected checkpoints, normalization, metric definitions,
statistical reduction, source manifests, and hardware/timing records. It does
not require the proposed method to beat a baseline. Missing or mismatched inputs
produce an error instead of a partial report presented as complete.
