# Methods and comparison baselines

All methods use the physical scenario and metrics declared in the maintained
model and experiment protocol. Their settings are explicit in
`configs/algorithms/` and materialized by `src/evaluation/revision.py`.

| Method | Learned decisions | Follower execution |
| --- | --- | --- |
| TE-PPO (`proposed`) | RSU prices and bandwidth | Analytical best response |
| JDLS | Centralized leader and vehicle decisions | Learned categorical task and continuous effort/compression |
| MAPPO (`mappo`, figure label MAPS) | Leader actor and shared vehicle actor | Learned task and continuous effort/compression |
| EBAS | Prices | Analytical best response with equal bandwidth |
| SPIS | Bandwidth | Analytical best response with fixed price 0.15 |
| RDUS | Prices and bandwidth | Analytical response with compression fixed at 1 |
| DBAS | None | Distance-based leader allocation and analytical response |

## Leader-only methods

Prices are decoded as `p_max * sigmoid(z_p)`. Bandwidth is decoded as utilization
`sigmoid(z_u)` times softmax shares over eligible vehicles. TE-PPO enables
candidate selection; comparison methods use their declared candidate settings.
EBAS forces equal shares, and SPIS supplies its configured fixed price.
DBAS calibrates one price on the validation bank; it never trains a policy.
RDUS additionally declares its minimum raw service-data requirement in the
scenario YAML; it is not a pure compression-only control.

## Joint-learning methods

JDLS uses a centralized Gaussian/MultiCategorical actor. MAPPO uses a leader
Gaussian actor and a shared hybrid vehicle actor with centralized value
estimation. Policies start from seeded random initialization, without analytical
teacher samples. Task support follows the declared candidate mask; compression
uses the multiplicative bounded mapping, and effort is projected into the
known latency interval. These constraints do not constitute an analytical
best-response optimization. The environment executes the resulting JointAction
through `step_joint`.

The joint shaping configuration uses `omega_vehicle=0.001`,
`omega_feasible=1`, `omega_pre_quality=0.01`, and `omega_pre_lat=0.01`.
Direct effort and bandwidth shaping coefficients are zero. Missing latency
receives the configured capped overshoot. These training signals are distinct
from raw RSU utility, social welfare, and the common evaluation reward.

The shaped reward is

```text
R_joint = U_RSU
        + omega_vehicle * raw_assigned_utility_sum
        + omega_feasible * num_feasible_assignments
        + omega_pre_quality * pre_filter_quality_log
        + omega_tau * tau_nonzero_reward
        + omega_bandwidth_use * bandwidth_use_reward
        - omega_pre_lat * pre_filter_latency_overshoot
        - penalty_fail - penalty_lat - penalty_part
```

Pre-filter terms use the assigned task before the participation filter:
finite conditional utilities, finite-utility feasible-assignment counts, summed
`ln(1+max(quality,0))`, and capped deadline-relative latency overshoot.
The direct effort term sums sensing durations divided by the maximum task
deadline, clipped to `[0,1]` per vehicle; the direct bandwidth term sums
shares clipped to `[0,1]`. Common penalties use accepted follower outcomes
and the weights in the experiment protocol. The pure implementation and
diagnostic field definitions are in `src/baselines/joint_learning_reward.py`.

## Controlled experiments

The ablation arms remove candidate selection, dynamic pricing, adaptive
bandwidth allocation, or the online analytical response according to their materialized
configs. The learned-follower arm uses the JDLS engine and its documented
shaping/normalization configuration; its difference is a complete alternative
training path, not an isolated causal estimate of the analytical solver.

Core budget, learning rate, clipping range, and network widths are shared.
Engine-specific rollout, minibatch, epoch, and normalization settings remain
explicit rather than being treated as interchangeable. See the experiment
protocol for seed aggregation and held-out evaluation.
