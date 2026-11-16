# DIAL-MPC G1 implementation

The project implements an **independent CPU NumPy/MuJoCo adaptation based on DIAL-MPC**, with a
G1-specific tracking objective and gait reference. It is an online optimizer,
not an RL policy; it needs no trained checkpoint. This is not the upstream H1
example, and successful walking/robustness must be established by the saved
physical runs rather than inferred from the optimizer implementation.

This is not the unchanged official code or a reproduced paper benchmark.
Our tests do not run the official optimizer side by side and do not establish
numerical or behavioral equivalence. The [provenance and differences](../mpc/UPSTREAM.md)
describe the changed gait prior, action space, costs, integration, horizon/cost
timing and numerical backend; these differences matter when interpreting results.

The current comparison treats all external impacts as **unknown to the
controller**. Its only runtime input is robot time, generalized position and
generalized velocity. Candidate rollouts contain no injected or aerodynamic
loads. The controller can react to their observed effects on robot motion;
it cannot read the force that caused that motion.

The current **calm** condition means native MuJoCo walking: aerodynamic drag
is completely disabled (`drag_model="none"`), there is no wind provider or
disturbance callback, and wind scale is zero. Gravity, actuators, contacts and
friction remain active. It is not the same condition as still air with a
surface-drag model enabled.

The fresh native calm trial and the retained moderate random-pulse trial
completed. The retained full configured vortex trial failed after 2.40 s.
The controller, initial state and seed 7 are unchanged. Current local
comparison videos and results are under
`outputs/dial_mpc/unknown_disturbances/`; the new calm source data are under
`outputs/dial_mpc/native_calm/calm/`.

| Case | Executed / requested time | Forward travel | Planar velocity RMSE | Walking check |
| --- | --- | --- | --- | --- |
| Native calm, aerodynamics off | 4.00 / 4.00 s | 0.517 m | 0.233 m/s | Passed |
| Six random joint/link pulses | 4.00 / 4.00 s | 0.510 m | 0.230 m/s | Passed |
| Full configured vortex | 2.40 / 4.00 s | 0.109 m | 0.880 m/s | **Failed** |

In the wind trial, pelvis height reached 0.440 m and crossed the 0.45 m
termination limit. The tilt limit was not crossed. Lateral drift was
1.337 m and peak modeled aerodynamic force was 25.58 N. The failure belongs
to this G1 adaptation, load-free prediction model and 64-sample/three-refinement
planning configuration; it is not a general impossibility result for DIAL-MPC.
Native calm recorded exactly zero aerodynamic and injected loads across
2,000 physics steps. Minimum pelvis height was 0.726 m and mean planning time
was 0.770 s per 0.02 s interval. The moderate-impact, large-impact and wind
trajectories are retained without rerunning or retuning. These are offline,
single-seed demonstrations, not a broad robustness or real-time benchmark.

The tracked [comparison](../mpc/assets/unknown_disturbances/comparison.mp4)
and [result bundle](../mpc/results/g1_unknown_disturbances.json) preserve the
earlier **still-air** condition, which recorded 0.042 N peak aerodynamic
drag. Its 0.502 m / 0.245 m/s calm result is historical and must not be
relabeled as native calm. Previous canonical local comparison files are
preserved in `outputs/dial_mpc/unknown_disturbances/before_native_calm/`.

The earlier [videos](../mpc/assets/comparison.mp4) and
[result bundle](../mpc/results/g1_demo.json) are **historical privileged-load
demonstrations**. Their planner could read the latest aerodynamic link
wrenches. They remain useful records of that different experiment, but do
not validate the current unknown-impact baseline.

## What the paper does

Xue et al., [Full-Order Sampling-Based MPC for Torque-Level Locomotion Control
via Diffusion-Style Annealing](https://arxiv.org/abs/2409.15610), optimize
control sequences by rolling out full robot dynamics. The objective may
include contact-dependent, nonlinear and nonsmooth terms; no dynamics
linearization is required.

Each planning cycle samples noisy candidate control sequences, evaluates
their costs, and moves the nominal sequence toward low-cost candidates:

```text
candidate[k] = nominal + noise[k]
weight[k] ∝ exp(-cost[k] / temperature)
updated_nominal = sum(weight[k] * candidate[k])
```

DIAL adds two annealing schedules to this MPPI-style update: exploration
shrinks across refinement iterations, and near-term actions receive less
noise than distant actions. Receding the horizon lets future actions undergo
many refinements before execution. The broad-to-narrow search addresses the
coverage/convergence tradeoff of fixed-noise sampling. “Diffusion-inspired”
does not mean a trained denoising network is used; simulation-based weighted
updates provide the refinement.

The official implementation uses action knots, quadratic interpolation,
sample-standardized rewards and softmax weighting. We preserve these
features. See [algorithm provenance](../mpc/UPSTREAM.md) for the pinned
upstream revision and exact numerical differences.

## How the G1 port works

- `mpc/dial.py` implements the two annealing axes and weighted update. Its
  noise standard deviation is
  `sigma * horizon_decay**(K-j) * trajectory_decay**i`, with refinement index
  `i`, knot index `j`, and final knot index `K`.
- `mpc/controller.py` evaluates candidates using threaded native MuJoCo
  rollouts of the full floating-base G1, including contacts, gravity and
  force-limited PD actuator dynamics. This is not reduced-order prediction.
- `mpc/reference.py` supplies a smooth CoM/foot reference and inverse-kinematic
  joint targets as a prior. The default optimizer searches residual targets
  on the 12 leg actuators; the remaining actuators follow reference targets.
  The model still contains all 29 actuators and full articulated dynamics.
- Costs track CoM position/velocity, torso orientation, feet and joint pose,
  while penalizing residual actions, stance-foot slip and falling. These are
  G1 adaptation choices, not an assertion that the paper used this exact cost.
- The runtime controller input is a `RobotObservation(time, qpos, qvel)`.
  It reconstructs prediction state from those values using independent
  simulation data. Only the chosen joint targets are applied to the real
  plant; its positions and velocities are never replaced by IK or predicted
  states after initialization.

The official H1 “torque” path also samples position references and computes
bounded PD torques. Our plant uses its native position-PD actuators, with
their gains, damping, force bounds and integration timestep reproduced in
the planning model. The actuator implementation/rate is a stated port
difference, not a claim that raw torque actions and position targets are
interchangeable.

The synchronous controller uses `freeze_first=False`, allowing the action
about to be executed to be optimized immediately. The optimizer also
supports upstream's frozen-first-knot convention. Both real and planning
paths apply physical actuator target limits, including spline overshoot.

## Strict unknown-disturbance interface

The planner receives only `RobotObservation(time, qpos, qvel)` at each update.
Its independent nominal model and fixed task/reference configuration are
known; the physical environment object is not retained by the controller.
The following simulator quantities are not controller inputs:

- aerodynamic or injected force/torque measurements;
- wind velocity, field parameters or access to the field provider;
- random event schedules, targets, amplitudes or future event times;
- plant solver acceleration warm starts or other solver-internal state.

Prediction starts with a zero solver warm start and zero external loads.
The prior action sequence is still shifted and refined between MPC updates;
this optimizer warm start is distinct from the excluded physics-solver
warm start. The former `wind_prediction="current_wrench"` mode is removed;
old configurations requesting it are rejected rather than silently granted
force access.

For the disturbed cases, the physical plant integrates surface drag and/or
random loads at every 0.002 s step. Native calm disables both completely.
The experiment runner logs loads, and the renderer uses them for drag
visualization when aerodynamics is enabled. Logging and drawing a force
does not feed it to the planner. With unknown wind, the planner uses a
mismatched load-free prediction model and corrects its actions after the
robot state changes.

`qpos` and `qvel` currently come from noiseless simulation state. They include
floating-base pose and velocity, so this remains an ideal state-estimation
assumption. A physical robot needs an estimator to supply those quantities.
Unknown external loads do not imply that all state-estimation and sim-to-real
issues have been solved.

## Run and inspect physical cases

From the repository root with the existing project environment:

```powershell
.\.venv\Scripts\python.exe -m mpc.run --cases calm random tornado --duration 4 --controller-config mpc/configs/g1_demo.json --wind-scale 1 --disturbance-strength 0.2 --video --output outputs/dial_mpc/new_run
.\.venv\Scripts\python.exe -m mpc.compare outputs/dial_mpc/new_run/calm outputs/dial_mpc/new_run/random outputs/dial_mpc/new_run/tornado --output outputs/dial_mpc/new_run/comparison.mp4
```

Each case writes `summary.json`, `planner.json`, `trajectory.npz` and a model
snapshot; `--video` renders the recorded physical states. Use a new output
directory for each experiment. The renderer does not impose a successful
trajectory or hide a failure. Native calm renders an opaque robot without
force arrows. In the other cases, surface drag arrows represent aerodynamic
forces only, with 50% robot opacity.

Useful arguments are `--controller-config <JSON>`, `--seed`, `--wind-scale`
(multiplies velocity), `--disturbance-strength` and `--surface-patches`.
Controller settings include samples, horizon, refinement iterations, knot
count, action scale, active actuators and cost weights. External load access
is excluded by the observation interface, not selected as a tuning option.
The random-case default uses 20% of the 150 N/20 N m/30 N m configured maxima;
report the actual saved strengths/events rather than calling every run a
large-disturbance test.

Summaries report completed duration, physical progress, foot lift/landing
events, velocity error, failure and planner wall time. Passing the duration
alone does not count as walking. The initial project walking check requires
forward progress, at least one landing per foot and low velocity RMSE. It is
a diagnostic criterion, not a validated locomotion benchmark definition.

## Limits and comparison protocol

This implementation runs **synchronously and offline on CPU**: simulated time
waits for planning. Compare measured planner latency with the 0.02 s control
interval before making any real-time claim. The upstream paper's GPU results
do not establish this port's timing or performance.

Compare calm, random-load and field conditions using the same model,
controller configuration and initial state. Keep failure cases. The surface
pressure model is approximate and the terrain is currently flat; these runs
do not establish CFD fidelity, rescue-terrain performance or sim-to-real
transfer. Later comparison with RL must also match actuation and disclose
the ideal full-state estimation used here. All external impacts remain
unknown to this controller.
