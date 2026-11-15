# DIAL-MPC G1 implementation

The project now implements a **CPU NumPy/MuJoCo port of DIAL-MPC**, with a
G1-specific tracking objective and gait reference. It is an online optimizer,
not an RL policy; it needs no trained checkpoint. This is not the upstream H1
example, and successful walking/robustness must be established by the saved
physical runs rather than inferred from the optimizer implementation.

The saved [three-case demonstration](../mpc/assets/comparison.mp4) completed
4 s of physical walking in calm, random-pulse and full configured vortex
conditions, with 0.509/0.502/0.495 m forward travel respectively. The same
controller configuration, initial state and seed were used. The random case
reached 27.57 N injected link force, and the vortex reached 26.48 N integrated
aerodynamic force. [Recorded results](../mpc/results/g1_demo.json) preserve
the actual event schedule and metrics. These short single-seed successes do
not establish a broad robustness benchmark. CPU planning averaged
1.28-1.45 s per 0.02 s control interval.

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
- Planning state is copied from the actual plant. Candidate trajectories use
  independent simulation data. Only the chosen joint targets are applied to
  the real plant; its positions and velocities are never replaced by IK or
  predicted states after initialization.

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

## Disturbance information and prediction

Two planner modes are supported:

| Setting | Prediction assumption |
| --- | --- |
| `wind_prediction="none"` | No aerodynamic load in candidate rollouts; feedback observes its effects through the current state. |
| `wind_prediction="current_wrench"` | Hold the most recently measured aerodynamic force/torque on every body constant over the horizon. |

The second mode is explicitly privileged simulated load measurement. It is
not a forecast of the wind field: actual surface drag changes with position,
orientation and body velocity. Neither mode queries future random impulses,
and neither has a future-event oracle. The plant applies aerodynamic and
random loads at every 0.002 s physics step.

## Run and inspect physical cases

From the repository root with the existing project environment:

```powershell
.\.venv\Scripts\python.exe -m mpc.run --cases calm random tornado --duration 4 --controller-config mpc/configs/g1_demo.json --disturbance-strength 0.2 --video --output mpc/outputs/comparison
.\.venv\Scripts\python.exe -m mpc.compare mpc/outputs/comparison/calm mpc/outputs/comparison/random mpc/outputs/comparison/tornado --output mpc/outputs/comparison/comparison.mp4
```

Each case writes `summary.json`, `planner.json`, `trajectory.npz` and a model
snapshot; `--video` renders the recorded physical states. Use a new output
directory for each experiment. The renderer does not impose a successful
trajectory or hide a failure. Surface drag arrows represent aerodynamic
forces only, with 50% robot opacity.

Useful arguments are `--controller-config <JSON>`, `--seed`, `--wind-scale`
(multiplies velocity), `--disturbance-strength` and `--surface-patches`.
Controller settings include samples, horizon, refinement iterations, knot
count, action scale, active actuators, cost weights and wind prediction.
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
full-state/current-wrench information advantages.
