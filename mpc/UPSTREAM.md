# DIAL-MPC algorithm provenance

This is a NumPy/SciPy numerical port of the sampling optimizer in
[LeCAR-Lab/dial-mpc](https://github.com/LeCAR-Lab/dial-mpc), pinned to commit
`871c84f1cefbe9fa9d83a06b05ab4f0243c1a05c`. The upstream Apache-2.0 license is
preserved in `LICENSE-DIAL-MPC`. The relevant source files are:

- [`dial_mpc/core/dial_core.py`](https://github.com/LeCAR-Lab/dial-mpc/blob/871c84f1cefbe9fa9d83a06b05ab4f0243c1a05c/dial_mpc/core/dial_core.py)
- [`dial_mpc/core/dial_config.py`](https://github.com/LeCAR-Lab/dial-mpc/blob/871c84f1cefbe9fa9d83a06b05ab4f0243c1a05c/dial_mpc/core/dial_config.py)
- [`dial_mpc/envs/base_env.py`](https://github.com/LeCAR-Lab/dial-mpc/blob/871c84f1cefbe9fa9d83a06b05ab4f0243c1a05c/dial_mpc/envs/base_env.py)

The method is described in Xue et al.,
[Full-Order Sampling-Based MPC for Torque-Level Locomotion Control via
Diffusion-Style Annealing](https://arxiv.org/abs/2409.15610), ICRA 2025.

## Optimizer behavior preserved

The trajectory has `Hnode+1` normalized action knots and quadratic-spline
interpolation to `Hsample+1` controls. Candidate knot noise has standard
deviation `sigma * horizon_decay**(Hnode-j) * trajectory_decay**i`, where `j`
is the knot index and `i` the refinement iteration. Thus later actions have
more exploration and successive iterations have less noise. Each batch also
contains the incumbent trajectory. Knots are clipped to [-1,1]; the optional
first-knot freeze matches upstream's pipelined control timing.

Full-dynamics trajectory costs are converted to standardized rewards and
softmax weights. The updated knots are the weighted mean of candidates.
The next planning cycle shifts the dense control sequence, appends a zero
terminal action, and fits the new knots. This includes both annealing axes;
it is not fixed-noise MPPI relabeled as DIAL-MPC.

## Explicit port differences

- NumPy/SciPy replace JAX/jax-cosmo; the caller provides native MuJoCo batched
  rollouts instead of Brax/MJX. This CPU execution path makes no real-time claim.
- The callback returns one cost per trajectory rather than per-step rewards.
  Mean and summed costs differ by a constant horizon factor, canceled by the
  upstream-style sample standardization for fixed horizons.
- Constant-cost batches retain the incumbent, avoiding upstream's zero
  standard-deviation division. Nonfinite trajectories receive zero weight;
  an entirely invalid batch raises an error.
- `freeze_first=False` is an explicit synchronous zero-delay option. The
  default remains the official first-knot-freezing behavior.
- A nonzero terminal action can be selected explicitly by the caller.

Quadratic interpolation can exceed the clipped knot bounds. The real plant
and the rollout decoder must enforce the same physical actuator limits.

## What upstream means by torque control

The official H1 examples set `leg_control: torque`, but the optimizer samples
normalized joint-position references. `BaseEnv.act2tau` converts each
reference to bounded torque using `kp*(q_target-q)-kd*qdot`. Therefore a G1
adaptation can optimize position references while modeling its force-limited
PD actuators. It must state that adaptation, reproduce the actual actuator
dynamics in planning, and must not relabel raw torque arrays as angles.

The upstream humanoid is H1, not G1. G1 objectives, action parameterization,
gait guidance and simulation settings remain project-specific adaptations
and must be reported separately from the optimizer's provenance.
