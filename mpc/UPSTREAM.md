# DIAL-MPC provenance and adaptation limits

This project contains an **independently implemented G1 adaptation of
DIAL-MPC**. It is not the official implementation running unchanged, a
numerically verified equivalent port, or a reproduction of the paper's
benchmarks. Here, "port" means that we implemented selected algorithmic
operations in a different software stack and built a different robot
controller around them. Our G1 results measure this adaptation only.

The algorithm reference is LeCAR-Lab's repository at commit
[`871c84f1cefbe9fa9d83a06b05ab4f0243c1a05c`](https://github.com/LeCAR-Lab/dial-mpc/tree/871c84f1cefbe9fa9d83a06b05ab4f0243c1a05c).
The upstream Apache-2.0 license is retained in [LICENSE-DIAL-MPC](LICENSE-DIAL-MPC)
for attribution; it does not replace the separately licensed G1 model assets.
The method is described by Xue et al. in
[Full-Order Sampling-Based MPC for Torque-Level Locomotion Control via Diffusion-Style Annealing](https://arxiv.org/abs/2409.15610).
The paper's reported performance is not evidence of this implementation's
performance.

## Algorithm correspondence established by source inspection

The pinned [optimizer source](https://github.com/LeCAR-Lab/dial-mpc/blob/871c84f1cefbe9fa9d83a06b05ab4f0243c1a05c/dial_mpc/core/dial_core.py)
uses quadratic action-knot interpolation, Gaussian perturbations, an appended
incumbent candidate, clipped knots, standardized rewards and a softmax-weighted
knot update. Its executed refinement loop has noise standard deviation

```text
std[i, j] = sigma_scale * horizon_diffuse_factor**(K-j)
                       * traj_diffuse_factor**i
```

For finite costs `C = -reward` above our negligible-spread threshold, the
corresponding formulas are

```text
weights = softmax(-(C - C_incumbent) / std(C) / temperature)
new_knots = sum(weights[n] * candidate_knots[n])
```

Our `noise_schedule` and `cost_weights` implement these algebraic operations.
Subtracting a common finite baseline does not change the normalized weights.
This correspondence is narrower than equivalence of the complete optimizer
or closed-loop controller: input trajectories, precision, samples and evaluated
objectives differ. There is no learned diffusion network in this controller.

## Substantive implementation differences

| Area | What this project changes or has not verified |
| --- | --- |
| Execution stack | `mpc/dial.py` uses NumPy/SciPy and `controller.py` uses threaded native MuJoCo rollouts. The upstream path uses JAX, jax-cosmo and Brax/MJX. We have not run both implementations on identical inputs or established matching dynamics, outputs or speed. CPU simulation waits for optimization; no real-time claim is made. |
| Precision and randomness | Our arrays and SciPy interpolation normally use NumPy float64; sampling uses `numpy.random.default_rng`. Upstream uses JAX arrays and split PRNG keys. Its effective precision depends on JAX configuration. Equal integer seeds do not imply equal samples, and numerical equivalence has not been established. |
| Interpolation | We use SciPy `make_interp_spline(k=2)` with its default boundary behavior; upstream uses jax-cosmo `InterpolatedUnivariateSpline(k=2)`. Matching degree and grid counts do not prove identical boundary conditions or off-knot values. Our quadratic-polynomial test does not compare the two libraries. |
| Clipping order and exceptional batches | We clip the initial mean before sampling; upstream perturbs it before clipping candidates. This can matter after shifting/interpolation overshoot. Our zero-spread fallback retains the incumbent, invalid costs get zero weight, and an entirely invalid batch raises. The G1 evaluator separately maps failed rollouts to a finite large penalty. |
| Receding horizon | Upstream shifts interpolated controls, fills the tail with normalized zero, then interpolates back to knots. We implement that sequence with a different spline and additionally allow multiple-step shifts/nonzero tails. In the G1 controller, zero means zero **residual around a moving reference**, not an absolute joint target. |
| Action timing | Upstream freezes the first knot and its example loop advances the plant before planning. Our G1 configuration explicitly sets `freeze_first=false` and plans before applying the first action, including at initialization. This changes the closed-loop algorithm. The standalone optimizer still defaults to a frozen first knot. |
| Prediction horizon and cost timing | Upstream rolls out all `Hsample+1` interpolated controls. Our G1 evaluator uses only the first `horizon_steps` controls of its `H+1` array. Its CoM/foot/orientation terms use interval-start sensors; joint tracking uses interval-end positions. These are not identical horizon or reward-indexing conventions. |

The original operations above can be inspected in the pinned
[core implementation](https://github.com/LeCAR-Lab/dial-mpc/blob/871c84f1cefbe9fa9d83a06b05ab4f0243c1a05c/dial_mpc/core/dial_core.py).
For an otherwise identical fixed-length objective, replacing a sum by its
mean is canceled by standardization. That algebra does **not** make different
horizons, state-time indices or cost terms equivalent.

## G1 controller changes

- **Robot and action space:** the local model has all 29 G1 actuators and
  full floating-base/contact dynamics. The default search adjusts only the
  12 leg targets, using normalized residuals scaled by 0.2 radians around
  a time-varying IK reference. Other joints follow reference targets.
  The official H1 controller maps actions through its own joint ranges;
  this project does not use that H1 action parameterization.
  [H1 environment](https://github.com/LeCAR-Lab/dial-mpc/blob/871c84f1cefbe9fa9d83a06b05ab4f0243c1a05c/dial_mpc/envs/unitree_h1_env.py)
- **Reference and objective:** `mpc/reference.py` adds a bounded LIP CoM
  trajectory, alternating foot trajectories and precomputed G1 IK.
  Our costs track CoM, velocity, torso orientation, foot position and joint
  pose, and penalize residual actions, foot slip and falling. Their weights,
  startup, gait timing and failure criteria are project choices. Upstream
  H1 rewards use its own gait/contact-height, velocity, uprightness, yaw,
  height and torque terms. This is a different control problem.
  [H1 reward definition](https://github.com/LeCAR-Lab/dial-mpc/blob/871c84f1cefbe9fa9d83a06b05ab4f0243c1a05c/dial_mpc/envs/unitree_h1_env.py#L207-L305)
- **Actuation and integration:** upstream's H1 `leg_control: torque` path
  maps normalized actions to joint targets, then computes and clips
  `kp*(q_target-q)-kd*qdot` before `pipeline_step`. Its H1 jog example uses
  both control and physics intervals of 0.02 s. Our G1 instead holds
  position targets for 0.02 s while native force-limited position-PD
  actuators act at 0.002 s, with the model's `implicitfast` integrator.
  Gains, force limits and joint ranges are G1-specific. Sharing the idea of
  PD tracking does not establish equivalent actuator dynamics.
  [BaseEnv action conversion](https://github.com/LeCAR-Lab/dial-mpc/blob/871c84f1cefbe9fa9d83a06b05ab4f0243c1a05c/dial_mpc/envs/base_env.py#L34-L61),
  [H1 stepping](https://github.com/LeCAR-Lab/dial-mpc/blob/871c84f1cefbe9fa9d83a06b05ab4f0243c1a05c/dial_mpc/envs/unitree_h1_env.py#L171-L181)
- **Unknown-disturbance interface:** our runtime input is only
  `RobotObservation(time, qpos, qvel)`. Prediction uses an independent model
  with zero external loads and zero solver acceleration warmstart. It
  receives neither force measurements nor field/event parameters. The
  shifted previous action plan remains available. This is a project-specific
  information boundary, not a reproduction of an upstream robustness
  experiment. Pose and velocity currently assume ideal state estimation;
  known model parameters and task/reference settings are supplied separately.
- **Scene and evaluation:** our flat 10 m test area, surface-drag model,
  random pulse schedule and walking checks are local benchmark choices.
  Their results cannot validate the paper's hardware, model-mismatch or
  terrain results.

For a concrete budget comparison, the pinned
[H1 jog configuration](https://github.com/LeCAR-Lab/dial-mpc/blob/871c84f1cefbe9fa9d83a06b05ab4f0243c1a05c/dial_mpc/examples/unitree_h1_jog.yaml)
and [base optimizer defaults](https://github.com/LeCAR-Lab/dial-mpc/blob/871c84f1cefbe9fa9d83a06b05ab4f0243c1a05c/dial_mpc/core/dial_config.py)
differ from our [G1 demonstration settings](configs/g1_demo.json):

| Setting | Upstream H1 jog example | Our G1 demonstration |
| --- | --- | --- |
| Noisy candidates per refinement, excluding incumbent | 2048 | 64 |
| Refinements per cycle / initialization | 4 / 10 | 3 / 6 |
| Dense horizon parameter / knot intervals | 25 / 5 | 20 / 5 |
| Noise scale | 1.0 (inherited default) | 0.5 |
| Horizon / iteration decay | 1.0 / 0.5 | 0.9 / 0.4 |
| Temperature | 0.05 | 0.05 |

These are distinct configurations, not a matched comparison; the horizon
interpretation also differs as described above. Other upstream examples
have other settings. The standalone `DialConfig` defaults are also distinct
from the settings used by `DIALController` and the saved G1 configuration.

## What the existing tests establish

[Optimizer unit tests](tests/test_dial.py) check the explicit annealing and
weight formulas, a quadratic polynomial and shift endpoints, incumbent
inclusion, first-knot freezing, local seed repeatability, a smooth toy
objective and exceptional-cost handling. They do not import or execute the
upstream implementation. They are not cross-implementation equivalence tests.

[Controller tests](tests/test_controller.py) check four control intervals
against this project's own native G1 plant from a static test posture,
target clipping, selected physical model parameters/derived masses, and
the exclusion of privileged force/solver inputs at a fixed observation and
seed. The walking reference is mocked in these tests. They do not establish
MJX parity, end-to-end gait equivalence, contact robustness across states,
hardware behavior or reproduced paper performance.

Accordingly, "same update formula" is supported within the finite-cost,
nondegenerate-spread scope above. "Unchanged official DIAL-MPC," "exact numerical
port," "paper-reproduced baseline," and "same robustness as the paper" are
unsupported descriptions of this code. Existing comments using "official,"
"matches upstream," or "exactly as upstream" must be read as references to
the named local operation, not claims of complete equivalence.
