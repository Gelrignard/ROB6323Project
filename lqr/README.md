# LQR for reduced biped walking

This directory extends the finite-horizon Riccati recursion from
`../../hw2/lq_solvers.py` to reference tracking with known affine disturbance
increments and, in the general solver, time-varying matrices. The controller
acts on a **constant-height linear inverted pendulum**, with prescribed
footsteps supplied by `sim_env`. It is not a joint-torque controller for a complete
humanoid.

`python -m lqr --viewer` now uses the Unitree G1 preset by default: measured
G1 mass, a reachable crouched CoM height, conservative G1 sole dimensions,
and articulated inverse-kinematics replay. Actual G1 contact physics is
available separately through `python -m sim_env.g1_physics --viewer`.
The remaining controller work is to connect CoM/foot references to physical
whole-body tracking; an IK animation is not validation of that layer.

The state is `x = [cx, vx, cy, vy]` and the action is the center of pressure
`u = [zx, zy]`. With horizontal wind force at the center of mass,

```
c_ddot = (g/h) (c - u) + F_w/m.
x[t+1] = A x[t] + B u[t] + E F_w[t].
```

`lip_discrete_dynamics` uses exact zero-order-hold discretization. Aerodynamic
force may still change with velocity in the simulation; a force supplied to
the LQR prediction is held constant on each controller interval.

## Connection to homework 2

The cost is

```
J = 1/2 sum((x-x_ref)' Q (x-x_ref) + (u-u_ref)' R (u-u_ref))
    + 1/2 (x_N-x_ref_N)' Qf (x_N-x_ref_N).
```

Expanding the references gives the linear terms used in the homework tracking
exercise. Let `V_t(x) = 1/2 x' P_t x + s_t' x + constant`. The backward pass is

```
S = R + B' P_next B
K = solve(S, B' P_next A)
g = B' (P_next d + s_next) - R u_ref
k = -solve(S, g)
P = Q + K' R K + (A-BK)' P_next (A-BK)
s = -Q x_ref + A' (P_next d+s_next) - K' g
u = -K x + k
```

Here `d = E F_w`. **This package uses `u = -K @ x + k`; the homework stored
the negative sign inside its K and used `u = K @ x + k`.** Linear solves
replace explicit matrix inverses. The running CoP penalty is a regulation
proxy, not actuator energy or battery consumption.

## Interfaces

- `affine_lqr(A, B, Q, R, Qf, x_ref, u_ref, d=None)` returns full-horizon
  `(K, k)`. References have shapes `(N+1,nx)` and `(N,nu)`; matrices can be
  constant or have a leading time dimension of length `N`.
- `lip_discrete_dynamics(dt, height, mass)` returns `(A, B, E)` for the
  four-state LIP model.
- `LIPWalkingLQR(dt, height, mass, horizon_steps)` caches the reference-
  independent Riccati pass. Its default terminal value comes from the
  discrete algebraic Riccati equation.
- `controller.plan(x_ref, u_ref, wind_forces=None)` returns `(K, k)`.
- `controller.command(state, x_ref, u_ref, wind_forces=None)` returns a
  `ControlResult` containing `cop`, `unconstrained_cop`, the first-stage `K`
  and `k`, and a `saturated` flag.

Optional `support_lower` and `support_upper` in `command` clip the action to
an axis-aligned support rectangle. The walking simulation should instead use
its actual current support polygon when appropriate. Either projection is a
**saturated LQR heuristic**, not a constrained optimum; the controller does
not anticipate future support constraints. This limitation is a useful
baseline for the later MPC comparison. No stability guarantee is claimed
after support saturation or aerodynamic-model mismatch.

Predicted wind forces must be labeled by their source. A measured force held
through the horizon is not an oracle forecast. Omitting them makes the
prediction wind-unaware. Fixed height and support bounds also do not replace
friction, actuator, reachability, and contact constraints in a full robot.

## Verification

From the project root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s lqr/tests -v
```

The tests compare affine, time-varying Riccati policies with an independent
dense KKT solve; check the discretization against a matrix exponential; verify
a wind-compensating equilibrium; and check nominal feedback stability and
explicit saturation reporting.
