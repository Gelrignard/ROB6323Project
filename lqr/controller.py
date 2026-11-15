"""LQR center-of-pressure feedback for a reduced constant-height walking model.

State: [x, vx, y, vy], in SI units.  Action: ground-contact CoP [zx, zy].
The model assumes a horizontal wind force at the CoM, constant CoM height,
zero centroidal angular-momentum rate, and no vertical wind component:

    c_ddot = (g / h) * (c - z) + F_w / m.

Foot timing and the nominal footsteps are supplied by the simulator.  This
module does not solve whole-body inverse dynamics or invent a gait.  Wind
prediction is an optional known force sequence, held constant on each sample.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_discrete_are

from .riccati import _linear_pass, _quadratic_pass, _validated_inputs


def lip_discrete_dynamics(dt, height, mass, gravity=9.81):
    """Exact zero-order-hold matrices (A, B, E), x_next = A*x + B*z + E*F.

    State order is [x, vx, y, vy], CoP and force order is [x, y].  CoP and
    horizontal wind force are assumed constant during ``dt``.  E maps newtons
    to state increments.  This linear discretization does not approximate the
    variation of aerodynamic drag within a time step.
    """
    values = np.asarray([dt, height, mass, gravity], dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("dt, height, mass, and gravity must be finite and positive")
    omega = np.sqrt(gravity / height)
    ch, sh = np.cosh(omega * dt), np.sinh(omega * dt)
    A_axis = np.array([[ch, sh / omega], [omega * sh, ch]])
    B_axis = np.array([1.0 - ch, -omega * sh])
    E_axis = np.array([(ch - 1.0) / (mass * omega**2), sh / (mass * omega)])
    A = np.zeros((4, 4))
    B = np.zeros((4, 2))
    E = np.zeros((4, 2))
    for axis in range(2):
        section = slice(2 * axis, 2 * axis + 2)
        A[section, section] = A_axis
        B[section, axis] = B_axis
        E[section, axis] = E_axis
    return A, B, E


@dataclass(frozen=True)
class ControlResult:
    """One applied CoP command and the first-stage unconstrained policy."""

    cop: np.ndarray
    unconstrained_cop: np.ndarray
    K: np.ndarray
    k: np.ndarray
    saturated: bool


class LIPWalkingLQR:
    """Finite-horizon preview tracking using a cached Riccati quadratic pass.

    Reference arrays explicitly provide the N+1 desired states and N nominal
    CoPs.  The current state may differ from the reference; feedback corrects
    this error.  R penalizes CoP displacement from the nominal support path,
    not motor energy.  Optional clipping in ``command`` enforces a rectangular
    support bound at the current sample only; it has no feasibility guarantee.
    """

    def __init__(self, dt, height, mass, horizon_steps, Q=None, R=None, Qf=None, gravity=9.81):
        if isinstance(horizon_steps, bool) or not isinstance(horizon_steps, (int, np.integer)) or horizon_steps < 1:
            raise ValueError("horizon_steps must be a positive integer")
        self.dt, self.height, self.mass = float(dt), float(height), float(mass)
        self.gravity, self.horizon_steps = float(gravity), int(horizon_steps)
        self.A, self.B, self.E = lip_discrete_dynamics(dt, height, mass, gravity)
        self.Q = np.diag([20.0, 2.0, 40.0, 4.0]) if Q is None else np.asarray(Q, dtype=float).copy()
        self.R = np.diag([1.0, 1.0]) if R is None else np.asarray(R, dtype=float).copy()
        if Qf is None:
            Qf = solve_discrete_are(self.A, self.B, self.Q, self.R)
        self.Qf = np.asarray(Qf, dtype=float).copy()
        N = self.horizon_steps
        values = _validated_inputs(self.A, self.B, self.Q, self.R, self.Qf,
                                   np.zeros((N + 1, 4)), np.zeros((N, 2)), None)
        self._A, self._B, self._Q, self._R = values[:4]
        self._P, self._K, self._S = _quadratic_pass(self._A, self._B, self._Q, self._R, self.Qf)

    def plan(self, x_ref, u_ref, wind_forces=None):
        """Return full-horizon (K, k); forces may be (2,) or (N, 2).

        Passing no forces uses the wind-unaware model.  Supplying a measured
        force held over the horizon is estimation, not exact future preview.
        The caller must label how any supplied force predictions were obtained.
        """
        N = self.horizon_steps
        x_ref, u_ref = np.asarray(x_ref, dtype=float), np.asarray(u_ref, dtype=float)
        if x_ref.shape != (N + 1, 4) or u_ref.shape != (N, 2):
            raise ValueError(f"references must have shapes ({N + 1}, 4) and ({N}, 2)")
        if not np.all(np.isfinite(x_ref)) or not np.all(np.isfinite(u_ref)):
            raise ValueError("references must contain only finite values")
        wind_forces = np.zeros((N, 2)) if wind_forces is None else np.asarray(wind_forces, dtype=float)
        if wind_forces.shape == (2,):
            wind_forces = np.broadcast_to(wind_forces, (N, 2))
        if wind_forces.shape != (N, 2) or not np.all(np.isfinite(wind_forces)):
            raise ValueError(f"wind_forces must be finite with shape (2,) or ({N}, 2)")
        d = wind_forces @ self.E.T
        k = _linear_pass(self._A, self._B, self._Q, self._R, self.Qf,
                         x_ref, u_ref, d, self._P, self._K, self._S)
        return self._K.copy(), k

    def command(self, state, x_ref, u_ref, wind_forces=None,
                support_lower=None, support_upper=None):
        """Compute measured-state feedback, optionally clip to current support.

        Both support bounds must be specified together, as finite [x,y]
        arrays.  The rectangle must represent an admissible current support
        region supplied by the gait scheduler.  A bounding box around diagonal
        double-support feet can include unsupported points; use a true support
        polygon projection in the simulator if that geometry is required.
        """
        state = np.asarray(state, dtype=float)
        if state.shape != (4,) or not np.all(np.isfinite(state)):
            raise ValueError("state must be a finite array of shape (4,)")
        K, k = self.plan(x_ref, u_ref, wind_forces)
        raw = -K[0] @ state + k[0]
        if (support_lower is None) != (support_upper is None):
            raise ValueError("support_lower and support_upper must be supplied together")
        cop = raw.copy()
        if support_lower is not None:
            lower, upper = np.asarray(support_lower, dtype=float), np.asarray(support_upper, dtype=float)
            if lower.shape != (2,) or upper.shape != (2,) or not np.all(np.isfinite([lower, upper])):
                raise ValueError("support bounds must be finite arrays of shape (2,)")
            if np.any(lower > upper):
                raise ValueError("support_lower must not exceed support_upper")
            cop = np.clip(raw, lower, upper)
        return ControlResult(cop=cop, unconstrained_cop=raw,
                             K=K[0], k=k[0], saturated=bool(np.any(cop != raw)))
