"""Finite-horizon affine LQR, following the quadratic-cost derivation in HW2.

The model is x[t+1] = A[t] x[t] + B[t] u[t] + d[t], and the objective is

    1/2 sum((x[t]-x_ref[t])' Q[t] (x[t]-x_ref[t])
            + (u[t]-u_ref[t])' R[t] (u[t]-u_ref[t]))
    + 1/2 (x[N]-x_ref[N])' Qf (x[N]-x_ref[N]).

The policy convention here is u[t] = -K[t] @ x[t] + k[t].  HW2's solve_lqr
stored the minus sign inside K; the mathematics is otherwise the same.
No inequality constraints are solved by this recursion.
"""

import numpy as np


def _matrix_sequence(value, horizon, rows, columns, name):
    value = np.asarray(value, dtype=float)
    if value.ndim == 0 and rows == columns == 1:
        value = value.reshape(1, 1)
    if value.shape == (rows, columns):
        value = np.broadcast_to(value, (horizon, rows, columns))
    if value.shape != (horizon, rows, columns):
        raise ValueError(
            f"{name} must have shape ({rows}, {columns}) or "
            f"({horizon}, {rows}, {columns})"
        )
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values")
    return value


def _check_weights(weight, name, positive_definite=False):
    if not np.allclose(weight, np.swapaxes(weight, -1, -2), atol=1e-12, rtol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    eigenvalues = np.linalg.eigvalsh(weight)
    if positive_definite:
        if np.any(eigenvalues <= 0):
            raise ValueError(f"{name} must be positive definite")
    elif np.any(eigenvalues < -1e-10):
        raise ValueError(f"{name} must be positive semidefinite")


def _validated_inputs(A, B, Q, R, Qf, x_ref, u_ref, d):
    x_ref = np.asarray(x_ref, dtype=float)
    u_ref = np.asarray(u_ref, dtype=float)
    if x_ref.ndim != 2 or x_ref.shape[0] < 2 or x_ref.shape[1] < 1:
        raise ValueError("x_ref must have shape (N+1, nx), with N >= 1")
    horizon, nx = x_ref.shape[0] - 1, x_ref.shape[1]
    if u_ref.ndim != 2 or u_ref.shape[0] != horizon or u_ref.shape[1] < 1:
        raise ValueError("u_ref must have shape (N, nu)")
    nu = u_ref.shape[1]
    if not np.all(np.isfinite(x_ref)) or not np.all(np.isfinite(u_ref)):
        raise ValueError("references must contain only finite values")
    A = _matrix_sequence(A, horizon, nx, nx, "A")
    B = _matrix_sequence(B, horizon, nx, nu, "B")
    Q = _matrix_sequence(Q, horizon, nx, nx, "Q")
    R = _matrix_sequence(R, horizon, nu, nu, "R")
    Qf = np.asarray(Qf, dtype=float)
    if Qf.shape != (nx, nx) or not np.all(np.isfinite(Qf)):
        raise ValueError("Qf must be a finite matrix with shape (nx, nx)")
    _check_weights(Q, "Q")
    _check_weights(R, "R", positive_definite=True)
    _check_weights(Qf, "Qf")
    d = np.zeros((horizon, nx)) if d is None else np.asarray(d, dtype=float)
    if d.shape == (nx,):
        d = np.broadcast_to(d, (horizon, nx))
    if d.shape != (horizon, nx) or not np.all(np.isfinite(d)):
        raise ValueError("d must be finite with shape (nx,) or (N, nx)")
    return A, B, Q, R, Qf, x_ref, u_ref, d


def _quadratic_pass(A, B, Q, R, Qf):
    """Compute reference-independent value Hessians and feedback gains."""
    horizon, nx, nu = len(A), A.shape[1], B.shape[2]
    P = np.empty((horizon + 1, nx, nx))
    K = np.empty((horizon, nu, nx))
    S = np.empty((horizon, nu, nu))
    P[-1] = Qf
    for t in range(horizon - 1, -1, -1):
        S[t] = R[t] + B[t].T @ P[t + 1] @ B[t]
        K[t] = np.linalg.solve(S[t], B[t].T @ P[t + 1] @ A[t])
        closed_loop = A[t] - B[t] @ K[t]
        # Equivalent to Q + A'PA - A'PB S^-1 B'PA; avoids subtraction.
        P[t] = Q[t] + K[t].T @ R[t] @ K[t] + closed_loop.T @ P[t + 1] @ closed_loop
        P[t] = 0.5 * (P[t] + P[t].T)
    return P, K, S


def _linear_pass(A, B, Q, R, Qf, x_ref, u_ref, d, P, K, S):
    """Update linear value terms for reference tracking and affine dynamics."""
    horizon, nu = len(A), B.shape[2]
    k = np.empty((horizon, nu))
    # V_t(x) = 1/2 x' P_t x + s_t' x + const.
    s = -Qf @ x_ref[-1]
    for t in range(horizon - 1, -1, -1):
        downstream = P[t + 1] @ d[t] + s
        g = B[t].T @ downstream - R[t] @ u_ref[t]
        k[t] = -np.linalg.solve(S[t], g)
        s = -Q[t] @ x_ref[t] + A[t].T @ downstream - K[t].T @ g
    return k


def affine_lqr(A, B, Q, R, Qf, x_ref, u_ref, d=None):
    """Return (K, k) for u[t] = -K[t] @ x[t] + k[t].

    ``x_ref`` and ``u_ref`` have shapes (N+1, nx), (N, nu).  A, B, Q,
    and R may be constant matrices or sequences with leading dimension N.
    ``d`` is an additive state increment, shape (nx,) or (N, nx), not a
    continuous-time force.  Q/Qf must be PSD and R must be positive definite.
    The output shapes are (N, nu, nx) and (N, nu).

    This is an exact unconstrained finite-horizon LQ solution, including the
    affine term.  Clipping its output is a heuristic, not constrained LQR/MPC.
    """
    values = _validated_inputs(A, B, Q, R, Qf, x_ref, u_ref, d)
    A, B, Q, R, Qf, x_ref, u_ref, d = values
    P, K, S = _quadratic_pass(A, B, Q, R, Qf)
    k = _linear_pass(A, B, Q, R, Qf, x_ref, u_ref, d, P, K, S)
    return K, k
