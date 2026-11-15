"""Independent numerical checks; run python -m unittest discover -s lqr/tests."""

import unittest

import numpy as np
from scipy.linalg import expm

from lqr import LIPWalkingLQR, affine_lqr, lip_discrete_dynamics


def dense_kkt(A, B, Q, R, Qf, x_ref, u_ref, d, x0):
    """Independent equality-constrained QP solve, not a Riccati recursion."""
    N, nx, nu = len(A), A.shape[1], B.shape[2]
    nstate, ncontrol = (N + 1) * nx, N * nu
    nvar, nconstraint = nstate + ncontrol, (N + 1) * nx
    H, f = np.zeros((nvar, nvar)), np.zeros(nvar)
    E, b = np.zeros((nconstraint, nvar)), np.zeros(nconstraint)
    E[:nx, :nx], b[:nx] = np.eye(nx), x0
    for t in range(N):
        sx = slice(t * nx, (t + 1) * nx)
        su = slice(nstate + t * nu, nstate + (t + 1) * nu)
        sn = slice((t + 1) * nx, (t + 2) * nx)
        H[sx, sx], H[su, su] = Q[t], R[t]
        f[sx], f[su] = -Q[t] @ x_ref[t], -R[t] @ u_ref[t]
        E[sn, sx], E[sn, su], E[sn, sn] = -A[t], -B[t], np.eye(nx)
        b[sn] = d[t]
    H[N * nx:nstate, N * nx:nstate] = Qf
    f[N * nx:nstate] = -Qf @ x_ref[-1]
    system = np.block([[H, E.T], [E, np.zeros((nconstraint, nconstraint))]])
    solution = np.linalg.solve(system, np.r_[-f, b])[:nvar]
    return solution[:nstate].reshape(N + 1, nx), solution[nstate:].reshape(N, nu)


class RiccatiTests(unittest.TestCase):
    def test_time_varying_affine_tracking_matches_dense_kkt(self):
        rng = np.random.default_rng(6323)
        N, nx, nu = 7, 3, 2
        A = 0.85 * np.eye(nx)[None, :, :] + 0.15 * rng.normal(size=(N, nx, nx))
        B = rng.normal(size=(N, nx, nu))
        C = rng.normal(size=(N, nx, nx))
        Q = np.swapaxes(C, -1, -2) @ C  # Positive semidefinite.
        C = rng.normal(size=(N, nu, nu))
        R = np.swapaxes(C, -1, -2) @ C + 0.4 * np.eye(nu)
        Qf = np.diag([2.0, 0.0, 1.0])  # Singular terminal weight is valid.
        x_ref, u_ref = rng.normal(size=(N + 1, nx)), rng.normal(size=(N, nu))
        d, x0 = 0.2 * rng.normal(size=(N, nx)), rng.normal(size=nx)
        K, k = affine_lqr(A, B, Q, R, Qf, x_ref, u_ref, d)
        x, u = np.empty((N + 1, nx)), np.empty((N, nu))
        x[0] = x0
        for t in range(N):
            u[t] = -K[t] @ x[t] + k[t]
            x[t + 1] = A[t] @ x[t] + B[t] @ u[t] + d[t]
        x_kkt, u_kkt = dense_kkt(A, B, Q, R, Qf, x_ref, u_ref, d, x0)
        np.testing.assert_allclose(x, x_kkt, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(u, u_kkt, rtol=1e-10, atol=1e-10)

    def test_constant_inputs_equal_explicit_sequences(self):
        N = 8
        A, B = np.array([[1.0, 0.1], [0.0, 1.0]]), np.array([[0.005], [0.1]])
        Q, R, Qf = np.diag([3.0, 1.0]), np.array([[0.2]]), np.eye(2)
        xr, ur, d = np.zeros((N + 1, 2)), np.zeros((N, 1)), np.array([0.02, -0.01])
        constant = affine_lqr(A, B, Q, R, Qf, xr, ur, d)
        sequence = affine_lqr(*(np.repeat(v[None], N, axis=0) for v in (A, B, Q, R)),
                              Qf, xr, ur, np.tile(d, (N, 1)))
        for actual, expected in zip(constant, sequence):
            np.testing.assert_allclose(actual, expected)

    def test_rejects_indefinite_control_cost(self):
        with self.assertRaisesRegex(ValueError, "positive definite"):
            affine_lqr(np.eye(1), np.eye(1), np.eye(1), np.array([[-1.0]]),
                       np.eye(1), np.zeros((3, 1)), np.zeros((2, 1)))


class LIPTests(unittest.TestCase):
    def test_exact_discretization_matches_matrix_exponential(self):
        dt, height, mass, gravity = 0.037, 0.92, 42.0, 9.81
        A, B, E = lip_discrete_dynamics(dt, height, mass, gravity)
        Ac = np.zeros((4, 4))
        Bc, Ec = np.zeros((4, 2)), np.zeros((4, 2))
        for axis in range(2):
            i = axis * 2
            Ac[i, i + 1], Ac[i + 1, i] = 1.0, gravity / height
            Bc[i + 1, axis], Ec[i + 1, axis] = -gravity / height, 1.0 / mass
        augmented = np.zeros((8, 8))
        augmented[:4, :4], augmented[:4, 4:6], augmented[:4, 6:] = Ac, Bc, Ec
        exact = expm(augmented * dt)
        np.testing.assert_allclose(A, exact[:4, :4], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(B, exact[:4, 4:6], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(E, exact[:4, 6:], rtol=1e-12, atol=1e-12)

    def test_wind_compensated_equilibrium_is_preserved(self):
        N, height, mass = 70, 0.9, 40.0
        controller = LIPWalkingLQR(0.02, height, mass, N)
        force = np.array([18.0, -7.0])
        state = np.array([1.7, 0.0, -0.6, 0.0])
        equilibrium_cop = state[[0, 2]] + height * force / (mass * controller.gravity)
        xr, ur = np.tile(state, (N + 1, 1)), np.tile(equilibrium_cop, (N, 1))
        result = controller.command(state, xr, ur, force)
        np.testing.assert_allclose(result.cop, equilibrium_cop, atol=1e-11)
        next_state = controller.A @ state + controller.B @ result.cop + controller.E @ force
        np.testing.assert_allclose(next_state, state, atol=1e-11)
        self.assertFalse(result.saturated)

    def test_feedback_stabilizes_unconstrained_lip_and_reports_clipping(self):
        N = 60
        controller = LIPWalkingLQR(0.02, 0.9, 40.0, N)
        xr, ur = np.zeros((N + 1, 4)), np.zeros((N, 2))
        K, k = controller.plan(xr, ur)
        eigenvalues = np.linalg.eigvals(controller.A - controller.B @ K[0])
        self.assertLess(np.max(np.abs(eigenvalues)), 1.0)
        state = np.array([0.12, 0.08, -0.06, 0.01])
        initial_norm = np.linalg.norm(state)
        for _ in range(300):
            control = -K[0] @ state + k[0]
            state = controller.A @ state + controller.B @ control
        self.assertLess(np.linalg.norm(state), initial_norm * 1e-4)
        result = controller.command(np.array([0.5, 0.0, -0.5, 0.0]), xr, ur,
                                    support_lower=[-0.1, -0.05], support_upper=[0.1, 0.05])
        self.assertTrue(result.saturated)
        np.testing.assert_allclose(result.cop, [0.1, -0.05])

    def test_cached_plan_matches_general_solver(self):
        rng = np.random.default_rng(8)
        N = 20
        controller = LIPWalkingLQR(0.02, 0.9, 40.0, N)
        xr, ur, force = rng.normal(size=(N + 1, 4)), rng.normal(size=(N, 2)), rng.normal(size=(N, 2))
        cached = controller.plan(xr, ur, force)
        general = affine_lqr(controller.A, controller.B, controller.Q, controller.R,
                             controller.Qf, xr, ur, force @ controller.E.T)
        for actual, expected in zip(cached, general):
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
