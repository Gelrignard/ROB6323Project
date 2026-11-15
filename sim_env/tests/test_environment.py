"""Physics and feedback checks that do not depend on rendered animation."""

import unittest

import numpy as np

from lqr import lip_discrete_dynamics
from sim_env.config import TornadoConfig
from sim_env.environment import ReducedWalkingEnv, SimulationConfig
from sim_env.gait import point_in_support
from sim_env.run import simulate


class MuJoCoDynamicsTests(unittest.TestCase):
    def test_state_is_integrated_by_mujoco_and_matches_reduced_dynamics(self):
        cfg = SimulationConfig(physics_dt=.0002)
        env = ReducedWalkingEnv(cfg, TornadoConfig(drag_coefficient=0.0))
        state = np.array([2.0, .12, 5.0, -.03])
        cop = np.array([2.04, 5.02])
        env.reset(state)
        actual = env.step(cop)
        A, B, _ = lip_discrete_dynamics(cfg.control_dt, cfg.height, cfg.mass)
        expected = A @ state + B @ cop
        # MuJoCo uses semi-implicit Euler; verify error AND step refinement,
        # rather than incorrectly requiring the exact-ZOH controller map.
        np.testing.assert_allclose(actual, expected, atol=4e-6, rtol=0)
        coarse = ReducedWalkingEnv(SimulationConfig(physics_dt=.002), TornadoConfig(drag_coefficient=0.0))
        coarse.reset(state)
        coarse_error = np.linalg.norm(coarse.step(cop) - expected)
        fine_error = np.linalg.norm(actual - expected)
        self.assertLess(fine_error, coarse_error / 8)
        self.assertFalse(np.array_equal(actual, state))
        self.assertEqual((env.model.nq, env.model.nv), (2, 2))
        self.assertAlmostEqual(env.data.time, cfg.control_dt)

    def test_support_and_friction_failures_are_reported(self):
        env = ReducedWalkingEnv(SimulationConfig(wind_scale=0))
        env.reset([1, 0, 5, 0])
        polygon = np.array([[.9, 4.9], [1.1, 4.9], [1.1, 5.1], [.9, 5.1]])
        with self.assertRaises(ValueError):
            env.step([2, 5], polygon)
        env.step([2, 5])
        self.assertEqual(env.failure_reason(), "required_ground_force_exceeded_friction_limit")
        env.reset([1, 0, 5, 0])
        np.testing.assert_array_equal(env.last_ground_force, [0, 0])

    def test_lqr_recovers_from_initial_error_and_respects_held_support(self):
        rows, summary, _, gait = simulate(duration=4, initial_offset=[.025, 0, -.015, 0])
        self.assertTrue(summary["tracking_success"])
        self.assertLess(summary["final_position_error_m"], .005)
        self.assertGreater(summary["completed_footsteps"], 0)
        self.assertGreater(summary["max_wind_force_N"], 20)
        self.assertLess(summary["max_friction_utilization"], 1)
        for row in rows:
            for offset in (0, .005, .010, .015, .0199):
                self.assertTrue(point_in_support(row["cop"], gait.sample(row["time"] + offset)["support_polygon"]))

    def test_open_loop_does_not_hide_a_kinematic_state_reset(self):
        _, closed, _, _ = simulate(duration=2, initial_offset=[.025, 0, 0, 0])
        _, opened, _, _ = simulate(duration=2, initial_offset=[.025, 0, 0, 0], controller_mode="open-loop")
        self.assertTrue(closed["tracking_success"])
        self.assertGreater(opened["max_position_error_m"], 5 * closed["max_position_error_m"])
        self.assertFalse(opened["tracking_success"])


if __name__ == "__main__":
    unittest.main()
