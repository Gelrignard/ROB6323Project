"""Verify G1 kinematic mapping without mistaking it for a physics rollout."""

import unittest
from unittest.mock import patch

import numpy as np

from sim_env.g1_replay import G1Replay
from sim_env.gait import GaitConfig, WalkingReference


class G1ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The vendor stand is near full leg extension; use a declared crouch.
        cls.replay = G1Replay()
        profile = cls.replay.profile
        cls.gait = WalkingReference(GaitConfig(
            com_height=cls.replay.com_height, step_width=profile.step_width,
            foot_half_length=profile.sole_half_length,
            foot_half_width=profile.sole_half_width))

    def test_analytic_jacobian_matches_finite_differences(self):
        r = self.replay
        pose = self.gait.sample(1.4)
        r.set_state(pose["com_ref"], pose["feet"], 1.4, pose["cop_ref"])
        variables = r._variables.copy()
        # Check a nonzero orientation error, not only the exactly flat pose.
        variables[4] += .025
        variables[8] -= .02
        target_com = np.array([pose["com_ref"][0], pose["com_ref"][2], r.com_height])
        analytic = r._jacobian(variables, target_com, pose["feet"])
        numeric = np.empty_like(analytic)
        step = 1e-6
        for column in range(len(variables)):
            positive, negative = variables.copy(), variables.copy()
            positive[column] += step
            negative[column] -= step
            numeric[:, column] = (r._residual(positive, target_com, pose["feet"])
                                  - r._residual(negative, target_com, pose["feet"])) / (2 * step)
        np.testing.assert_allclose(analytic, numeric, atol=2e-8, rtol=1e-6)

    def test_full_gait_samples_match_feet_and_actual_com_with_joint_bounds(self):
        r = self.replay
        with patch("mujoco.mj_step", side_effect=AssertionError("IK must not integrate physics")):
            for time in np.arange(0.0, 16.0, .04):
                pose = self.gait.sample(time)
                report = r.set_state(pose["com_ref"], pose["feet"], time, pose["cop_ref"])
                self.assertTrue(report.success)
                self.assertTrue(report.kinematic_only)
                self.assertLess(report.max_foot_error_m, 1e-5)
                self.assertLess(report.com_error_m, 1e-5)
                self.assertLess(report.max_foot_orientation_error_rad, 1e-4)
                np.testing.assert_allclose(r.data.site_xpos[list(r.sole_site_ids)], pose["feet"], atol=1e-5)
                self.assertTrue(np.all(r._variables >= r.lower - 1e-10))
                self.assertTrue(np.all(r._variables <= r.upper + 1e-10))
                self.assertAlmostEqual(r.data.time, time)
                unchanged = np.ones(r.model.nq, dtype=bool)
                unchanged[r.qpos_indices] = False
                np.testing.assert_array_equal(r.data.qpos[unchanged], r.home_qpos[unchanged])
        self.assertTrue(r.diagnostics()["kinematic_only"])

    def test_rejects_invalid_targets(self):
        pose = self.gait.sample(0)
        with self.assertRaises(ValueError):
            self.replay.set_state([1, 2], pose["feet"], 0, pose["cop_ref"])
        with self.assertRaises(ValueError):
            self.replay.set_state([1, 0, np.nan, 0], pose["feet"], 0, pose["cop_ref"])


if __name__ == "__main__":
    unittest.main()
