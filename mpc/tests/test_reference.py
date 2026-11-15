"""Kinematic reference invariants; these do not claim physical gait stability."""

import unittest

import mujoco
import numpy as np

from mpc.reference import G1WalkingReference


class G1ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = G1WalkingReference(duration=2., preview_padding=.5, sample_dt=.02)

    def test_reference_matches_sole_and_com_geometry(self):
        reference = self.reference
        data = mujoco.MjData(reference.model)
        for time in (0., .7, 1.0, 1.5, 2.0):
            row = reference.sample(time)
            data.qpos[:] = row["qpos"]
            mujoco.mj_forward(reference.model, data)
            np.testing.assert_allclose(data.subtree_com[reference.profile.root_body_id], row["com"], atol=2e-5)
            np.testing.assert_allclose(data.site_xpos[list(reference.profile.sole_site_ids)], row["feet"], atol=2e-5)
            np.testing.assert_allclose(row["joint_targets"], row["qpos"][reference.joint_qpos_indices])
        self.assertFalse(reference.metadata()["plant_state_prescribed"])

    def test_reference_has_advancement_and_alternating_swing(self):
        reference = self.reference
        rows = reference.batch(np.arange(.7, 2.4, .02))
        self.assertGreater(rows["com"][-1, 0] - rows["com"][0, 0], .1)
        for foot in (0, 1):
            self.assertGreater(rows["feet"][:, foot, 2].max(), .03)
            self.assertTrue(np.any(rows["stance"][:, foot]))
            self.assertTrue(np.any(~rows["stance"][:, foot]))
        np.testing.assert_allclose(np.linalg.norm(rows["qpos"][:, 3:7], axis=1), 1.)

    def test_invalid_query_cannot_silently_extrapolate(self):
        for time in (-.1, np.nan, self.reference.times[-1] + .01):
            with self.assertRaises(ValueError):
                self.reference.sample(time)


if __name__ == "__main__":
    unittest.main()
