"""Moving surface loads require generalized velocities, including rotation."""

import unittest
import mujoco
import numpy as np

from sim_env.visualize.surface_profile import pose_velocities


class PoseVelocityTests(unittest.TestCase):
    def test_constant_translation_and_rotation_are_recovered(self):
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body><freejoint/><geom type="box" size=".1 .1 .1"/></body></worldbody></mujoco>')
        times = np.arange(0, .201, .04)
        poses = np.zeros((len(times), model.nq))
        poses[:, :3] = np.array([1, 2, 3]) + times[:, None] * [1.5, -.2, .1]
        poses[:, 3] = np.cos(.4 * times / 2)
        poses[:, 6] = np.sin(.4 * times / 2)
        actual = pose_velocities(model, times, poses)
        expected = np.tile([1.5, -.2, .1, 0, 0, .4], (len(times), 1))
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        # Single-pose snapshots cannot infer motion; explicitly return zero.
        np.testing.assert_array_equal(pose_velocities(model, times[:1], poses[:1]), np.zeros((1, 6)))
        with self.assertRaises(ValueError):
            pose_velocities(model, [0, 0], poses[:2])


if __name__ == '__main__':
    unittest.main()
