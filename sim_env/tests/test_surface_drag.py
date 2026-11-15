"""Independent pressure-law, frame, velocity and wrench conservation checks."""

import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from sim_env.g1_model import load_g1_model, get_g1_profile
from sim_env.surface_drag import SurfaceDrag, panel_pressure_forces


def box_model():
    return mujoco.MjModel.from_xml_string('''
    <mujoco><asset><mesh name="box" scale=".1 .2 .3"
      vertex="-1 -1 -1  1 -1 -1  1 1 -1  -1 1 -1  -1 -1 1  1 -1 1  1 1 1  -1 1 1"
      face="0 2 1  0 3 2  4 5 6  4 6 7  0 1 5  0 5 4  1 2 6  1 6 5  2 3 7  2 7 6  3 0 4  3 4 7"/>
    </asset><worldbody><body pos="2 3 1" quat=".7071067812 .7071067812 0 0">
      <freejoint/><inertial pos=".1 .05 0" mass="2" diaginertia=".1 .1 .1"/>
      <geom type="mesh" mesh="box" group="2" contype="0" conaffinity="0"
        pos=".3 -.2 .1" quat=".7071067812 0 0 .7071067812"/>
      <geom type="mesh" mesh="box" group="3" pos=".3 -.2 .1" quat=".7071067812 0 0 .7071067812"/>
    </body></worldbody></mujoco>''')


class SurfacePressureTests(unittest.TestCase):
    def test_flat_plate_incidence_and_dissipation(self):
        normals = np.array([[-1., 0, 0], [1., 0, 0], [0, 1., 0]])
        velocity = np.tile([3., 0, 0], (3, 1))
        force, pressure = panel_pressure_forces(normals, np.array([2., 2., 2.]), velocity, density=2.)
        np.testing.assert_allclose(force, [[18, 0, 0], [0, 0, 0], [0, 0, 0]])
        np.testing.assert_allclose(pressure, [9, 0, 0])
        reverse, _ = panel_pressure_forces(normals, np.ones(3) * 2, -velocity, density=2.)
        np.testing.assert_allclose(reverse, [[0, 0, 0], [-18, 0, 0], [0, 0, 0]])
        zero, _ = panel_pressure_forces(normals, np.ones(3), np.zeros((3, 3)))
        np.testing.assert_array_equal(zero, np.zeros((3, 3)))
        self.assertTrue(np.all(np.einsum("ij,ij->i", force, velocity) >= 0))
        with self.assertRaises(ValueError):
            panel_pressure_forces(normals * 2, np.ones(3), velocity)

    def test_compiled_mesh_transform_matches_raw_geometry(self):
        model = box_model()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        surface = SurfaceDrag(model, max_patches=None)
        result = surface.evaluate(data, wind_scale=0)
        # Box local half widths(.1,.2,.3), geom Rz90, body Rx90:
        # world half widths(.2,.3,.1), center(2.3,2.9,.8).
        expected_center = np.array([2.3, 2.9, .8])
        np.testing.assert_allclose(np.average(result.points, axis=0, weights=result.areas), expected_center, atol=2e-8)
        np.testing.assert_allclose(np.max(np.abs(result.points - expected_center), axis=0), [.2, .3, .1], atol=2e-8)
        self.assertAlmostEqual(result.areas.sum(), 8 * (.1 * .2 + .2 * .3 + .1 * .3), places=6)
        self.assertEqual(surface.source_triangle_count, 12)  # collision duplicate excluded
        self.assertEqual(surface.metadata()["visual_mesh_geoms"], 1)
        outward = np.einsum("ij,ij->i", result.points - expected_center, result.normals)
        self.assertTrue(np.all(outward > 0))

    def test_surface_velocity_and_wrench_conservation(self):
        model = box_model()
        data = mujoco.MjData(model)
        data.qvel[:] = [.2, -.3, .4, .1, .7, -.2]
        mujoco.mj_forward(model, data)
        surface = SurfaceDrag(model, max_patches=None)
        qpos, qvel = data.qpos.copy(), data.qvel.copy()
        result = surface.evaluate(data, wind_scale=0)
        # Independently differentiate mesh point positions using MuJoCo's
        # quaternion-aware generalized-position integrator.
        tiny = 1e-6
        shifted = []
        for sign in (-1, 1):
            sample = mujoco.MjData(model)
            sample.qpos[:] = qpos
            mujoco.mj_integratePos(model, sample.qpos, qvel, sign * tiny)
            mujoco.mj_forward(model, sample)
            shifted.append(surface.evaluate(sample, wind_scale=0).points)
        point_velocity = (shifted[1] - shifted[0]) / (2 * tiny)
        np.testing.assert_allclose(result.relative_velocities, -point_velocity, atol=4e-9)
        np.testing.assert_allclose(result.total_force, result.patch_forces.sum(axis=0), atol=1e-12)
        moment = result.body_torques.sum(axis=0) + np.cross(data.xipos - result.com, result.body_forces).sum(axis=0)
        np.testing.assert_allclose(result.total_torque_com, moment, atol=1e-12)
        self.assertLessEqual(float(np.sum(result.patch_forces * point_velocity)), 1e-12)
        np.testing.assert_array_equal(data.qpos, qpos)
        np.testing.assert_array_equal(data.qvel, qvel)
        surface.apply(data, result)
        np.testing.assert_array_equal(data.xfrc_applied[:, :3], result.body_forces)
        np.testing.assert_array_equal(data.xfrc_applied[:, 3:], result.body_torques)

    def test_co_moving_uniform_air_has_zero_load(self):
        model = box_model()
        data = mujoco.MjData(model)
        data.qvel[:3] = [2, -1, 0]
        mujoco.mj_forward(model, data)
        surface = SurfaceDrag(model, max_patches=None)
        with patch("sim_env.surface_drag.tornado_velocity", side_effect=lambda positions, *args, **kwargs: np.tile([2., -1.], (len(positions), 1))):
            result = surface.evaluate(data)
        np.testing.assert_allclose(result.patch_forces, 0, atol=1e-12)

    def test_g1_quadrature_preserves_area_and_approximates_full_surface(self):
        model = load_g1_model()
        data = mujoco.MjData(model)
        data.qpos[:] = get_g1_profile(model).home_qpos
        data.qpos[:2] = [1., 5.]
        mujoco.mj_forward(model, data)
        approximate = SurfaceDrag(model)
        exact = SurfaceDrag(model, max_patches=None)
        small, full = approximate.evaluate(data), exact.evaluate(data)
        self.assertEqual(len(small.areas), 4096)
        self.assertEqual(approximate.metadata()["visual_mesh_geoms"], 35)
        self.assertEqual(approximate.source_triangle_count, 393270)
        self.assertAlmostEqual(small.areas.sum(), full.areas.sum(), places=12)
        error = np.linalg.norm(small.total_force - full.total_force) / np.linalg.norm(full.total_force)
        self.assertLess(error, .025)
        self.assertLess(np.linalg.norm(small.total_torque_com - full.total_torque_com), .15)


if __name__ == "__main__":
    unittest.main()
