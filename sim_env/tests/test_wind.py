"""Physical and coordinate consistency checks for the synthetic field."""

from dataclasses import replace
import unittest

import numpy as np

from sim_env.config import DEFAULT_TORNADO, TornadoConfig
from sim_env.wind import (aerodynamic_force, local_to_world, tornado_velocity,
                      wind_force, world_to_local)


class WindTests(unittest.TestCase):
    def test_center_is_finite_and_zero(self):
        np.testing.assert_array_equal(tornado_velocity([50, 50]), [0, 0])
        values = tornado_velocity([[50, 50], [50 + 1e-10, 50]])
        self.assertTrue(np.isfinite(values).all())

    def test_component_maxima_at_core_radius(self):
        cfg = DEFAULT_TORNADO
        at_east_core = np.array(cfg.center_m) + [cfg.core_radius_m, 0]
        np.testing.assert_allclose(tornado_velocity(at_east_core),
                                   [-cfg.peak_inflow_speed_mps,
                                    cfg.peak_tangential_speed_mps])
        radii = np.linspace(0, 200, 10001)
        points = np.stack((50 + radii, np.full_like(radii, 50)), axis=-1)
        velocity = tornado_velocity(points)
        self.assertLessEqual(np.max(np.abs(velocity[:, 1])),
                             cfg.peak_tangential_speed_mps + 1e-12)

    def test_rotational_symmetry(self):
        rotation = np.array([[0, -1], [1, 0]])
        center = np.asarray(DEFAULT_TORNADO.center_m)
        points = np.array([[62, 50], [71, 46], [33, 21]])
        rotated = (points - center) @ rotation.T + center
        np.testing.assert_allclose(tornado_velocity(rotated),
                                   tornado_velocity(points) @ rotation.T)

    def test_relative_wind_zero_and_dissipation(self):
        np.testing.assert_array_equal(aerodynamic_force([2, -3], [2, -3]), [0, 0])
        velocities = np.array([[1, 0], [0, -3], [2, 4]])
        forces = aerodynamic_force([0, 0], velocities)
        self.assertTrue(np.all(np.sum(forces * velocities, axis=-1) < 0))

    def test_quadratic_drag_and_vectorization(self):
        force = aerodynamic_force([3, 4], [0, 0])
        expected = 0.5 * 1.225 * 0.5 * 5 * np.array([3, 4])
        np.testing.assert_allclose(force, expected)
        np.testing.assert_allclose(aerodynamic_force([6, 8], [0, 0]), 4 * force)
        positions = np.zeros((3, 4, 2)) + [70, 50]
        self.assertEqual(wind_force(positions, [0, 0]).shape, (3, 4, 2))

    def test_local_world_frame_roundtrip(self):
        local = np.array([[0, 0], [10, 10], [1, 5], [8, 5]])
        world = local_to_world(local)
        np.testing.assert_array_equal(world[0], [68, 45])
        np.testing.assert_array_equal(world[1], [78, 55])
        np.testing.assert_array_equal(world_to_local(world), local)
        self.assertEqual(DEFAULT_TORNADO.patch_bounds_m, (68, 78, 45, 55))

    def test_zero_scale_and_custom_field(self):
        np.testing.assert_array_equal(tornado_velocity([70, 50], scale=0), [0, 0])
        np.testing.assert_allclose(tornado_velocity([70, 50], scale=2),
                                   2 * tornado_velocity([70, 50]))
        cfg = replace(DEFAULT_TORNADO, peak_inflow_speed_mps=0)
        self.assertEqual(tornado_velocity([70, 50], cfg)[0], 0)

    def test_invalid_configuration_and_inputs(self):
        for kwargs in ({"core_radius_m": 0}, {"patch_origin_m": (95, 50)},
                       {"projected_area_m2": -1}, {"center_m": (float("nan"), 0)}):
            with self.assertRaises(ValueError):
                TornadoConfig(**kwargs)
        with self.assertRaises(ValueError):
            tornado_velocity([1, 2, 3])
        with self.assertRaises(ValueError):
            tornado_velocity([1, 2], scale=-1)


if __name__ == "__main__":
    unittest.main()
