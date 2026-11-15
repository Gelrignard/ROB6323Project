"""Compare prescribed held-out fields and their 3D physics integration."""

import unittest

import numpy as np

from sim_env.g1_physics import G1PhysicsEnv
from sim_env.wind import tornado_velocity
from sim_env.wind_fields import WindFieldConfig, make_wind_field


class WindFieldTests(unittest.TestCase):
    def test_original_vortex_and_reverse_circulation(self):
        points = np.array([[70, 50, .6], [50, 62, 1], [50, 50, 0]])
        field = make_wind_field({'kind': 'tornado'})
        np.testing.assert_allclose(field(points, 5)[:, :2], tornado_velocity(points[:, :2]))
        reverse = make_wind_field({'kind': 'tornado', 'tangential_speed_mps': -12})
        self.assertLess(reverse(points, 0)[0, 1], 0)
        np.testing.assert_array_equal(field(points, 0)[:, 2], 0)
        np.testing.assert_array_equal(make_wind_field()(points, 1), 0)

    def test_gust_seed_and_field_sampling_are_order_independent(self):
        config = {'kind': 'gust', 'seed': 72, 'velocity_mps': (0, 8, 0), 'gust_amplitude_mps': (0, 3, 1)}
        a, b = make_wind_field(config), make_wind_field(config)
        points = np.array([[70, 48, .5], [75, 53, 1.2]])
        expected = a(points, .45)
        a(points, 9.2)
        np.testing.assert_array_equal(a(points, .45), expected)
        np.testing.assert_array_equal(b(points, .45), expected)
        self.assertGreater(np.linalg.norm(a(points, .45) - make_wind_field({**config, 'seed': 73})(points, .45)), .01)

    def test_shear_height_and_translating_center(self):
        field = make_wind_field({'kind': 'shear', 'velocity_mps': [0, 5, 0], 'shear_mps_per_m': [0, 2, 0]})
        np.testing.assert_allclose(field([[70, 50, 0], [70, 50, 2]], 0), [[0, 3, 0], [0, 7, 0]])
        moving = make_wind_field({'kind': 'moving_vortex', 'center_velocity_mps': [1, -.5]})
        np.testing.assert_allclose(moving([52, 49, .8], 2), [0, 0, 0], atol=1e-12)
        with self.assertRaises(ValueError):
            WindFieldConfig(kind='gust', gust_period_s=0)

    def test_provider_reaches_surface_and_lumped_physics_in_world_frame(self):
        for model in ('lumped', 'surface'):
            seen = []
            def field(points, time_s):
                p = np.asarray(points)
                seen.append((p.copy(), time_s))
                return np.broadcast_to([0, 0, 5], p.shape).copy()
            env = G1PhysicsEnv(wind_scale=1, drag_model=model, wind_field=field)
            env._apply_wind()
            self.assertGreater(env.last_wind_force[2], .1)
            self.assertTrue(np.all(seen[-1][0][..., 0] > 68))
            self.assertTrue(np.all(seen[-1][0][..., 1] > 49))
            self.assertEqual(seen[-1][1], 0)


if __name__ == '__main__':
    unittest.main()
