"""Behavioral checks for prescribed contacts and model-consistent references."""

import unittest

import numpy as np

from sim_env.gait import (GaitConfig, WalkingReference, intersect_support_polygons,
                      point_in_support, project_to_support)


class GaitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gait = WalkingReference()

    def test_stance_feet_are_stationary_and_support_is_valid(self):
        for t in np.arange(0, self.gait.duration, 0.017):
            a, b = self.gait.sample(t), self.gait.sample(t + 1e-5)
            self.assertTrue(point_in_support(a["cop_ref"], a["support_polygon"]))
            for index in a["stance_feet"]:
                self.assertAlmostEqual(a["feet"][index, 2], 0.0)
                np.testing.assert_allclose(a["feet"][index], b["feet"][index], atol=1e-6)
            lo, hi = a["support_lower"], a["support_upper"]
            for corner in [[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]]:
                self.assertTrue(point_in_support(corner, a["support_polygon"]))

    def test_steps_alternate_and_close(self):
        cfg = self.gait.config
        phases = []
        for event in range(cfg.n_steps + 1):
            t = cfg.walk_start + (event + 0.6) * cfg.step_duration
            sample = self.gait.sample(t)
            phases.append(sample["phase"])
            swing = 1 - event % 2
            self.assertGreater(sample["feet"][swing, 2], 0.01)
        self.assertEqual(phases[:4], ["left_support", "right_support", "left_support", "right_support"])
        final = self.gait.sample(cfg.duration)
        self.assertEqual(final["step_count"], cfg.n_steps + 1)
        self.assertEqual(final["phase"], "standing")
        np.testing.assert_allclose(final["feet"][:, 0], cfg.start_x + cfg.n_steps * self.gait.step_length)

    def test_foot_and_cop_continuity_at_transitions(self):
        cfg = self.gait.config
        for event in range(cfg.n_steps + 2):
            for fraction in [0.0, cfg.double_support_fraction]:
                t = cfg.walk_start + (event + fraction) * cfg.step_duration
                a, b = self.gait.sample(t - 1e-7), self.gait.sample(t + 1e-7)
                np.testing.assert_allclose(a["feet"], b["feet"], atol=1e-5)
                np.testing.assert_allclose(a["cop_ref"], b["cop_ref"], atol=1e-5)

    def test_reference_satisfies_lip_dynamics(self):
        cfg = self.gait.config
        dt = cfg.reference_dt
        times = np.arange(0.1, cfg.duration - 0.1, dt)
        states = np.asarray([self.gait.sample(t)["com_ref"] for t in times])
        cops = np.asarray([self.gait.sample(t)["cop_ref"] for t in times])
        velocity_derivative = (states[2:, [1, 3]] - states[:-2, [1, 3]]) / (2 * dt)
        expected = cfg.gravity / cfg.com_height * (states[1:-1, [0, 2]] - cops[1:-1])
        np.testing.assert_allclose(velocity_derivative, expected, atol=0.006, rtol=0.01)

    def test_polygon_projection_rejects_unsupported_bounding_box_corner(self):
        t = self.gait.config.walk_start + 1.05 * self.gait.config.step_duration
        polygon = self.gait.sample(t)["support_polygon"]
        point = np.array([polygon[:, 0].max(), polygon[:, 1].max()])
        self.assertFalse(point_in_support(point, polygon))
        projected = project_to_support(point, polygon)
        self.assertTrue(point_in_support(projected, polygon))
        self.assertGreater(np.linalg.norm(projected - point), 1e-3)

    def test_horizon_shapes(self):
        horizon = self.gait.sample_horizon(2.0, 0.02, 50)
        self.assertEqual(horizon["com_ref"].shape, (51, 4))
        self.assertEqual(horizon["cop_ref"].shape, (50, 2))
        self.assertEqual(horizon["feet"].shape, (51, 2, 3))
        self.assertEqual(len(horizon["support_polygons"]), 50)

    def test_support_intersection_of_offset_rectangles(self):
        first = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
        second = np.array([[1, 1], [3, 1], [3, 3], [1, 3]], dtype=float)
        result = intersect_support_polygons([first, second[::-1]])
        np.testing.assert_allclose(result.min(axis=0), [1, 1])
        np.testing.assert_allclose(result.max(axis=0), [2, 2])
        for vertex in result:
            self.assertTrue(point_in_support(vertex, first))
            self.assertTrue(point_in_support(vertex, second))
        with self.assertRaises(ValueError):
            intersect_support_polygons([first, second + 5])
        with self.assertRaises(ValueError):
            intersect_support_polygons([first, first + [2, 0]])

    def test_interval_support_handles_mid_hold_contact_switch(self):
        # The first single-support transition is 1.13 s, halfway through this hold.
        start, duration = 1.12, 0.02
        before = self.gait.sample(start)["support_polygon"]
        held = self.gait.support_over_interval(start, duration)
        request = self.gait.sample(start)["feet"][1, :2]
        self.assertTrue(point_in_support(request, before))
        self.assertFalse(point_in_support(request, held))
        command = project_to_support(request, held)
        for time in np.linspace(start, start + duration, 101, endpoint=False):
            self.assertTrue(point_in_support(command, self.gait.sample(time)["support_polygon"]))

    def test_all_default_control_intervals_have_feasible_held_support(self):
        for time in np.arange(0, self.gait.duration, 0.02):
            held = self.gait.support_over_interval(time, 0.02)
            for offset in np.arange(0, 0.02, 0.002):
                physical = self.gait.sample(time + offset)["support_polygon"]
                for vertex in held:
                    self.assertTrue(point_in_support(vertex, physical))


if __name__ == "__main__":
    unittest.main()
