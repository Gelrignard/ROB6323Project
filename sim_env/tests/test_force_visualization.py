"""Drag-only display conventions, conservation and no-state-mutation checks."""

import unittest

import mujoco
import numpy as np

from sim_env.environment import SimulationConfig
from sim_env.visualize import ForceVisualizer, drag_vector, surface_samples


class ForceVisualizationTests(unittest.TestCase):
    def setUp(self):
        self.config = SimulationConfig(mass=10, gravity=10, height=1)
        self.row = {"time": 0.0, "state": np.array([2, 0, 3, 0]),
                    "wind_force": np.array([4, -3]), "ground_force": np.array([-10, 20])}
        self.visual = ForceVisualizer([self.row], self.config)

    def test_lumped_drag_only_and_isotropic_arrow(self):
        np.testing.assert_allclose(drag_vector(self.row), [4, -3, 0])
        arrows = self.visual.arrow_specs(self.row)
        self.assertEqual(len(arrows), 1)
        np.testing.assert_allclose(arrows[0].origin, [2, 3, 1])
        np.testing.assert_allclose(arrows[0].endpoint, [2.68, 2.49, 1])
        self.assertEqual(arrows[0].name, "drag")

    def test_surface_profile_conserves_local_load_and_uses_integrated_drag(self):
        row = {**self.row, "drag_model": "surface", "drag_origin": [2, 3, 1],
               "drag_force": [2, 2, 1], "surface_drag": {
                   "points": np.array([[2, 3, .2], [2, 3, .8], [2, 3, 1.2]]),
                   "forces": np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]]),
                   "pressure": np.array([10, 20, 5])}}
        visual = ForceVisualizer([row], self.config)
        np.testing.assert_allclose(drag_vector(row), [2, 2, 1])
        self.assertAlmostEqual(float(np.sum(visual._height_profile(surface_samples(row)))), 5.0)
        self.assertEqual(visual.metadata()["pressure_color_max_Pa"], 20)
        self.assertEqual(visual.metadata()["drag_models"], ["surface"])

    def test_zero_drag_produces_no_other_force_arrows(self):
        row = {**self.row, "wind_force": np.zeros(2)}
        model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
        scene = mujoco.MjvScene(model, maxgeom=8)
        self.visual.decorate(scene, row)
        self.assertEqual(scene.ngeom, 0)

    def test_local_vectors_are_arrows_pointing_into_surface_with_linear_lengths(self):
        points = np.array([[2, 3, .2], [2, 3, .8], [2, 3, 1.2], [2.1, 3, .5]])
        forces = np.array([[2, 0, 0], [0, -2, 0], [0, 0, 1], [0, 0, 0]])
        row = {**self.row, "drag_model": "surface", "drag_force": [2, -2, 1],
               "surface_drag": {"points": points, "forces": forces, "pressure": [10, 20, 5, 0]}}
        visual = ForceVisualizer([row], self.config)
        arrows = visual.local_arrow_specs(row)
        self.assertEqual(len(arrows), 3)
        for arrow in arrows:
            index = int(arrow.name.rsplit("_", 1)[1])
            np.testing.assert_allclose(arrow.endpoint, points[index], atol=1e-14)
            np.testing.assert_allclose(arrow.endpoint - arrow.origin,
                                       visual.local_arrow_scale * forces[index], atol=1e-14)
            self.assertGreater(np.dot(arrow.endpoint - arrow.origin, forces[index]), 0)
        lengths = [np.linalg.norm(arrow.endpoint - arrow.origin) for arrow in arrows]
        self.assertAlmostEqual(lengths[0] / lengths[2], 2.0)
        model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
        scene = mujoco.MjvScene(model, maxgeom=8)
        visual.decorate(scene, row)
        self.assertEqual(scene.ngeom, 4)
        self.assertTrue(all(scene.geoms[i].type == mujoco.mjtGeom.mjGEOM_ARROW for i in range(scene.ngeom)))
        zero = {**row, "drag_force": [0, 0, 0], "surface_drag": {"points": points, "forces": np.zeros_like(forces), "pressure": np.zeros(4)}}
        scene.ngeom = 0
        visual.decorate(scene, zero)
        self.assertEqual(scene.ngeom, 0)

    def test_side_panel_preserves_rgb_and_does_not_mutate_records(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        before = self.row["wind_force"].copy()
        result = self.visual.compose(frame, self.row)
        self.assertEqual(result.shape, (720, 1760, 3))
        np.testing.assert_array_equal(result[:, :1280], frame)
        np.testing.assert_array_equal(self.row["wind_force"], before)
        with self.assertRaises(ValueError):
            drag_vector({**self.row, "wind_force": [np.nan, 0]})


if __name__ == "__main__":
    unittest.main()
