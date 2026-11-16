"""Native playback must bypass aerodynamic computation and force decoration."""

from contextlib import ExitStack, redirect_stdout
from dataclasses import asdict
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from mpc.render import frame_drag_record, render_run, setup_drag_visualization
from sim_env.config import TornadoConfig


class RenderTests(unittest.TestCase):
    def test_native_setup_and_frame_record_never_touch_aerodynamics(self):
        with patch("mpc.render.SurfaceDrag") as surface, patch("mpc.render.make_wind_field") as wind:
            self.assertEqual(setup_drag_visualization(object(), {"drag_model": "none"}), (None, None))
            surface.assert_not_called()
            wind.assert_not_called()
        drag = MagicMock()
        record = frame_drag_record(object(), {"drag_model": "none"}, drag, object())
        drag.evaluate.assert_not_called()
        self.assertEqual(record["drag_model"], "none")
        np.testing.assert_array_equal(record["drag_force"], np.zeros(3))
        self.assertNotIn("surface_drag", record)

    def test_surface_setup_and_frame_keep_existing_aerodynamic_path(self):
        summary = {"drag_model": "surface", "wind_config": asdict(TornadoConfig()),
                   "wind_scale": .6, "wind_field": {"kind": "calm"},
                   "surface_model": {"quadrature_patches": 128}}
        model, data = object(), object()
        result = SimpleNamespace(com=np.ones(3), total_force=np.arange(3), points=np.ones((2, 3)),
                                 patch_forces=np.ones((2, 3)), pressure_Pa=np.ones(2))
        with patch("mpc.render.SurfaceDrag") as surface, patch("mpc.render.make_wind_field") as wind:
            surface.return_value.evaluate.return_value = result
            drag, provider = setup_drag_visualization(model, summary)
            record = frame_drag_record(data, summary, drag, provider)
            self.assertIs(surface.call_args.args[0], model)
            self.assertEqual(surface.call_args.kwargs["max_patches"], 128)
            wind.assert_called_once_with({"kind": "calm"})
            drag.evaluate.assert_called_once_with(data, wind_scale=.6, wind_field=provider)
            np.testing.assert_array_equal(record["drag_force"], result.total_force)

    def test_native_render_uses_opaque_robot_and_no_force_visualizer(self):
        # All rendering/encoding and MuJoCo functions are mocks. This exercises
        # the orchestration without running physics or generating video media.
        with TemporaryDirectory() as temporary, ExitStack() as stack:
            directory = Path(temporary)
            summary = {"drag_model": "none", "control_dt_s": .02, "scenario": "calm",
                       "command_speed_mps": .18, "wind_field": None}
            (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            np.savez(directory / "trajectory.npz", time=[0., .02], qpos=np.zeros((2, 3)),
                     qvel=np.zeros((2, 6)), ctrl=np.zeros((2, 1)), foot_contact=np.ones((2, 2)))
            data = SimpleNamespace(qpos=np.zeros(3), qvel=np.zeros(6), ctrl=np.zeros(1), time=0.)
            camera = SimpleNamespace(lookat=np.zeros(3))
            stack.enter_context(patch("mpc.render.mujoco.MjModel"))
            stack.enter_context(patch("mpc.render.mujoco.MjData", return_value=data))
            stack.enter_context(patch("mpc.render.mujoco.mj_forward"))
            stack.enter_context(patch("mpc.render.mujoco.MjvCamera", return_value=camera))
            renderer = stack.enter_context(patch("mpc.render.mujoco.Renderer"))
            writer = stack.enter_context(patch("mpc.render.imageio.get_writer"))
            panel = stack.enter_context(patch("mpc.render._panel", return_value=np.zeros((2, 2, 3), dtype=np.uint8)))
            stack.enter_context(patch("mpc.render.Image.fromarray"))
            surface = stack.enter_context(patch("mpc.render.SurfaceDrag"))
            wind = stack.enter_context(patch("mpc.render.make_wind_field"))
            arrows = stack.enter_context(patch("mpc.render.ForceVisualizer"))
            opacity = stack.enter_context(patch("mpc.render.set_robot_opacity"))
            with redirect_stdout(StringIO()):
                output = render_run(directory)
            surface.assert_not_called()
            wind.assert_not_called()
            arrows.assert_not_called()
            self.assertEqual(opacity.call_count, 2)
            self.assertTrue(all(call.kwargs["opacity"] == 1. for call in opacity.call_args_list))
            self.assertTrue(all(call.args[-1] is None for call in panel.call_args_list))
            renderer.return_value.close.assert_called_once()
            writer.return_value.close.assert_called_once()
            metadata = json.loads(output.with_suffix(".render.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["drag_model"], "none")
            self.assertEqual(metadata["robot_opacity"], 1.)
            self.assertEqual(metadata["force_arrows"], "none; aerodynamics disabled")
            self.assertIn("no aerodynamic model or provider", metadata["drag_visualization"])


if __name__ == "__main__":
    unittest.main()
