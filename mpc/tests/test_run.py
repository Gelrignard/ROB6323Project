"""Physical recording and failure reporting; no assertion of walking skill."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from mpc.run import run_case, run_episode
from sim_env.g1_physics import G1PhysicsEnv


class DiagnosticController:
    name = "standing diagnostic for recorder test"
    command_speed = .2
    last_plan_info = {}

    def __init__(self, env):
        self.env = env

    def act(self, observation):
        return self.env.profile.home_ctrl.copy()


class CaseDiagnosticController(DiagnosticController):
    """Exercise scenario routing without constructing the walking optimizer."""

    def __init__(self, env, config):
        super().__init__(env)
        self.initial_qpos = env.profile.home_qpos.copy()
        self.initial_qpos[:2] = [1., 5.]
        self.closed = False

    def reset(self, seed):
        self.seed = seed

    def act(self, observation):
        targets = super().act(observation)
        targets[self.env.model.actuator("left_elbow_joint").id] += .05
        return targets

    def close(self):
        self.closed = True


class RecorderTests(unittest.TestCase):
    def test_calm_case_disables_air_and_injected_loads_even_during_joint_motion(self):
        controllers = []
        def diagnostic(env, config):
            controllers.append(CaseDiagnosticController(env, config))
            return controllers[-1]

        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(controller_config=None, duration=.06, seed=7,
                                   wind_scale=3., surface_patches=128,
                                   disturbance_strength=1., output=folder, video=False)
            with patch("mpc.controller.DIALController", side_effect=diagnostic), \
                 patch("mpc.run.make_wind_field", side_effect=AssertionError("Calm must not construct a field")), \
                 patch("sim_env.surface_drag.SurfaceDrag", side_effect=AssertionError("Calm must not construct surface drag")):
                summary = run_case(args, "calm")
            controller = controllers[0]
            env = controller.env
            self.assertTrue(controller.closed)
            self.assertEqual(env.drag_model, "none")
            self.assertEqual(env.wind_scale, 0.)
            self.assertIsNone(env.wind_field)
            self.assertIsNone(env.surface_drag)
            self.assertIsNone(env.disturbance_callback)
            self.assertEqual(env.model.opt.density, 0.)
            self.assertEqual(env.model.opt.viscosity, 0.)
            np.testing.assert_array_equal(env.model.opt.wind, np.zeros(3))
            self.assertTrue(summary["completed_requested_duration"])
            self.assertIsNone(summary["wind_field"])
            self.assertIsNone(summary["surface_model"])
            self.assertFalse(summary["disturbances"]["config"]["enabled"])
            self.assertEqual(summary["disturbances"]["events"], [])
            with np.load(Path(folder) / "calm" / "trajectory.npz") as states:
                self.assertGreater(np.max(np.abs(states["qvel"][:, 6:])), 1e-3)
                for key in ("wind_force_N", "wind_torque_about_com_Nm", "wind_body_forces_N",
                            "wind_body_torques_Nm", "disturbance_body_forces_N",
                            "disturbance_body_torques_Nm", "disturbance_joint_torques_Nm"):
                    with self.subTest(load=key):
                        self.assertEqual(len(states[key]), 30)
                        np.testing.assert_array_equal(states[key], np.zeros_like(states[key]))

    def test_random_and_tornado_cases_retain_surface_load_routing(self):
        for scenario, field_kind, wind_scale in (("random", "calm", 1.), ("tornado", "tornado", 2.4)):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as folder:
                args = SimpleNamespace(controller_config=None, duration=4., seed=7,
                                       wind_scale=2.4, surface_patches=128,
                                       disturbance_strength=1., output=folder, video=False)
                def inspect_routing(env, controller, **kwargs):
                    self.assertEqual(env.drag_model, "surface")
                    self.assertIsNotNone(env.surface_drag)
                    self.assertEqual(env.wind_field.config.kind, field_kind)
                    self.assertEqual(env.wind_scale, wind_scale)
                    self.assertEqual(kwargs["wind_metadata"]["kind"], field_kind)
                    disturbances = kwargs["disturbance_metadata"]
                    self.assertEqual(disturbances["config"]["enabled"], scenario == "random")
                    self.assertEqual(bool(disturbances["events"]), scenario == "random")
                    if scenario == "random":
                        event = disturbances["events"][0]
                        sample = env.disturbance_callback(event["start_s"] + event["duration_s"] / 2)
                        self.assertGreater(np.linalg.norm(sample.body_forces), 0.)
                    return {"scenario": scenario, "elapsed_simulation_s": 0., "forward_progress_m": 0.,
                            "walking_success": False, "failure_reason": None, "wall_seconds": 0.}
                with patch("mpc.controller.DIALController", side_effect=CaseDiagnosticController), \
                     patch("mpc.run.run_episode", side_effect=inspect_routing) as episode:
                    run_case(args, scenario)
                episode.assert_called_once()

    def test_records_distinct_control_and_load_clocks_and_never_calls_standing_walking(self):
        env = G1PhysicsEnv(drag_model="none", record_wind_history=True)
        initial = env.data.qpos.copy()
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "run"
            summary = run_episode(env, DiagnosticController(env), duration=.06,
                                  output=destination, scenario="calm", seed=7)
            self.assertTrue(summary["completed_requested_duration"])
            self.assertFalse(summary["walking_success"])
            self.assertEqual(summary["controller"], DiagnosticController.name)
            np.testing.assert_array_equal(initial, summary["initial_qpos"])
            with np.load(destination / "trajectory.npz") as states:
                np.testing.assert_allclose(states["time"], [0., .02, .04, .06])
                self.assertEqual(states["qpos"].shape, (4, env.model.nq))
                self.assertEqual(states["foot_contact"].shape, (4, 2))
                np.testing.assert_allclose(states["wind_sample_time_s"], np.arange(30) * .002)
                np.testing.assert_allclose(states["disturbance_sample_time_s"], np.arange(30) * .002)
            restored = mujoco.MjModel.from_binary_path(str(destination / "model.mjb"))
            np.testing.assert_array_equal(restored.actuator_gainprm, env.model.actuator_gainprm)
            self.assertEqual(len(json.loads((destination / "planner.json").read_text())), 3)

    def test_bad_controller_retains_failure_and_does_not_overwrite_results(self):
        env = G1PhysicsEnv(drag_model="none", record_wind_history=True)
        controller = DiagnosticController(env)
        controller.act = lambda observation: np.full(env.model.nu, np.nan)
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "run"
            result = run_episode(env, controller, duration=.1, output=destination, scenario="calm", seed=7)
            self.assertFalse(result["completed_requested_duration"])
            self.assertFalse(result["walking_success"])
            self.assertIn("invalid joint-position targets", result["failure_reason"])
            self.assertEqual(result["elapsed_simulation_s"], 0.)
            with self.assertRaises(FileExistsError):
                run_episode(env, controller, duration=.1, output=destination, scenario="calm", seed=7)


if __name__ == "__main__":
    unittest.main()
