"""Physical recording and failure reporting; no assertion of walking skill."""

import json
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from mpc.run import run_episode
from sim_env.g1_physics import G1PhysicsEnv


class DiagnosticController:
    name = "standing diagnostic for recorder test"
    command_speed = .2
    last_plan_info = {}

    def __init__(self, env):
        self.env = env

    def act(self, observation):
        return self.env.profile.home_ctrl.copy()


class RecorderTests(unittest.TestCase):
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
