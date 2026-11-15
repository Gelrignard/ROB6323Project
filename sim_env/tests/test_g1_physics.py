"""Verify authentic G1 model semantics and physically integrated motion."""

import unittest
import tempfile
from pathlib import Path
import mujoco
import numpy as np

from sim_env.g1_model import load_g1_model, get_g1_profile
from sim_env.g1_physics import G1PhysicsEnv, run_standing


class G1PhysicsTests(unittest.TestCase):
    def test_model_preserves_free_base_and_position_actuators(self):
        model = load_g1_model()
        profile = get_g1_profile(model)
        self.assertEqual((model.nq, model.nv, model.nu), (36, 35, 29))
        self.assertEqual(model.jnt_type[0], mujoco.mjtJoint.mjJNT_FREE)
        self.assertTrue(np.all(model.actuator_gainprm[:, 0] > 0))
        self.assertTrue(np.all(model.actuator_biasprm[:, 1] < 0))
        self.assertAlmostEqual(profile.mass, 33.341142, places=5)
        self.assertTrue(.6 < profile.com_height < .8)
        np.testing.assert_allclose(profile.sole_positions[:, 2], 0, atol=1e-8)

    def test_real_contacts_and_motion_without_state_reset(self):
        env = G1PhysicsEnv()
        initial = env.data.qpos.copy()
        for _ in range(10):
            env.step_joint_targets(env.profile.home_ctrl)
        self.assertAlmostEqual(env.data.time, .2)
        self.assertGreater(env.max_contacts, 0)
        self.assertGreater(np.linalg.norm(env.data.qpos - initial), 1e-5)
        self.assertIsNone(env.failure_reason())
        self.assertTrue(np.all(env.data.warning.number == 0))

    def test_torso_wind_changes_physical_motion(self):
        calm, windy = G1PhysicsEnv(wind_scale=0), G1PhysicsEnv(wind_scale=1)
        for _ in range(10):
            calm.step_joint_targets(calm.profile.home_ctrl)
            windy.step_joint_targets(windy.profile.home_ctrl)
        self.assertGreater(windy.max_wind_force, 10)
        self.assertGreater(np.linalg.norm(windy.data.qpos[:3] - calm.data.qpos[:3]), 1e-4)
        self.assertEqual(np.count_nonzero(np.linalg.norm(windy.data.xfrc_applied[:, :3], axis=1)), 1)
        with self.assertRaises(ValueError):
            windy.step_joint_targets([0, 0])

    def test_surface_loads_preserve_patch_resultant_and_moment(self):
        env = G1PhysicsEnv(wind_scale=1, drag_model="surface")
        env._apply_wind()
        result = env.surface_drag.evaluate(env.data, wind_scale=1)
        applied_forces = env.data.xfrc_applied[:, :3]
        applied_torques = env.data.xfrc_applied[:, 3:]
        self.assertGreater(np.count_nonzero(np.linalg.norm(applied_forces, axis=1) > 1e-8), 5)
        self.assertGreater(np.linalg.norm(applied_torques), 1e-8)
        np.testing.assert_allclose(applied_forces.sum(axis=0), result.patch_forces.sum(axis=0), atol=1e-10)
        applied_moment = np.sum(np.cross(env.data.xipos - result.com, applied_forces) + applied_torques, axis=0)
        patch_moment = np.sum(np.cross(result.points - result.com, result.patch_forces), axis=0)
        np.testing.assert_allclose(applied_moment, patch_moment, atol=1e-10)
        np.testing.assert_allclose(env.last_wind_torque_com, applied_moment, atol=1e-10)
        # Surface evaluation is exclusive: adding the former torso load would
        # violate the resultant comparison above.
        with self.assertRaises(ValueError):
            G1PhysicsEnv(drag_model="both")

    def test_surface_wind_changes_actual_free_base_motion(self):
        calm = G1PhysicsEnv(wind_scale=0, drag_model="surface")
        windy = G1PhysicsEnv(wind_scale=1, drag_model="surface")
        initial = windy.data.qpos.copy()
        for _ in range(10):
            calm.step_joint_targets(calm.profile.home_ctrl)
            observation = windy.step_joint_targets(windy.profile.home_ctrl)
        self.assertAlmostEqual(windy.data.time, .2)
        self.assertAlmostEqual(observation["wind_force_sample_time"], .198)
        self.assertGreater(np.linalg.norm(windy.data.qpos - initial), 1e-5)
        self.assertGreater(np.linalg.norm(windy.data.qpos[:3] - calm.data.qpos[:3]), 1e-7)
        self.assertGreater(windy.max_wind_force, .1)
        self.assertGreater(windy.max_wind_torque_com, .01)
        self.assertGreater(windy.max_contacts, 0)
        self.assertTrue(np.all(windy.data.warning.number == 0))

    def test_saved_surface_loads_use_physics_sample_times_and_body_moments(self):
        with tempfile.TemporaryDirectory() as temporary:
            summary = run_standing(duration=.04, wind_scale=1, drag_model="surface", output=Path(temporary))
            with np.load(Path(temporary) / "physical_state.npz") as saved:
                np.testing.assert_allclose(saved["time"], [0, .02, .04])
                np.testing.assert_allclose(saved["wind_sample_time_s"], np.arange(20) * .002)
                forces, torques = saved["wind_body_forces_N"], saved["wind_body_torques_Nm"]
                np.testing.assert_allclose(forces.sum(axis=1), saved["wind_force_N"], atol=1e-10)
                arms = saved["wind_body_com_m"] - saved["wind_com_m"][:, None, :]
                np.testing.assert_allclose(np.sum(np.cross(arms, forces) + torques, axis=1),
                                           saved["wind_torque_about_com_Nm"], atol=1e-10)
                self.assertEqual(saved["qvel"].shape, (3, 35))
            self.assertEqual(summary["drag_model"], "surface")
            self.assertEqual(summary["wind_load_samples"], 20)
            self.assertTrue(summary["completed_requested_duration"])
            self.assertIsNone(summary["projected_area_m2_assumption"])


if __name__ == "__main__":
    unittest.main()
