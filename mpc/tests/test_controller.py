"""Full-order rollout parity and the controller's information boundary."""

import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from mpc.controller import DIALController
from mpc.observation import RobotObservation
from sim_env.g1_physics import G1PhysicsEnv


class _RecordingRollout:
    def __init__(self, original):
        self.original = original

    def rollout(self, model, data, initial, controls, **kwargs):
        self.initial = initial.copy()
        self.controls = controls.copy()
        self.arguments = kwargs
        self.states, self.sensors = self.original.rollout(model, data, initial, controls, **kwargs)
        return self.states, self.sensors

    def close(self):
        self.original.close()


class ControllerRolloutTests(unittest.TestCase):
    def make_controller(self, wind_prediction="none", **config):
        env = G1PhysicsEnv(drag_model="none")
        mass_scale = config.pop("_mass_scale", 1.)
        env.model.body_mass[:] *= mass_scale
        env.model.body_inertia[:] *= mass_scale
        mujoco.mj_setConst(env.model, env.data)
        mujoco.mj_forward(env.model, env.data)
        with patch("mpc.controller.G1WalkingReference") as reference_type:
            reference_type.return_value.initial_qpos = env.profile.home_qpos.copy()
            nominal = env.profile.home_ctrl.copy()
            com = env.data.subtree_com[env.profile.root_body_id].copy()
            feet = env.data.site_xpos[list(env.profile.sole_site_ids)].copy()
            def fixed_reference(times):
                length = len(times)
                return {"joint_targets": np.tile(nominal, (length, 1)),
                        "com": np.tile(com, (length, 1)),
                        "com_velocity": np.zeros((length, 3)),
                        "feet": np.tile(feet, (length, 1, 1)),
                        "stance": np.ones((length, 2), dtype=bool)}
            reference_type.return_value.batch.side_effect = fixed_reference
            controller = DIALController(env, {"samples": 2, "horizon_steps": 4,
                                             "knot_intervals": 2, "iterations": 1,
                                             "init_iterations": 1, "threads": 1,
                                             "wind_prediction": wind_prediction, **config})
        controller.rollout = _RecordingRollout(controller.rollout)
        h = controller.dial_config.horizon_steps
        controller._reference_controls = np.tile(env.profile.home_ctrl, (h + 1, 1))
        controller._cost_reference = {
            "com": np.tile(env.data.subtree_com[env.profile.root_body_id], (h, 1)),
            "com_velocity": np.zeros((h, 3)),
            "feet": np.tile(env.data.site_xpos[list(env.profile.sole_site_ids)], (h, 1, 1)),
            "stance": np.ones((h, 2), dtype=bool),
            "joint_targets": np.tile(env.profile.home_ctrl, (h, 1)),
        }
        controller._joint_reference = np.tile(env.profile.home_ctrl, (h, 1))
        controller._initial_state = np.concatenate(([env.data.time], env.data.qpos, env.data.qvel))
        env.data.qacc_warmstart[:] = 0.
        self.addCleanup(controller.close)
        return env, controller

    def test_one_and_multiple_control_steps_match_physical_environment(self):
        env, controller = self.make_controller()
        n = controller.dial_config.horizon_steps
        candidates = np.zeros((1, n + 1, len(controller.active)))
        candidates[0, :, 0] = np.linspace(-.04, .04, n + 1)
        costs = controller._rollout_cost(candidates)
        self.assertTrue(np.all(np.isfinite(costs)))
        self.assertEqual(controller.rollout.states.shape[1], n * env.substeps)
        for index in range(n):
            predicted_sensors = controller.rollout.sensors[0, index * env.substeps]
            np.testing.assert_allclose(predicted_sensors[controller.sensor_slices["com"]],
                                       env.data.subtree_com[env.profile.root_body_id], atol=1e-9)
            np.testing.assert_allclose(predicted_sensors[controller.sensor_slices["left_foot_pos"]],
                                       env.data.site_xpos[env.profile.sole_site_ids[0]], atol=1e-9)
            target = controller._reference_controls[index].copy()
            target[controller.active] += controller.action_scale * candidates[0, index]
            env.step_joint_targets(target)
            actual = np.zeros_like(controller._initial_state)
            mujoco.mj_getState(env.model, env.data, actual, mujoco.mjtState.mjSTATE_FULLPHYSICS)
            predicted = controller.rollout.states[0, (index + 1) * env.substeps - 1]
            np.testing.assert_allclose(predicted, actual, rtol=1e-8, atol=1e-9)

    def test_actions_ignore_external_force_fields_schedule_and_solver_warmstart(self):
        env, controller = self.make_controller()
        observation = RobotObservation.from_data(env.data)
        controller.reset(seed=19)
        expected = controller.act(observation)
        expected_controls = controller.rollout.controls.copy()
        expected_states = controller.rollout.states.copy()

        # Change every external-load source while keeping the permitted robot
        # observation fixed. A force-aware planner or a solver-warmstart leak
        # would change the answer or encounter these deliberately invalid data.
        env.last_body_wind_forces[:] = 1e6
        env.last_body_wind_torques[:] = -1e6
        env.last_disturbance_body_forces[:] = -2e6
        env.last_disturbance_body_torques[:] = 2e6
        env.last_disturbance_joint_torques[:] = 3e6
        env.data.xfrc_applied[:] = 4e6
        env.data.qfrc_applied[:] = -4e6
        env.data.qacc_warmstart[:] = np.nan
        def forbidden_oracle(*args):
            raise AssertionError("Controller must not access wind or disturbance oracles")
        env.wind_field = env.disturbance_callback = forbidden_oracle
        for data in controller.data:
            data.xfrc_applied[:] = 8e6
            data.qfrc_applied[:] = -8e6
            data.qacc_warmstart[:] = np.nan
        controller.reset(seed=19)
        actual = controller.act(observation)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(controller.rollout.controls, expected_controls)
        np.testing.assert_array_equal(controller.rollout.states, expected_states)
        np.testing.assert_array_equal(controller.rollout.arguments["initial_warmstart"],
                                      np.zeros((1, env.model.nv)))
        self.assertEqual(controller.rollout.arguments["control_spec"], int(mujoco.mjtState.mjSTATE_CTRL))
        self.assertEqual(controller.rollout.controls.shape[-1], env.model.nu)
        self.assertFalse(hasattr(controller, "env"))

    def test_observation_copies_only_time_qpos_and_qvel(self):
        env, _ = self.make_controller()
        class RestrictedData:
            time = env.data.time
            qpos = env.data.qpos.copy()
            qvel = env.data.qvel.copy()
            def __getattr__(self, name):
                raise AssertionError(f"Unexpected privileged data read: {name}")
        data = RestrictedData()
        observation = RobotObservation.from_data(data)
        expected_qpos, expected_qvel = data.qpos.copy(), data.qvel.copy()
        data.qpos[:] = 900.
        data.qvel[:] = -900.
        np.testing.assert_array_equal(observation.qpos, expected_qpos)
        np.testing.assert_array_equal(observation.qvel, expected_qvel)
        self.assertEqual(observation.time, data.time)

    def test_force_aware_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            self.make_controller(wind_prediction="current_wrench")

    def test_decoded_controls_obey_same_physical_actuator_clipping(self):
        env, controller = self.make_controller(action_scale=10.)
        n = controller.dial_config.horizon_steps
        candidates = np.ones((1, n + 1, len(controller.active)))
        controller._rollout_cost(candidates)
        expected = env.profile.home_ctrl.copy()
        expected[controller.active] += controller.action_scale
        expected = np.clip(expected, env.model.actuator_ctrlrange[:, 0], env.model.actuator_ctrlrange[:, 1])
        np.testing.assert_allclose(controller.rollout.controls,
                                   np.broadcast_to(expected, controller.rollout.controls.shape))

    def test_added_sensors_do_not_change_model_physical_parameters(self):
        env, controller = self.make_controller()
        self.assertEqual((env.model.nq, env.model.nv, env.model.nu),
                         (controller.model.nq, controller.model.nv, controller.model.nu))
        self.assertGreater(controller.model.nsensordata, env.model.nsensordata)
        for name in ("body_mass", "body_inertia", "geom_friction", "dof_damping",
                     "actuator_gainprm", "actuator_biasprm", "actuator_ctrlrange", "actuator_forcerange"):
            np.testing.assert_array_equal(getattr(controller.model, name), getattr(env.model, name))

    def test_changed_plant_masses_update_planning_model_derived_constants(self):
        env, controller = self.make_controller(_mass_scale=1.3)
        np.testing.assert_allclose(controller.model.body_subtreemass, env.model.body_subtreemass)
        self.assertAlmostEqual(float(controller.model.body_subtreemass[env.profile.root_body_id]),
                               1.3 * env.profile.mass)


if __name__ == "__main__":
    unittest.main()
