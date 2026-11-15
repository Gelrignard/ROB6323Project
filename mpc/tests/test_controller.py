"""Full-order rollout parity and the controller's information boundary."""

import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from mpc.controller import DIALController
from sim_env.g1_physics import G1PhysicsEnv


class _RecordingRollout:
    def __init__(self, original):
        self.original = original

    def rollout(self, model, data, initial, controls, **kwargs):
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
        controller._initial_state = np.zeros(mujoco.mj_stateSize(env.model, mujoco.mjtState.mjSTATE_FULLPHYSICS))
        mujoco.mj_getState(env.model, env.data, controller._initial_state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
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

    def test_current_wrench_is_repeated_and_future_disturbances_are_not_read(self):
        env, controller = self.make_controller(wind_prediction="current_wrench")
        body = env.profile.torso_body_id
        env.last_body_wind_forces[body] = [3., -2., .5]
        env.last_body_wind_torques[body] = [.1, -.2, .3]
        wrench = np.concatenate((env.last_body_wind_forces, env.last_body_wind_torques), axis=-1)
        n = controller.dial_config.horizon_steps
        candidates = np.zeros((1, n + 1, len(controller.active)))
        def oracle(_):
            raise AssertionError("Planner must not query future disturbance events")
        env.disturbance_callback = oracle
        controller._rollout_cost(candidates)
        expected = np.broadcast_to(wrench.reshape(-1), controller.rollout.controls[..., env.model.nu:].shape)
        np.testing.assert_array_equal(controller.rollout.controls[..., env.model.nu:], expected)
        self.assertTrue(controller.rollout.arguments["control_spec"] & int(mujoco.mjtState.mjSTATE_XFRC_APPLIED))

        # Compare against a plant with this same constant current wrench. This
        # is a parity test of the planner's documented zero-order load hold,
        # not a claim to forecast changing aerodynamic forces exactly.
        env.disturbance_callback = None
        def constant_wind():
            mujoco.mj_forward(env.model, env.data)
            env.data.xfrc_applied[:] = wrench
        with patch.object(env, "_apply_wind", side_effect=constant_wind):
            for index in range(n):
                env.step_joint_targets(env.profile.home_ctrl)
                actual = np.zeros_like(controller._initial_state)
                mujoco.mj_getState(env.model, env.data, actual, mujoco.mjtState.mjSTATE_FULLPHYSICS)
                predicted = controller.rollout.states[0, (index + 1) * env.substeps - 1]
                np.testing.assert_allclose(predicted, actual, rtol=1e-8, atol=1e-9)

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
