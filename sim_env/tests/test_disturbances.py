"""External perturbations have reproducible timing and physical force units."""

import unittest

import mujoco
import numpy as np

from sim_env.disturbances import DisturbanceConfig, DisturbanceEvent, DisturbanceScheduler
from sim_env.g1_model import load_g1_model
from sim_env.g1_physics import G1PhysicsEnv
from sim_env.wind_fields import make_wind_field


class DisturbanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_g1_model()

    def test_seeded_schedule_does_not_depend_on_sampling_order(self):
        config = DisturbanceConfig(enabled=True)
        first, second = DisturbanceScheduler(self.model, config), DisturbanceScheduler(self.model, config)
        self.assertEqual(first.reset(seed=29, horizon_s=20), second.reset(seed=29, horizon_s=20))
        self.assertGreater(len(first.events), 0)
        for time in (3.1, .0, 10.5, 1.8, 3.1):
            a, b = first.sample(time), second.sample(time)
            np.testing.assert_array_equal(a.body_forces, b.body_forces)
            np.testing.assert_array_equal(a.body_torques, b.body_torques)
            np.testing.assert_array_equal(a.joint_torques, b.joint_torques)
        self.assertNotEqual(first.events, second.reset(seed=30, horizon_s=20))
        coarse = DisturbanceScheduler(self.model, config)
        coarse.reset(seed=29, horizon_s=20, strength=.5)
        for original, scaled in zip(first.events, coarse.events):
            self.assertEqual(original['start_s'], scaled['start_s'])
            np.testing.assert_allclose(np.asarray(original['amplitude']) * .5, scaled['amplitude'])

    def test_force_impulse_and_joint_dof_mapping(self):
        scheduler = DisturbanceScheduler(self.model)
        event = DisturbanceEvent('link_force', 'torso_link', .2, .4, (20, -10, 5))
        joint = DisturbanceEvent('joint_torque', 'left_knee_joint', .2, .4, (8,))
        scheduler.reset(seed=1, horizon_s=1, events=[event, joint])
        body = self.model.body('torso_link').id
        times = np.linspace(.2, .6, 2001)
        values = np.array([scheduler.sample(t).body_forces[body] for t in times])
        np.testing.assert_allclose(np.trapezoid(values, times, axis=0), event.impulse(), atol=2e-6)
        peak = scheduler.sample(.4)
        np.testing.assert_allclose(peak.body_forces[body], [20, -10, 5])
        dof = self.model.jnt_dofadr[self.model.joint('left_knee_joint').id]
        self.assertAlmostEqual(peak.joint_torques[dof], 8)
        np.testing.assert_array_equal(peak.joint_torques[:6], 0)
        self.assertEqual(np.count_nonzero(peak.joint_torques), 1)
        self.assertEqual(np.count_nonzero(scheduler.sample(.7).body_forces), 0)
        with self.assertRaises(ValueError):
            scheduler.reset(horizon_s=1, events=[{'kind': 'joint_torque', 'target': 'floating_base_joint',
                                                'start_s': 0, 'duration_s': .1, 'amplitude': [2]}])

    def test_loads_add_to_wind_and_expire_without_state_reset(self):
        env = G1PhysicsEnv(wind_scale=1, drag_model='lumped',
                           wind_field=make_wind_field({'kind': 'uniform', 'velocity_mps': [0, 4, 0]}))
        scheduler = DisturbanceScheduler(env.model)
        scheduler.reset(horizon_s=1, events=[
            DisturbanceEvent('link_force', 'left_hip_yaw_link', 0, .02, (10, 0, 0), 'constant'),
            DisturbanceEvent('joint_torque', 'left_knee_joint', 0, .02, (2,), 'constant')])
        env.disturbance_callback = scheduler.sample
        initial = env.data.qpos.copy()
        observation = env.step_joint_targets(env.profile.home_ctrl)
        np.testing.assert_allclose(env.data.xfrc_applied[:, :3], env.last_body_wind_forces + env.last_disturbance_body_forces)
        np.testing.assert_allclose(env.data.qfrc_applied, env.last_disturbance_joint_torques)
        self.assertAlmostEqual(observation['disturbance_sample_time'], .018)
        self.assertGreater(np.linalg.norm(env.data.qpos - initial), 1e-6)
        self.assertGreater(np.linalg.norm(env.last_body_wind_forces), 0)
        env.step_joint_targets(env.profile.home_ctrl)
        np.testing.assert_array_equal(env.data.qfrc_applied, 0)
        np.testing.assert_array_equal(env.last_disturbance_body_forces, 0)

    def test_explicit_no_aerodynamics_retains_disturbance_forces(self):
        env = G1PhysicsEnv(drag_model='none')
        scheduler = DisturbanceScheduler(env.model)
        scheduler.reset(horizon_s=1, events=[DisturbanceEvent('link_torque', 'torso_link', 0, .1, (0, 1, 0), 'constant')])
        env.disturbance_callback = scheduler.sample
        env.step_joint_targets(env.profile.home_ctrl)
        np.testing.assert_array_equal(env.last_wind_force, 0)
        self.assertEqual(np.count_nonzero(env.data.xfrc_applied[:, 3:]), 1)
        self.assertAlmostEqual(env.data.time, .02)


if __name__ == '__main__':
    unittest.main()
