"""Physical Gymnasium task, deterministic randomization and policy contract."""

from dataclasses import replace
import unittest
from unittest.mock import patch

from gymnasium.utils.env_checker import check_env
import mujoco
import numpy as np

from sim_env.rl_config import RLConfig
from sim_env.rl_env import G1RobustEnv


class RobustEnvTests(unittest.TestCase):
    def simple_config(self, **kwargs):
        return replace(RLConfig.preset("calm"), drag_model="none", episode_duration_s=.2, **kwargs)

    def test_gym_api_and_stable_policy_schema(self):
        env = G1RobustEnv(self.simple_config())
        try:
            check_env(env, skip_render_check=True)
            obs, info = env.reset(seed=1)
            self.assertEqual(obs.shape, (99,))
            self.assertEqual(obs.dtype, np.float32)
            self.assertEqual(env.action_space.shape, (29,))
            self.assertEqual(len(env.schema()["joint_names"]), 29)
            schema = env.schema()
            env.config = replace(env.config, wind={"kind": "tornado"}, curriculum_strength=.2)
            self.assertEqual(schema, env.schema())
            np.testing.assert_array_equal(env.action_from_joint_targets(env.nominal_targets), np.zeros(29))
            with self.assertRaises(ValueError):
                env.action_from_joint_targets(env.nominal_targets + 2)
            self.assertFalse(info["is_success"])
        finally:
            env.close()

    def test_seed_replays_training_state_domain_noise_and_disturbances(self):
        config = replace(RLConfig.preset("train_random"), episode_duration_s=.3,
                         disturbances={"enabled": True, "settle_time_s": 0.0,
                                       "interval_s": (.03, .04), "duration_s": (.03, .04),
                                       "link_force_max_n": 10., "link_torque_max_nm": 1., "joint_torque_max_nm": 1.})
        first, second = G1RobustEnv(config), G1RobustEnv(config)
        try:
            a, ia = first.reset(seed=91)
            b, ib = second.reset(seed=91)
            np.testing.assert_array_equal(a, b)
            self.assertEqual(ia["domain_parameters"], ib["domain_parameters"])
            self.assertEqual(ia["disturbance_events"], ib["disturbance_events"])
            self.assertTrue(ia["disturbance_events"])
            for _ in range(8):
                a, ra, ta, xa, _ = first.step(np.zeros(29))
                b, rb, tb, xb, _ = second.step(np.zeros(29))
                np.testing.assert_array_equal(a, b)
                self.assertEqual((ra, ta, xa), (rb, tb, xb))
                np.testing.assert_array_equal(first.data.qpos, second.data.qpos)
                if ta or xa:
                    break
            _, changed = second.reset(seed=92)
            self.assertNotEqual(ia["disturbance_events"], changed["disturbance_events"])
        finally:
            first.close()
            second.close()

    def test_joint_target_adapter_respects_physical_and_normalized_limits(self):
        env = G1RobustEnv(self.simple_config())
        try:
            knee = env.joint_names.index("left_knee_joint")
            requested = env.nominal_targets.copy()
            requested[knee] = env.model.actuator_ctrlrange[knee, 0] - .01
            # This physical violation is still within the normalized action range.
            self.assertLess(abs((requested[knee] - env.nominal_targets[knee]) / env.config.action_scale), 1.)
            with self.assertRaisesRegex(ValueError, "physical actuator"):
                env.action_from_joint_targets(requested, clip=False)
            action = env.action_from_joint_targets(requested, clip=True)
            targets = env.nominal_targets + env.config.action_scale * action
            np.testing.assert_allclose(targets[knee], env.model.actuator_ctrlrange[knee, 0], atol=1e-7)
            extreme = env.action_from_joint_targets(np.full(29, 100.), clip=True)
            clipped_targets = env.nominal_targets + env.config.action_scale * extreme
            self.assertTrue(np.all(np.abs(extreme) <= 1.))
            self.assertTrue(np.all(clipped_targets <= env.model.actuator_ctrlrange[:, 1] + 1e-7))
            self.assertTrue(np.all(clipped_targets >= env.model.actuator_ctrlrange[:, 0] - 1e-7))
        finally:
            env.close()

    def test_domain_reset_does_not_accumulate_and_updates_priority_feet(self):
        env = G1RobustEnv(replace(RLConfig.preset("train_random"), episode_duration_s=.2))
        try:
            _, info = env.reset(seed=4)
            mass, inertia = env.model.body_mass.copy(), env.model.body_inertia.copy()
            friction = env.model.geom_friction.copy()
            env.reset(seed=9)
            env.reset(seed=4)
            np.testing.assert_array_equal(env.model.body_mass, mass)
            np.testing.assert_array_equal(env.model.body_inertia, inertia)
            np.testing.assert_array_equal(env.model.geom_friction, friction)
            foot_geoms = np.flatnonzero(env.model.geom_priority > 0)
            self.assertGreater(len(foot_geoms), 0)
            np.testing.assert_allclose(env.model.geom_friction[foot_geoms, 0], info["domain_parameters"]["friction"])
            env.config = replace(env.config, curriculum_strength=0.)
            env.reset(seed=4)
            np.testing.assert_array_equal(env.model.body_mass, env._nominal["body_mass"])
            np.testing.assert_array_equal(env.model.actuator_gainprm, env._nominal["actuator_gainprm"])
            np.testing.assert_array_equal(env.model.geom_friction, env._nominal["geom_friction"])
            self.assertEqual(env.domain_parameters["action_delay_steps"], 0)
        finally:
            env.close()

    def test_curriculum_updates_take_effect_only_on_reset(self):
        env = G1RobustEnv(replace(RLConfig.preset("train_random"), episode_duration_s=.2))
        try:
            env.reset(seed=1)
            env.set_curriculum_strength(0.0)
            self.assertEqual(env.config.curriculum_strength, 1.0)
            _, info = env.reset(seed=1)
            self.assertEqual(env.config.curriculum_strength, 0.0)
            self.assertFalse(info["initialization_randomization"]["enabled"])
            np.testing.assert_array_equal(env.model.geom_friction, env._nominal["geom_friction"])
        finally:
            env.close()

    def test_actor_does_not_observe_wind_or_future_disturbance_schedule(self):
        calm = G1RobustEnv(self.simple_config())
        disturbed = G1RobustEnv(self.simple_config(wind={"kind": "uniform"}, disturbances={"enabled": True}))
        try:
            a, _ = calm.reset(seed=44)
            b, _ = disturbed.reset(seed=44)
            np.testing.assert_array_equal(a, b)
            names = [field["name"] for field in calm.schema()["observation"]["fields"]]
            self.assertFalse(any("wind" in name or "disturbance" in name for name in names))
        finally:
            calm.close()
            disturbed.close()

    def test_evaluation_initial_pose_varies_by_seed_without_training_randomization(self):
        env = G1RobustEnv(self.simple_config())
        try:
            a, first = env.reset(seed=15)
            b, second = env.reset(seed=16)
            repeated, replay = env.reset(seed=15)
            self.assertFalse(np.array_equal(a, b))
            np.testing.assert_array_equal(a, repeated)
            self.assertEqual(first["initial_qpos"], replay["initial_qpos"])
            self.assertNotEqual(first["initial_qpos"], second["initial_qpos"])
            self.assertEqual(first["initial_qvel"], [0.0] * env.model.nv)
            self.assertTrue(first["initialization_randomization"]["enabled"])
            self.assertFalse(first["initialization_randomization"]["training_domain_randomization"])
            np.testing.assert_array_equal(env.model.body_mass, env._nominal["body_mass"])
            np.testing.assert_array_equal(env.model.geom_friction, env._nominal["geom_friction"])
            env.config = replace(env.config, initial_joint_noise_std=0.)
            nominal_a, _ = env.reset(seed=15)
            nominal_b, info = env.reset(seed=16)
            np.testing.assert_array_equal(nominal_a, nominal_b)
            self.assertFalse(info["initialization_randomization"]["enabled"])
        finally:
            env.close()

    def test_time_limit_is_truncation_and_standing_is_not_walking_success(self):
        env = G1RobustEnv(self.simple_config())
        try:
            env.reset(seed=1)
            for _ in range(10):
                _, reward, terminated, truncated, info = env.step(np.zeros(29))
                self.assertTrue(np.isfinite(reward))
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertFalse(info["is_success"])
            self.assertAlmostEqual(info["elapsed_s"], .2)
            with self.assertRaises(RuntimeError):
                env.step(np.zeros(29))
        finally:
            env.close()

    def test_physics_failure_returns_finite_terminal_observation_and_info(self):
        env = G1RobustEnv(self.simple_config())
        try:
            env.reset(seed=1)
            def invalid_step(targets):
                env.data.qvel[:] = np.nan
                raise RuntimeError("invalid physical state")
            with patch.object(env.physics, "step_joint_targets", side_effect=invalid_step):
                obs, reward, terminated, truncated, info = env.step(np.zeros(29))
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertTrue(np.all(np.isfinite(obs)))
            self.assertTrue(np.isfinite(reward))
            self.assertFalse(info["physical_state_valid"])
            self.assertIsNotNone(info["failure_reason"])
            self.assertTrue(all(np.isfinite(value) for value in info["metrics"].values()))
        finally:
            env.close()

    def test_rgb_render_path_constructs_renderer_and_closes(self):
        env = G1RobustEnv(self.simple_config(), render_mode="rgb_array")
        try:
            env.reset(seed=1)
            with patch("mujoco.Renderer") as renderer:
                renderer.return_value.render.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
                frame = env.render()
                self.assertEqual(frame.shape, (480, 640, 3))
                renderer.assert_called_once()
                env.close()
                renderer.return_value.close.assert_called_once()
        finally:
            env.close()

    def test_config_json_roundtrip_and_evaluation_constraints(self):
        config = RLConfig.preset("train_random")
        self.assertEqual(config.to_dict(), RLConfig.from_dict(config.to_dict()).to_dict())
        with self.assertRaises(ValueError):
            RLConfig(training=False, domain_randomization=True)
        with self.assertRaises(ValueError):
            RLConfig(action_scale=0)


if __name__ == "__main__":
    unittest.main()
