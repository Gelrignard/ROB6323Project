"""Avoid falsely reporting checkpoint compatibility or walking success."""

import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from rl.contracts import check_contract, load_bundle, sha256
from rl.evaluate import SCENARIOS, controller_reset_info, scenario_config, summarize_episode
from rl.policies import ExternalController
from sim_env.rl_config import RLConfig


class ProtocolTests(unittest.TestCase):
    def test_reset_info_excludes_future_events_and_wind(self):
        source = {"command": np.array([.35, 0, 0]), "elapsed_s": 0.0,
                  "disturbance_events": [{"start_s": 1}], "wind_field": {"kind": "tornado"},
                  "config": {"privileged": True}, "domain_parameters": {"mass": 1.1}}
        public = controller_reset_info(source)
        self.assertEqual(set(public), {"command", "elapsed_s"})
        public["command"][0] = 9
        self.assertEqual(source["command"][0], .35)

    def test_scenarios_preserve_common_custom_policy_semantics(self):
        base = RLConfig(training=True, action_scale=.4, control_dt=.01, drag_model="none",
                        nominal_joint_offsets={"left_knee_joint": .1},
                        wind={"kind": "uniform"}, domain_randomization=True, observation_noise=True,
                        curriculum_strength=.1)
        for scenario in SCENARIOS:
            config = scenario_config(scenario, 1, base)
            self.assertEqual(config.action_scale, .4)
            self.assertEqual(config.control_dt, .01)
            self.assertEqual(config.nominal_joint_offsets, base.nominal_joint_offsets)
            self.assertFalse(config.domain_randomization)
            self.assertFalse(config.observation_noise)
            self.assertFalse(config.training)
            self.assertEqual(config.curriculum_strength, 1.0)
        self.assertEqual(scenario_config("reverse_vortex", 1, base).wind["kind"], "tornado")
        self.assertLess(scenario_config("reverse_vortex", 1, base).wind["tangential_speed_mps"], 0)
        self.assertTrue(scenario_config("random_push", 1, base).disturbances["enabled"])

    def test_external_controller_rejects_torque_action_mode(self):
        controller = SimpleNamespace(name="MPC", action_mode="torques", information_access="state",
                                     wind_access="none", act=lambda observation, state: np.zeros(29))
        module = SimpleNamespace(factory=lambda env, config: controller)
        with patch("rl.policies.importlib.import_module", return_value=module):
            with self.assertRaisesRegex(ValueError, "motor torques are not position targets"):
                ExternalController(None, "fake:factory")

    def test_same_shape_but_different_joint_order_is_rejected(self):
        a = {"environment_schema": {"action_shape": [2], "joints": ["left", "right"]}}
        b = {"environment_schema": {"action_shape": [2], "joints": ["right", "left"]}}
        check_contract(a, a)
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            check_contract(a, b)

    def test_checkpoint_requires_matching_normalization_artifact(self):
        with tempfile.TemporaryDirectory(prefix="rob6323_bundle_") as directory:
            root = Path(directory)
            (root / "model.zip").write_bytes(b"placeholder model for artifact validation")
            manifest = {"algorithm": "stable_baselines3.PPO", "contract": {"version": 1},
                        "model_sha256": sha256(root / "model.zip"), "normalize_observations": True,
                        "normalization_sha256": "missing"}
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "normalization"):
                load_bundle(root, {"version": 1})
            (root / "vecnormalize.pkl").write_bytes(b"stats")
            manifest["normalization_sha256"] = sha256(root / "vecnormalize.pkl")
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertIsNotNone(load_bundle(root, {"version": 1}).normalization_path)
            with self.assertRaises(FileNotFoundError):
                load_bundle(root / "misspelled_checkpoint.zip", {"version": 1})
            (root / "model.zip").write_bytes(b"different checkpoint")
            with self.assertRaisesRegex(ValueError, "manifest hash"):
                load_bundle(root, {"version": 1})

    def test_survival_and_reward_do_not_count_as_walking(self):
        info = {"base_position_m": [1, 5, .8], "command": [.35, 0, 0],
                "elapsed_s": 20, "failure_reason": None}
        result = summarize_episode([1, 5, .8], info, [[-.35, 0]], [0], [1e6], [.001], False, True, 20)
        self.assertTrue(result["completed_horizon"])
        self.assertFalse(result["walking_success"])
        moving = {**info, "base_position_m": [8, 5, .8]}
        result = summarize_episode([1, 5, .8], moving, [[0, 0]], [0], [0], [.001], False, True, 20)
        self.assertTrue(result["walking_success"])
        fallen = {**moving, "failure_reason": "pelvis_height_or_tilt_limit"}
        result = summarize_episode([1, 5, .8], fallen, [[0, 0]], [0], [1e6], [.001], True, False, 20)
        self.assertFalse(result["walking_success"])
        self.assertTrue(result["fell"])


if __name__ == "__main__":
    unittest.main()
