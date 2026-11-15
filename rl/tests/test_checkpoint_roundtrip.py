"""Exercise native SB3 serialization, including PyTorch inside the SB3 ZIP."""

from pathlib import Path
import tempfile
import unittest

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO


class NativeCheckpointTests(unittest.TestCase):
    def test_ppo_roundtrip_preserves_weights_and_prediction(self):
        torch.set_num_threads(1)
        env = gym.make("CartPole-v1")
        try:
            model = PPO("MlpPolicy", env, n_steps=8, batch_size=4,
                        policy_kwargs={"net_arch": [8]}, seed=7, device="cpu")
            observation, _ = env.reset(seed=3)
            expected, _ = model.predict(observation, deterministic=True)
            with tempfile.TemporaryDirectory(prefix="rob6323_native_checkpoint_") as directory:
                path = Path(directory) / "policy.zip"
                model.save(path)
                restored = PPO.load(path, device="cpu")
                actual, _ = restored.predict(observation, deterministic=True)
                np.testing.assert_array_equal(actual, expected)
                for name, parameter in model.policy.state_dict().items():
                    torch.testing.assert_close(parameter, restored.policy.state_dict()[name], rtol=0, atol=0)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
