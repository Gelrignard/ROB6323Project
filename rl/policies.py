"""Controller adapters sharing the physical G1 evaluator's action interface.

External factory protocol: factory(env, config) -> controller with attributes
name, action_mode ('normalized' or 'joint_targets'), information_access, wind_access, and
act(observation, *, state). Optional reset(seed=..., info=...) and close().
Torque controllers need a separate matched-actuation environment; torque is
never silently interpreted as a position target.
"""

import importlib
import numpy as np

from .contracts import environment_contract, load_bundle


def checked_action(action, env):
    value = np.asarray(action, dtype=np.float32)
    if value.shape != env.action_space.shape or not np.all(np.isfinite(value)):
        raise ValueError(f"Controller action must be finite shape {env.action_space.shape}.")
    if np.any(value < env.action_space.low - 1e-6) or np.any(value > env.action_space.high + 1e-6):
        raise ValueError("Controller action exceeds the declared normalized position-action range.")
    return value


class StandController:
    """Integration diagnostic only; not a walking or MPC baseline."""
    name = "stand_diagnostic"
    information_access = "none; constant home joint targets"
    wind_access = "none"

    def __init__(self, env):
        self.env = env

    def reset(self, **kwargs):
        pass

    def act(self, observation, **kwargs):
        return np.zeros(self.env.action_space.shape, dtype=np.float32)

    def close(self):
        pass


class PPOController:
    information_access = "99-element proprioceptive observation; includes simulated base velocity estimate"
    wind_access = "none"

    def __init__(self, env, checkpoint, device="cpu"):
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        self.env = env
        self.bundle = load_bundle(checkpoint, environment_contract(env))
        self.name = self.bundle.directory.name
        self.model = PPO.load(str(self.bundle.model_path), device=device)
        self.normalizer = None
        if self.bundle.normalization_path is not None:
            # Used only to load compatible spaces. Evaluation drives raw env
            # directly and calls normalize_obs with frozen training statistics.
            self.normalizer = VecNormalize.load(str(self.bundle.normalization_path), DummyVecEnv([lambda: env]))
            self.normalizer.training = False
            self.normalizer.norm_reward = False

    def reset(self, **kwargs):
        pass  # SB3 PPO here is feed-forward, not RecurrentPPO.

    def act(self, observation, **kwargs):
        value = np.asarray(observation, dtype=np.float32)
        if self.normalizer is not None:
            value = self.normalizer.normalize_obs(value.copy())
        action, _ = self.model.predict(value, deterministic=True)
        return checked_action(action, self.env)

    def close(self):
        if self.normalizer is not None:
            self.normalizer.close()


class ExternalController:
    def __init__(self, env, factory_spec, config=None):
        module_name, separator, factory_name = factory_spec.partition(":")
        if not separator or not module_name or not factory_name:
            raise ValueError("External controller must be specified as module:factory.")
        self.env = env
        factory = getattr(importlib.import_module(module_name), factory_name)
        self.controller = factory(env, config or {})
        for attribute in ("name", "action_mode", "information_access", "wind_access", "act"):
            if not hasattr(self.controller, attribute):
                raise ValueError(f"External controller must declare {attribute}.")
        self.name = str(self.controller.name)
        self.information_access = str(self.controller.information_access)
        self.wind_access = str(self.controller.wind_access)
        if self.wind_access not in ("none", "current_measurement", "known_field"):
            raise ValueError("wind_access must be none, current_measurement, or known_field; future random disturbance oracles are excluded.")
        if self.controller.action_mode not in ("normalized", "joint_targets"):
            raise ValueError("Only normalized actions or joint_targets are supported; motor torques are not position targets.")

    def reset(self, **kwargs):
        if hasattr(self.controller, "reset"):
            self.controller.reset(**kwargs)

    def act(self, observation, **kwargs):
        state = self.env.get_state()
        # Full instantaneous simulated state is allowed for a declared MPC
        # baseline. Future disturbance events are deliberately not forwarded.
        action = self.controller.act(np.asarray(observation).copy(), state=state)
        if self.controller.action_mode == "joint_targets":
            action = self.env.action_from_joint_targets(action, clip=False)
        return checked_action(action, self.env)

    def close(self):
        if hasattr(self.controller, "close"):
            self.controller.close()
