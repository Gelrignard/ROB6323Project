"""Kinematic walking references for a full-order predictive controller.

These trajectories guide the optimizer; they are never imposed on the plant.
The lateral CoM sway is a bounded LIP reference, and inverse kinematics maps
its planted/swinging feet onto G1 joints. Actual motion must still result from
MuJoCo actuation and contacts. No learned policy or state-reset gait is used.
"""

from dataclasses import dataclass
import math

import mujoco
import numpy as np

from sim_env.g1_model import get_g1_profile, load_g1_model
from sim_env.g1_replay import G1Replay
from sim_env.gait import GaitConfig, WalkingReference


@dataclass(frozen=True)
class ReferenceConfig:
    duration: float = 8.0
    sample_dt: float = .02
    preview_padding: float = 1.5
    speed: float = .18
    step_duration: float = .65
    walk_start: float = .7
    double_support_fraction: float = .3
    swing_height: float = .04
    com_lowering: float = .04
    start_xy: tuple = (1., 5.)

    def __post_init__(self):
        positive = (self.duration, self.sample_dt, self.preview_padding,
                    self.speed, self.step_duration, self.swing_height)
        if not all(np.isfinite(x) and x > 0 for x in positive):
            raise ValueError("reference times, speed and swing height must be positive")
        if not 0 < self.double_support_fraction < 1 or not 0 <= self.com_lowering <= .1:
            raise ValueError("invalid support fraction or CoM lowering")
        if not np.isfinite(self.walk_start) or self.walk_start < 0:
            raise ValueError("walk_start must be nonnegative")
        if np.asarray(self.start_xy).shape != (2,) or not np.all(np.isfinite(self.start_xy)):
            raise ValueError("start_xy must be a finite pair")


class G1WalkingReference:
    """Precompute a finite straight walking reference with a smooth startup.

    Construction performs IK on a separate model. ``sample`` and ``batch``
    only interpolate arrays; neither modifies the physical environment.
    ``initial_qpos`` may initialize the plant once before an experiment.
    ``joint_targets`` follow the model's actual actuator order in radians.
    """

    def __init__(self, config=None, model=None, **kwargs):
        if config is not None and kwargs:
            raise ValueError("provide config or keyword settings, not both")
        self.config = config or ReferenceConfig(**kwargs)
        self.model = load_g1_model() if model is None else model
        self.profile = get_g1_profile(self.model)
        cfg = self.config
        horizon = cfg.duration + cfg.preview_padding
        n_steps = max(2, math.ceil((horizon - cfg.walk_start) / cfg.step_duration))
        total_duration = cfg.walk_start + (n_steps + 1.5) * cfg.step_duration
        self.gait = WalkingReference(GaitConfig(
            start_x=cfg.start_xy[0], start_y=cfg.start_xy[1],
            forward_speed=cfg.speed, walk_start=cfg.walk_start,
            step_duration=cfg.step_duration, n_steps=n_steps,
            double_support_fraction=cfg.double_support_fraction,
            step_width=self.profile.step_width,
            foot_half_length=self.profile.sole_half_length,
            foot_half_width=self.profile.sole_half_width,
            swing_height=cfg.swing_height,
            com_height=self.profile.com_height - cfg.com_lowering,
            duration=total_duration))
        self.times = np.arange(0., horizon + cfg.sample_dt * .5, cfg.sample_dt)
        self.times = np.append(self.times, horizon) if self.times[-1] < horizon else self.times
        ik = G1Replay(com_height=self.gait.config.com_height)
        qpos, feet, com, stance, cop = [], [], [], [], []
        for t in self.times:
            sample = self.gait.sample(float(t))
            ik.set_state(sample["com_ref"], sample["feet"], float(t), sample["cop_ref"])
            qpos.append(ik.data.qpos.copy())
            feet.append(sample["feet"])
            com.append([sample["com_ref"][0], sample["com_ref"][2], self.gait.config.com_height])
            stance.append([side in sample["stance_feet"] for side in (0, 1)])
            cop.append(sample["cop_ref"])
        self.qpos = np.asarray(qpos)
        self.feet = np.asarray(feet)
        self.com = np.asarray(com)
        self.stance = np.asarray(stance, dtype=bool)
        self.cop = np.asarray(cop)
        self.qvel = np.zeros((len(self.times), self.model.nv))
        for i in range(len(self.times)):
            lo, hi = max(0, i - 1), min(len(self.times) - 1, i + 1)
            mujoco.mj_differentiatePos(self.model, self.qvel[i], self.times[hi] - self.times[lo],
                                     self.qpos[lo], self.qpos[hi])
        self.com_velocity = np.gradient(self.com, self.times, axis=0)
        self.joint_ids = self.model.actuator_trnid[:, 0].astype(int)
        self.joint_qpos_indices = self.model.jnt_qposadr[self.joint_ids].astype(int)
        self.joint_dof_indices = self.model.jnt_dofadr[self.joint_ids].astype(int)
        self.joint_targets = self.qpos[:, self.joint_qpos_indices].copy()
        self.initial_qpos = self.qpos[0].copy()
        self.initial_qvel = self.qvel[0].copy()
        self.ik_diagnostics = ik.diagnostics()

    def _interpolate(self, values, times):
        times = np.asarray(times, dtype=float)
        original_shape = times.shape
        flat = values.reshape(len(self.times), -1)
        columns = [np.interp(times.reshape(-1), self.times, flat[:, i]) for i in range(flat.shape[1])]
        return np.asarray(columns).T.reshape((*original_shape, *values.shape[1:]))

    def batch(self, times):
        times = np.asarray(times, dtype=float)
        if not np.all(np.isfinite(times)) or np.any(times < 0) or np.any(times > self.times[-1] + 1e-8):
            raise ValueError("reference sample time lies outside its precomputed horizon")
        lower_indices = np.clip(np.searchsorted(self.times, times, side="right") - 1, 0, len(self.times) - 1)
        return {"qpos": self._interpolate(self.qpos, times),
                "qvel": self._interpolate(self.qvel, times),
                "joint_targets": self._interpolate(self.joint_targets, times),
                "com": self._interpolate(self.com, times),
                "com_velocity": self._interpolate(self.com_velocity, times),
                "feet": self._interpolate(self.feet, times),
                "stance": self.stance[lower_indices].copy(),
                "cop": self._interpolate(self.cop, times)}

    def sample(self, time):
        return self.batch(float(time))

    def metadata(self):
        return {"kind": "kinematic objective and warm-start prior only",
                "source": "bounded LIP CoM + alternating foot references + full G1 IK",
                "plant_state_prescribed": False,
                "sample_dt_s": self.config.sample_dt,
                "speed_mps": self.config.speed,
                "step_duration_s": self.config.step_duration,
                "swing_height_m": self.config.swing_height,
                "com_height_m": self.gait.config.com_height,
                "ik": self.ik_diagnostics}
