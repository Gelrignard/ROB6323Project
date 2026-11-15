"""Serializable experiment settings for the full-body G1 robustness task."""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RLConfig:
    training: bool = False
    episode_duration_s: float = 20.0
    control_dt: float = .02
    physics_dt: float = .002
    action_scale: float = .75
    nominal_joint_offsets: dict = field(default_factory=lambda: {
        "left_hip_pitch_joint": -.15, "left_knee_joint": .3, "left_ankle_pitch_joint": -.15,
        "right_hip_pitch_joint": -.15, "right_knee_joint": .3, "right_ankle_pitch_joint": -.15})
    curriculum_strength: float = 1.0
    command: tuple = (.35, 0.0, 0.0)
    command_ranges: tuple | None = None
    base_xy: tuple = (1.0, 5.0)
    drag_model: str = "surface"
    surface_max_patches: int = 4096
    wind: dict = field(default_factory=lambda: {"kind": "calm"})
    disturbances: dict = field(default_factory=lambda: {"enabled": False})
    domain_randomization: bool = False
    observation_noise: bool = False
    max_action_delay_steps: int = 2
    mass_scale_range: tuple = (.85, 1.15)
    friction_range: tuple = (.4, 1.0)
    pd_gain_scale_range: tuple = (.8, 1.2)
    # Initial-state uncertainty is sampled in both training and evaluation.
    # Set to zero for an exactly nominal initial pose; this is not sensor noise.
    initial_joint_noise_std: float = .005
    linear_velocity_noise_std: float = .02
    angular_velocity_noise_std: float = .02
    gravity_noise_std: float = .005
    joint_position_noise_std: float = .005
    joint_velocity_noise_std: float = .02
    tracking_sigma: float = .5
    yaw_tracking_sigma: float = .5
    success_tracking_rmse_mps: float = .25
    success_yaw_rmse_radps: float = .5
    success_progress_fraction: float = .6
    reward_weights: dict = field(default_factory=lambda: {
        "velocity_tracking": 2.0, "yaw_tracking": .5, "upright": .25,
        "alive": .1, "action_rate": -.02, "actuator_effort": -.01,
        "foot_slip": -.1, "joint_velocity": -.0005, "fall": -5.0})

    def __post_init__(self):
        for name in ("episode_duration_s", "control_dt", "physics_dt", "action_scale",
                     "tracking_sigma", "yaw_tracking_sigma", "success_tracking_rmse_mps", "success_yaw_rmse_radps"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isclose(self.control_dt / self.physics_dt, round(self.control_dt / self.physics_dt)):
            raise ValueError("control_dt must be an integer multiple of physics_dt")
        if self.control_dt < self.physics_dt:
            raise ValueError("control_dt must be >= physics_dt")
        if not np.isfinite(self.curriculum_strength) or not 0 <= self.curriculum_strength <= 1:
            raise ValueError("curriculum_strength must lie in [0,1]")
        if self.drag_model not in ("none", "lumped", "surface"):
            raise ValueError("drag_model must be none, lumped or surface")
        if not isinstance(self.surface_max_patches, int) or self.surface_max_patches < 35:
            raise ValueError("surface_max_patches must be an integer >=35")
        if not isinstance(self.max_action_delay_steps, int) or self.max_action_delay_steps < 0:
            raise ValueError("max_action_delay_steps must be a nonnegative integer")
        if np.asarray(self.command).shape != (3,) or not np.all(np.isfinite(self.command)):
            raise ValueError("command must be finite (vx,vy,yaw_rate)")
        if np.asarray(self.base_xy).shape != (2,) or not np.all(np.isfinite(self.base_xy)) or np.any(np.asarray(self.base_xy) <= .2) or np.any(np.asarray(self.base_xy) >= 9.8):
            raise ValueError("base_xy must start inside the 10m patch")
        for name in ("mass_scale_range", "friction_range", "pd_gain_scale_range"):
            pair = np.asarray(getattr(self, name), dtype=float)
            if pair.shape != (2,) or not np.all(np.isfinite(pair)) or pair[0] <= 0 or pair[1] < pair[0]:
                raise ValueError(f"{name} must be ordered positive bounds")
        if self.command_ranges is not None:
            bounds = np.asarray(self.command_ranges, dtype=float)
            if bounds.shape != (3, 2) or not np.all(np.isfinite(bounds)) or np.any(bounds[:, 1] < bounds[:, 0]):
                raise ValueError("command_ranges must be three ordered finite bounds")
        for name in ("initial_joint_noise_std", "linear_velocity_noise_std", "angular_velocity_noise_std",
                     "gravity_noise_std", "joint_position_noise_std", "joint_velocity_noise_std"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 < self.success_progress_fraction <= 1:
            raise ValueError("success_progress_fraction must lie in (0,1]")
        if not isinstance(self.wind, dict) or not isinstance(self.disturbances, dict):
            raise ValueError("wind and disturbances must be JSON objects")
        if not isinstance(self.nominal_joint_offsets, dict) or not all(np.isfinite(v) for v in self.nominal_joint_offsets.values()):
            raise ValueError("nominal_joint_offsets must map joint names to finite radians")
        expected = {"velocity_tracking", "yaw_tracking", "upright", "alive", "action_rate",
                    "actuator_effort", "foot_slip", "joint_velocity", "fall"}
        if set(self.reward_weights) != expected or not all(np.isfinite(v) for v in self.reward_weights.values()):
            raise ValueError("reward_weights must contain the documented finite reward terms")
        if self.domain_randomization and not self.training:
            raise ValueError("domain randomization is only enabled in training mode")
        if self.observation_noise and not self.training:
            raise ValueError("training sensor noise is disabled in evaluation mode")

    def to_dict(self):
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def from_dict(cls, value):
        return cls(**value)

    @classmethod
    def from_json(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def preset(cls, name):
        if name == "train_random":
            return cls(training=True, drag_model="none", domain_randomization=True,
                       observation_noise=True, disturbances={"enabled": True},
                       command_ranges=((.2, .5), (-.1, .1), (-.15, .15)))
        if name in ("calm", "tornado", "uniform", "gust", "shear", "moving_vortex"):
            return cls(wind={"kind": name}, max_action_delay_steps=0)
        raise ValueError(f"Unknown preset {name!r}")
