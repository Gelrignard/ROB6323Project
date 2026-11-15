"""MuJoCo integration of a constant-height linear inverted pendulum (LIP).

This is a reduced walking-pattern benchmark, not a floating-base humanoid.
The center of mass has two dynamic slide coordinates. Ground reaction is
mg/h * (c - cop), with Fz=mg supplied by the prescribed height constraint.
Feet/support schedules and the rendered legs are kinematic. Contact
feasibility is checked analytically, not solved by MuJoCo foot contacts.
"""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .wind import local_to_world, tornado_velocity, wind_force
from .config import DEFAULT_TORNADO
from .gait import point_in_support


@dataclass(frozen=True)
class SimulationConfig:
    mass: float = 60.0
    height: float = 0.9
    gravity: float = 9.81
    physics_dt: float = 0.002
    control_dt: float = 0.02
    friction: float = 0.7
    wind_scale: float = 1.0
    patch_size: float = 10.0

    def __post_init__(self):
        for name in ("mass", "height", "gravity", "physics_dt", "control_dt", "friction", "patch_size"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not np.isfinite(self.wind_scale) or self.wind_scale < 0:
            raise ValueError("wind_scale must be nonnegative and finite")
        if self.patch_size != 10.0:
            raise ValueError("the current MJCF scene is a fixed 10 m test patch")
        ratio = self.control_dt / self.physics_dt
        if not np.isclose(ratio, round(ratio), atol=1e-10) or ratio < 1:
            raise ValueError("control_dt must be an integer multiple of physics_dt")


class ReducedWalkingEnv:
    """State [x,vx,y,vy] in patch-local metres/seconds; action is CoP [x,y]."""

    def __init__(self, config=None, wind_config=None):
        self.config = config or SimulationConfig()
        self.wind_config = DEFAULT_TORNADO if wind_config is None else wind_config
        if self.wind_config.patch_size_m != self.config.patch_size:
            raise ValueError("wind map and MuJoCo scene must have the same patch size")
        self.model = mujoco.MjModel.from_xml_path(str(Path(__file__).parent / "models" / "lip_walker.xml"))
        self.model.opt.timestep = self.config.physics_dt
        self.model.opt.gravity[:] = [0, 0, -self.config.gravity]
        self.com_id = self.model.body("com").id
        self.model.body_mass[self.com_id] = self.config.mass
        self.model.body_pos[self.com_id, 2] = self.config.height
        self.data = mujoco.MjData(self.model)
        mujoco.mj_setConst(self.model, self.data)
        self.substeps = int(round(self.config.control_dt / self.config.physics_dt))
        self._mocap = {name: int(self.model.body_mocapid[self.model.body(name).id])
                       for name in ("left_foot", "right_foot", "left_thigh", "left_shin",
                                    "right_thigh", "right_shin", "cop_marker", "reference_marker")}
        self.last_wind_force = np.zeros(2)
        self.last_wind_velocity = np.zeros(2)
        self.last_ground_force = np.zeros(2)
        self.max_friction_ratio = 0.0
        self.ground_force_squared_integral = 0.0
        self.max_wind_speed = 0.0
        self.max_wind_force = 0.0

    @property
    def state(self):
        return np.array([self.data.qpos[0], self.data.qvel[0], self.data.qpos[1], self.data.qvel[1]])

    def reset(self, state):
        state = np.asarray(state, dtype=float)
        if state.shape != (4,) or not np.all(np.isfinite(state)):
            raise ValueError("state must be a finite vector [x,vx,y,vy]")
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = state[[0, 2]]
        self.data.qvel[:] = state[[1, 3]]
        self.max_friction_ratio = 0.0
        self.ground_force_squared_integral = 0.0
        self.max_wind_speed = 0.0
        self.max_wind_force = 0.0
        self.last_wind_force = np.zeros(2)
        self.last_wind_velocity = np.zeros(2)
        self.last_ground_force = np.zeros(2)
        mujoco.mj_forward(self.model, self.data)
        return self.state

    def step(self, cop, support_polygon=None):
        cop = np.asarray(cop, dtype=float)
        if cop.shape != (2,) or not np.all(np.isfinite(cop)):
            raise ValueError("CoP must be finite with shape (2,)")
        if support_polygon is not None and not point_in_support(cop, support_polygon):
            raise ValueError("CoP is outside the supplied support region")
        start_time = float(self.data.time)
        for _ in range(self.substeps):
            position = self.data.qpos.copy()
            velocity = self.data.qvel.copy()
            world = local_to_world(position, self.wind_config)
            self.last_wind_velocity = tornado_velocity(world, self.wind_config, scale=self.config.wind_scale)
            self.last_wind_force = wind_force(world, velocity, self.wind_config, scale=self.config.wind_scale)
            self.last_ground_force = self.config.mass * self.config.gravity / self.config.height * (position - cop)
            self.ground_force_squared_integral += float(np.dot(self.last_ground_force, self.last_ground_force) * self.config.physics_dt)
            self.max_wind_speed = max(self.max_wind_speed, float(np.linalg.norm(self.last_wind_velocity)))
            self.max_wind_force = max(self.max_wind_force, float(np.linalg.norm(self.last_wind_force)))
            ratio = np.linalg.norm(self.last_ground_force) / (self.config.friction * self.config.mass * self.config.gravity)
            self.max_friction_ratio = max(self.max_friction_ratio, float(ratio))
            self.data.qfrc_applied[:] = self.last_ground_force + self.last_wind_force
            mujoco.mj_step(self.model, self.data)
        expected_time = start_time + self.config.control_dt
        if not np.all(np.isfinite(self.state)) or not np.isclose(self.data.time, expected_time, atol=1e-9, rtol=0):
            raise RuntimeError("Invalid MuJoCo state or unexpected simulation reset")
        if np.any(self.data.warning.number):
            raise RuntimeError(f"MuJoCo warning counts: {self.data.warning.number.tolist()}")
        return self.state

    def failure_reason(self, reference=None):
        c = self.state[[0, 2]]
        if np.any(c < 0) or np.any(c > self.config.patch_size):
            return "center_of_mass_left_test_region"
        if self.max_friction_ratio > 1.0 + 1e-6:
            return "required_ground_force_exceeded_friction_limit"
        if reference is not None:
            stance = np.asarray(reference["stance_feet"], dtype=int)
            feet = np.asarray(reference["feet"])[stance, :2]
            # A geometric reach proxy; this is NOT a full-body fall detector.
            if np.min(np.linalg.norm(feet - c, axis=1)) > 0.65:
                return "center_of_mass_exceeded_reach_proxy"
        return None

    def _segment(self, name, a, b):
        delta = b - a
        length = float(np.linalg.norm(delta))
        unit = delta / max(length, 1e-12)
        # Rotation taking capsule's local z axis onto the segment.
        quat = np.array([1.0 + unit[2], -unit[1], unit[0], 0.0])
        if np.linalg.norm(quat) < 1e-9:
            quat = np.array([0.0, 1.0, 0.0, 0.0])
        quat /= np.linalg.norm(quat)
        self.data.mocap_pos[self._mocap[name]] = (a + b) / 2
        self.data.mocap_quat[self._mocap[name]] = quat
        self.model.geom_size[self.model.geom(name + "_geom").id, 1] = length / 2

    def update_visuals(self, reference, cop):
        """Visual-only articulated shape; no feedback forces come from it."""
        c = self.state[[0, 2]]
        for side, index, lateral in (("left", 0, 0.09), ("right", 1, -0.09)):
            foot = np.asarray(reference["feet"][index], dtype=float).copy()
            foot[2] += 0.025
            self.data.mocap_pos[self._mocap[side + "_foot"]] = foot
            hip = np.array([c[0], c[1] + lateral, self.config.height - 0.10])
            knee = (hip + foot) / 2 + np.array([0.13, 0.0, 0.0])
            self._segment(side + "_thigh", hip, knee)
            self._segment(side + "_shin", knee, foot)
        self.data.mocap_pos[self._mocap["cop_marker"]] = [*cop, 0.035]
        self.data.mocap_pos[self._mocap["reference_marker"]] = [*reference["com_ref"][[0, 2]], self.config.height]
        mujoco.mj_forward(self.model, self.data)
