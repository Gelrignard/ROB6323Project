"""Gymnasium full-body G1 task for PPO robustness experiments.

All motion comes from MuJoCo contacts/actuators. Actor observations never
contain wind, disturbance timing, force labels, or future events. Base linear
velocity is simulator state and would require an estimator on real hardware.
The 99/29 interface is project-specific, not automatically compatible with
arbitrary pretrained G1 policies.
"""

from collections import deque
from dataclasses import replace

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from .g1_physics import G1PhysicsEnv
from .rl_config import RLConfig
from .disturbances import DisturbanceConfig, DisturbanceScheduler
from .wind_fields import WindFieldConfig, make_wind_field


class G1RobustEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    schema_version = "g1-proprioception-99-position-offset-29-v1"

    def __init__(self, config=None, render_mode=None):
        super().__init__()
        self.config = RLConfig() if config is None else RLConfig.from_dict(config) if isinstance(config, dict) else config
        if not isinstance(self.config, RLConfig):
            raise TypeError("config must be RLConfig or its serialized dictionary")
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError("render_mode must be human, rgb_array or None")
        self.render_mode = render_mode
        cfg = self.config
        self.physics = G1PhysicsEnv(control_dt=cfg.control_dt, physics_dt=cfg.physics_dt,
                                    wind_scale=1.0, drag_model=cfg.drag_model,
                                    surface_max_patches=cfg.surface_max_patches)
        self.model, self.data = self.physics.model, self.physics.data
        self.root_body_id = self.physics.profile.root_body_id
        self.joint_ids = self.model.actuator_trnid[:, 0].astype(int)
        self.qpos_indices = self.model.jnt_qposadr[self.joint_ids].astype(int)
        self.dof_indices = self.model.jnt_dofadr[self.joint_ids].astype(int)
        self.nominal_targets = self.physics.profile.home_ctrl.copy()
        self.joint_names = [self.model.joint(int(i)).name for i in self.joint_ids]
        for name, offset in cfg.nominal_joint_offsets.items():
            if name not in self.joint_names:
                raise ValueError(f"unknown actuated joint in nominal_joint_offsets: {name}")
            self.nominal_targets[self.joint_names.index(name)] += offset
        if np.any(self.nominal_targets < self.model.actuator_ctrlrange[:, 0]) or np.any(self.nominal_targets > self.model.actuator_ctrlrange[:, 1]):
            raise ValueError("nominal joint targets must respect actuator angle limits")
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.model.nu,), dtype=np.float32)
        self.observation_space = spaces.Box(-1e6, 1e6, shape=(99,), dtype=np.float32)
        self._nominal = {name: getattr(self.model, name).copy() for name in
                         ("body_mass", "body_inertia", "geom_friction", "actuator_gainprm", "actuator_biasprm")}
        self._robot_bodies = np.flatnonzero(self.model.body_rootid == self.root_body_id)
        self._foot_bodies = [self.model.body(f"{side}_ankle_roll_link").id for side in ("left", "right")]
        self._renderer = None
        self._viewer = None
        self._camera = mujoco.MjvCamera()
        self._camera.azimuth, self._camera.elevation, self._camera.distance = 135, -20, 3.0
        self._needs_reset = True
        self.command = np.asarray(cfg.command, dtype=float)
        self.previous_action = np.zeros(self.model.nu)
        self.domain_parameters = {}
        self.scheduler = None
        self._pending_curriculum_strength = None

    def schema(self):
        """Stable policy/checkpoint interface, independent of disturbance preset."""
        return {"version": self.schema_version, "robot": "Unitree G1 Menagerie 29-DoF",
                "joint_names": self.joint_names,
                "observation": {"shape": [99], "dtype": "float32", "fields": [
                    {"name": "base_linear_velocity_body_mps", "slice": [0, 3]},
                    {"name": "base_angular_velocity_body_radps", "slice": [3, 6]},
                    {"name": "projected_gravity_unit", "slice": [6, 9]},
                    {"name": "command_body_vx_vy_yawrate", "slice": [9, 12]},
                    {"name": "joint_position_minus_nominal_rad", "slice": [12, 41]},
                    {"name": "joint_velocity_radps", "slice": [41, 70]},
                    {"name": "previous_requested_normalized_action", "slice": [70, 99]}],
                    "scaling": "physical units; no hidden per-field scaling",
                    "base_linear_velocity_source": "simulator state; real deployment needs estimator"},
                "action": {"shape": [29], "dtype": "float32", "bounds": [-1.0, 1.0],
                           "meaning": "nominal_joint_target + action_scale * normalized_action; actuator angle range clipping",
                           "action_scale_rad": self.config.action_scale,
                           "nominal_targets_rad": self.nominal_targets.tolist(),
                           "target_limits_rad": self.model.actuator_ctrlrange.tolist()},
                "control_dt_s": self.config.control_dt, "physics_dt_s": self.config.physics_dt}

    def action_from_joint_targets(self, targets, clip=False):
        """Convert angles, rejecting any target the actuator or policy would alter."""
        targets = np.asarray(targets, dtype=float)
        if targets.shape != (self.model.nu,) or not np.all(np.isfinite(targets)):
            raise ValueError("joint targets must be finite shape (29,)")
        limits = self.model.actuator_ctrlrange
        if not clip and (np.any(targets < limits[:, 0]) or np.any(targets > limits[:, 1])):
            raise ValueError("joint targets lie outside physical actuator angle limits")
        if clip:
            targets = np.clip(targets, limits[:, 0], limits[:, 1])
        action = (targets - self.nominal_targets) / self.config.action_scale
        if not clip and np.any(np.abs(action) > 1 + 1e-8):
            raise ValueError("joint targets lie outside this policy's action range")
        return np.clip(action, -1, 1).astype(np.float32)

    def get_state(self):
        return {"qpos": self.data.qpos.copy(), "qvel": self.data.qvel.copy(), "time": float(self.data.time)}

    def set_curriculum_strength(self, strength):
        """Set next episode's randomization strength; never change mid-episode."""
        validated = replace(self.config, curriculum_strength=float(strength))
        self._pending_curriculum_strength = validated.curriculum_strength

    def _randomize_domain(self, rng):
        for name, value in self._nominal.items():
            getattr(self.model, name)[:] = value
        parameters = {"mass_scale": 1.0, "friction": .6, "pd_gain_scale": 1.0, "action_delay_steps": 0}
        if self.config.training and self.config.domain_randomization:
            strength = self.config.curriculum_strength
            parameters["mass_scale"] = 1 + strength * (rng.uniform(*self.config.mass_scale_range) - 1)
            sampled_friction = rng.uniform(*self.config.friction_range)
            parameters["friction"] = .6 + strength * (sampled_friction - .6)
            parameters["pd_gain_scale"] = 1 + strength * (rng.uniform(*self.config.pd_gain_scale_range) - 1)
            parameters["action_delay_steps"] = int(rng.integers(0, int(np.ceil(strength * self.config.max_action_delay_steps)) + 1))
            self.model.body_mass[self._robot_bodies] *= parameters["mass_scale"]
            self.model.body_inertia[self._robot_bodies] *= parameters["mass_scale"]
            # Include priority foot geoms, whose friction overrides the floor.
            self.model.geom_friction[:, 0] += strength * (sampled_friction - self.model.geom_friction[:, 0])
            self.model.actuator_gainprm[:] *= parameters["pd_gain_scale"]
            self.model.actuator_biasprm[:] *= parameters["pd_gain_scale"]
        mujoco.mj_setConst(self.model, self.data)
        parameters["sliding_friction_by_geom"] = self.model.geom_friction[:, 0].tolist()
        self.domain_parameters = parameters

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self._pending_curriculum_strength is not None:
            self.config = replace(self.config, curriculum_strength=self._pending_curriculum_strength)
            self._pending_curriculum_strength = None
        options = {} if options is None else dict(options)
        unsupported = set(options) - {"command", "disturbance_events"}
        if unsupported:
            raise ValueError(f"unsupported reset options: {sorted(unsupported)}")
        episode_seed = int(self.np_random.integers(0, 2**32, dtype=np.uint32))
        self.episode_seed = episode_seed
        streams = np.random.SeedSequence(episode_seed).spawn(5)
        domain_rng = np.random.default_rng(streams[0])
        self._sensor_rng = np.random.default_rng(streams[1])
        disturbance_seed = int(streams[2].generate_state(1)[0])
        command_rng = np.random.default_rng(streams[3])
        initialization_rng = np.random.default_rng(streams[4])
        self._randomize_domain(domain_rng)
        self.command = np.asarray(options.get("command", self.config.command), dtype=float).copy()
        if "command" not in options and self.config.training and self.config.command_ranges is not None:
            bounds = np.asarray(self.config.command_ranges)
            self.command = command_rng.uniform(bounds[:, 0], bounds[:, 1])
        if self.command.shape != (3,) or not np.all(np.isfinite(self.command)):
            raise ValueError("command must be finite (vx,vy,yaw_rate)")
        pose = self.physics.profile.home_qpos.copy()
        pose[self.qpos_indices] = self.nominal_targets
        initialization_std = self.config.initial_joint_noise_std * (self.config.curriculum_strength if self.config.training else 1.0)
        pose[self.qpos_indices] += initialization_rng.normal(0, initialization_std, self.model.nu)
        pose[self.qpos_indices] = np.clip(pose[self.qpos_indices], self.model.jnt_range[self.joint_ids, 0], self.model.jnt_range[self.joint_ids, 1])
        self.physics.reset(base_xy=self.config.base_xy, qpos=pose)
        self.data.qpos[2] -= min(self.data.site_xpos[list(self.physics.profile.sole_site_ids), 2])
        self.data.ctrl[:] = self.nominal_targets
        mujoco.mj_forward(self.model, self.data)
        self.physics.wind_field = make_wind_field(WindFieldConfig(**self.config.wind))
        self.scheduler = DisturbanceScheduler(self.model, DisturbanceConfig(**self.config.disturbances))
        self.scheduler.reset(seed=disturbance_seed, horizon_s=self.config.episode_duration_s,
                             strength=self.config.curriculum_strength, events=options.get("disturbance_events"))
        self.physics.disturbance_callback = self.scheduler.sample
        self.previous_action = np.zeros(self.model.nu)
        self._action_queue = deque([np.zeros(self.model.nu) for _ in range(self.domain_parameters["action_delay_steps"])])
        self._steps = 0
        self._tracking_square_integral = self._yaw_square_integral = 0.0
        self._travel_distance = self._commanded_distance = self._commanded_progress = 0.0
        self._start_position = self.data.qpos[:2].copy()
        self._last_position = self._start_position.copy()
        self._needs_reset = False
        self._last_observation = self._observation()
        info = self._info(None, False)
        info.update({"episode_seed": episode_seed, "disturbance_seed": disturbance_seed,
                     "domain_parameters": self.domain_parameters.copy(), "config": self.config.to_dict(),
                     "initial_qpos": self.data.qpos.tolist(), "initial_qvel": self.data.qvel.tolist(),
                     "initialization_randomization": {"enabled": initialization_std > 0,
                         "distribution": "independent Gaussian joint angles, clipped to joint limits; sole height grounded",
                         "joint_position_std_rad": initialization_std,
                         "joint_velocity_std_radps": 0.0,
                         "training_domain_randomization": self.config.training and self.config.domain_randomization},
                     "disturbance_events": self.scheduler.events,
                     "wind_field": self.physics.wind_field.metadata()})
        if self.render_mode == "human":
            self.render()
        return self._last_observation.copy(), info

    def _body_state(self):
        rotation = self.data.xmat[self.root_body_id].reshape(3, 3)
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jacBodyCom(self.model, self.data, jp, jr, self.root_body_id)
        return rotation.T @ (jp @ self.data.qvel), rotation.T @ (jr @ self.data.qvel), rotation.T @ [0., 0., -1.]

    def _observation(self):
        linear, angular, gravity = self._body_state()
        obs = np.concatenate((linear, angular, gravity, self.command,
                              self.data.qpos[self.qpos_indices] - self.nominal_targets,
                              self.data.qvel[self.dof_indices], self.previous_action))
        if self.config.training and self.config.observation_noise:
            noise = np.zeros(99)
            for begin, end, sigma in ((0, 3, self.config.linear_velocity_noise_std),
                                       (3, 6, self.config.angular_velocity_noise_std),
                                       (6, 9, self.config.gravity_noise_std),
                                       (12, 41, self.config.joint_position_noise_std),
                                       (41, 70, self.config.joint_velocity_noise_std)):
                noise[begin:end] = self._sensor_rng.normal(0, sigma * self.config.curriculum_strength, end - begin)
            obs += noise
        if not np.all(np.isfinite(obs)):
            raise RuntimeError("nonfinite actor observation")
        return np.clip(obs, self.observation_space.low, self.observation_space.high).astype(np.float32)

    def _foot_contacts_and_slip(self):
        contacts = np.zeros(2, dtype=bool)
        for contact in self.data.contact[:self.data.ncon]:
            bodies = (int(self.model.geom_bodyid[contact.geom1]), int(self.model.geom_bodyid[contact.geom2]))
            for index, foot in enumerate(self._foot_bodies):
                if foot in bodies and 0 in bodies:
                    contacts[index] = True
        slip = 0.0
        for foot, site in enumerate(self.physics.profile.sole_site_ids):
            if contacts[foot]:
                jp = np.zeros((3, self.model.nv))
                mujoco.mj_jacSite(self.model, self.data, jp, None, site)
                velocity = jp @ self.data.qvel
                slip += float(velocity[:2] @ velocity[:2])
        return contacts, slip

    def _info(self, failure, completed):
        linear, angular, _ = self._body_state()
        state_valid = bool(np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qvel)))
        linear = np.nan_to_num(linear, nan=0., posinf=0., neginf=0.)
        angular = np.nan_to_num(angular, nan=0., posinf=0., neginf=0.)
        position = np.nan_to_num(self.data.qpos[:3], nan=0., posinf=0., neginf=0.)
        elapsed = float(self.data.time) if np.isfinite(self.data.time) else self._steps * self.config.control_dt
        rmse = float(np.sqrt(self._tracking_square_integral / max(elapsed, 1e-12)))
        yaw_rmse = float(np.sqrt(self._yaw_square_integral / max(elapsed, 1e-12)))
        success = bool(completed and failure is None and self._commanded_distance >= .1
                       and self._commanded_progress >= self.config.success_progress_fraction * self._commanded_distance
                       and rmse < self.config.success_tracking_rmse_mps
                       and yaw_rmse < self.config.success_yaw_rmse_radps)
        contacts, _ = self._foot_contacts_and_slip()
        return {"elapsed_s": elapsed, "base_position_m": position,
                "body_linear_velocity_mps": linear.copy(), "body_angular_velocity_radps": angular.copy(),
                "command": self.command.copy(), "foot_contacts": contacts,
                "velocity_tracking_error_mps": float(np.linalg.norm(linear[:2] - self.command[:2])),
                "yaw_tracking_error_radps": float(abs(angular[2] - self.command[2])),
                "failure_reason": failure, "is_success": success, "physical_state_valid": state_valid,
                "metrics": {"distance_m": float(np.linalg.norm(position[:2] - self._start_position)),
                            "path_length_m": self._travel_distance, "commanded_progress_m": self._commanded_progress,
                            "commanded_distance_m": self._commanded_distance,
                            "tracking_rmse_mps": rmse, "yaw_rmse_radps": yaw_rmse,
                            "survival_s": elapsed, "is_success": success}}

    def step(self, action):
        if self._needs_reset:
            raise RuntimeError("reset must be called before step and after episode completion")
        action = np.asarray(action, dtype=float)
        if action.shape != (self.model.nu,) or not np.all(np.isfinite(action)):
            raise ValueError("action must be a finite vector of 29 normalized joint targets")
        requested = np.clip(action, -1.0, 1.0)
        self._action_queue.append(requested.copy())
        applied = self._action_queue.popleft()
        targets = self.nominal_targets + self.config.action_scale * applied
        clipped_targets = np.clip(targets, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
        last_action = self.previous_action.copy()
        self.previous_action = requested.copy()
        failure = None
        try:
            self.physics.step_joint_targets(clipped_targets)
            failure = self.physics.failure_reason()
            observation = self._observation()
        except RuntimeError as error:
            failure = f"physics_error: {error}"
            observation = self._last_observation.copy()
        self._steps += 1
        linear, angular, gravity = self._body_state()
        finite = np.all(np.isfinite(np.concatenate((linear, angular, gravity))))
        if not finite:
            failure = "nonfinite_physical_state"
            linear, angular, gravity = np.zeros(3), np.zeros(3), np.array([0., 0., 1.])
        tracking_sq = float(np.sum((linear[:2] - self.command[:2])**2))
        yaw_sq = float((angular[2] - self.command[2])**2)
        dt = self.config.control_dt
        self._tracking_square_integral += tracking_sq * dt
        self._yaw_square_integral += yaw_sq * dt
        planar_speed = float(np.linalg.norm(self.command[:2]))
        self._commanded_distance += planar_speed * dt
        if planar_speed > 1e-9:
            self._commanded_progress += float(linear[:2] @ (self.command[:2] / planar_speed)) * dt
        position = self.data.qpos[:2]
        if np.all(np.isfinite(position)):
            self._travel_distance += float(np.linalg.norm(position - self._last_position))
            self._last_position = position.copy()
        _, slip = self._foot_contacts_and_slip()
        actuator_scales = np.maximum(np.max(np.abs(self.model.actuator_forcerange), axis=1), 1.0)
        effort = float(np.mean((self.data.actuator_force / actuator_scales)**2))
        terms = {"velocity_tracking": float(np.exp(-tracking_sq / self.config.tracking_sigma**2)),
                 "yaw_tracking": float(np.exp(-yaw_sq / self.config.yaw_tracking_sigma**2)),
                 "upright": float(max(0., -gravity[2])), "alive": 1.0,
                 "action_rate": float(np.mean((requested - last_action)**2)),
                 "actuator_effort": effort, "foot_slip": slip,
                 "joint_velocity": float(np.mean(self.data.qvel[self.dof_indices]**2))}
        if not all(np.isfinite(value) for value in terms.values()):
            terms = {name: 0.0 for name in terms}
            failure = "nonfinite_reward_state"
        reward = dt * sum(self.config.reward_weights[name] * value for name, value in terms.items())
        terminated = failure is not None
        if terminated:
            reward += self.config.reward_weights["fall"]
        truncated = not terminated and self._steps * dt >= self.config.episode_duration_s - 1e-10
        self._needs_reset = terminated or truncated
        self._last_observation = observation.copy()
        info = self._info(failure, truncated)
        info["reward_terms"] = terms
        info["action_clipped_fraction"] = float(np.mean(np.abs(action) > 1.0))
        info["target_clipped_fraction"] = float(np.mean(targets != clipped_targets))
        if self.render_mode == "human":
            self.render()
        return observation, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        if self.render_mode is None:
            return None
        self._camera.lookat[:] = [*self.data.qpos[:2], .7]
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data, camera=self._camera)
            return self._renderer.render().copy()
        if self._viewer is None:
            from mujoco import viewer as mj_viewer
            self._viewer = mj_viewer.launch_passive(self.model, self.data)
            self._viewer.cam.azimuth, self._viewer.cam.elevation, self._viewer.cam.distance = 135, -20, 3.0
        if self._viewer.is_running():
            self._viewer.cam.lookat[:] = self._camera.lookat
            self._viewer.sync()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
