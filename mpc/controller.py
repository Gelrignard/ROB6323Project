"""G1 port of DIAL-MPC using native, threaded full-order MuJoCo rollouts.

This is an offline CPU port, with G1-specific motion objectives and a
kinematic warm-start prior. Only the real plant's actuators move the robot.
External loads are unknown: runtime feedback is limited to time/pose/velocity.
"""

import time
import os
import numpy as np
import mujoco
import mujoco.rollout

from sim_env.g1_model import load_g1_model
from .reference import G1WalkingReference
from .dial import DialConfig, DIALOptimizer
from .observation import RobotObservation


class DIALController:
    def __init__(self, env, config=None):
        self.config = dict(config or {})
        c = self.config
        self.wind_prediction = c.setdefault("wind_prediction", "none")
        if self.wind_prediction != "none":
            raise ValueError("DIAL-MPC tests unknown external loads; wind_prediction must be 'none'. "
                             "Privileged current_wrench input is no longer supported.")
        self.control_dt, self.physics_dt, self.substeps = env.control_dt, env.physics_dt, env.substeps
        self.command_speed = float(c.get("speed", .18))
        self.reference = G1WalkingReference(duration=float(c.get("duration", 6.)),
            speed=self.command_speed, sample_dt=env.control_dt,
            preview_padding=max(1.5, env.control_dt * int(c.get("horizon_steps", 20)) + .1),
            step_duration=float(c.get("step_duration", .65)),
            walk_start=float(c.get("walk_start", .7)),
            swing_height=float(c.get("swing_height", .04)),
            com_lowering=float(c.get("com_lowering", .04)),
            double_support_fraction=float(c.get("double_support_fraction", .3)))
        self.initial_qpos = self.reference.initial_qpos.copy()
        self.model = load_g1_model(mpc_sensors=True)
        if self.model.na or self.model.npluginstate:
            raise ValueError("The state-only G1 controller requires no actuator activation or plugin state")
        self.model.opt.timestep = env.physics_dt
        # Explicitly reproduce the plant's actuator and dynamics parameters.
        for name in ("body_mass", "body_inertia", "geom_friction", "dof_damping",
                     "actuator_gainprm", "actuator_biasprm", "actuator_ctrlrange", "actuator_forcerange"):
            getattr(self.model, name)[:] = getattr(env.model, name)
        self.nthread = max(1, min(int(c.get("threads", 8)), os.cpu_count() or 1))
        self.data = [mujoco.MjData(self.model) for _ in range(self.nthread)]
        mujoco.mj_setConst(self.model, self.data[0])
        self._observation_data = mujoco.MjData(self.model)
        self.rollout = mujoco.rollout.Rollout(nthread=self.nthread)
        self.active = np.asarray(c.get("active_actuators", list(range(12))), dtype=int)
        if len(set(self.active)) != len(self.active) or np.any(self.active < 0) or np.any(self.active >= self.model.nu):
            raise ValueError("active_actuators must be distinct valid actuator indices")
        self.action_scale = float(c.get("action_scale", .2))
        self.dial_config = DialConfig(samples=int(c.get("samples", 64)),
            horizon_steps=int(c.get("horizon_steps", 20)),
            knot_intervals=int(c.get("knot_intervals", 5)),
            iterations=int(c.get("iterations", 3)), init_iterations=int(c.get("init_iterations", 6)),
            temperature=float(c.get("temperature", .05)), sigma=float(c.get("sigma", .5)),
            horizon_decay=float(c.get("horizon_decay", .9)),
            trajectory_decay=float(c.get("trajectory_decay", .4)),
            freeze_first=bool(c.get("freeze_first", False)))
        self.sensor_slices = {}
        for name in ("left_foot_pos", "right_foot_pos", "left_foot_vel", "right_foot_vel", "com", "com_vel", "torso_quat"):
            sensor = self.model.sensor("mpc_" + name)
            self.sensor_slices[name] = slice(int(sensor.adr[0]), int(sensor.adr[0] + sensor.dim[0]))
        self.weights = {"com_xy": 100., "com_z": 600., "velocity": 8.,
                        "orientation": 500., "foot_xy": 100., "foot_z": 1500.,
                        "joint": 1., "residual": .1, "foot_slip": .25,
                        **c.get("weights", {})}
        self.last_plan_info = {}
        self.reset(seed=0)

    def reset(self, seed=0):
        self.optimizer = DIALOptimizer(len(self.active), self.dial_config, seed=seed)
        self.nodes = np.zeros((self.dial_config.knot_intervals + 1, len(self.active)))
        self._first = True
        self.last_plan_info = {}

    def _rollout_cost(self, candidates):
        n, length, _ = candidates.shape
        horizon = self.dial_config.horizon_steps
        base = self._reference_controls[None, :horizon, :]
        targets = np.broadcast_to(base, (n, horizon, self.model.nu)).copy()
        targets[:, :, self.active] += self.action_scale * candidates[:, :horizon]
        targets = np.clip(targets, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
        controls = np.repeat(targets, self.substeps, axis=1)
        control_spec = int(mujoco.mjtState.mjSTATE_CTRL)
        states, sensors = self.rollout.rollout(self.model, self.data, self._initial_state[None], controls,
                                              control_spec=control_spec,
                                              initial_warmstart=np.zeros((1, self.model.nv)))
        # MuJoCo position sensors describe the integration interval start.
        # Select one per control interval and compare with the matching reference.
        y = sensors[:, ::self.substeps]
        s = states[:, self.substeps - 1::self.substeps]
        ref = self._cost_reference
        w = self.weights
        com = y[..., self.sensor_slices["com"]]
        velocity = y[..., self.sensor_slices["com_vel"]]
        feet = np.stack((y[..., self.sensor_slices["left_foot_pos"]], y[..., self.sensor_slices["right_foot_pos"]]), axis=-2)
        footvel = np.stack((y[..., self.sensor_slices["left_foot_vel"]], y[..., self.sensor_slices["right_foot_vel"]]), axis=-2)
        quat = y[..., self.sensor_slices["torso_quat"]]
        cost = w["com_xy"] * np.sum((com[..., :2] - ref["com"][None, :, :2])**2, axis=-1)
        cost += w["com_z"] * (com[..., 2] - ref["com"][None, :, 2])**2
        cost += w["velocity"] * np.sum((velocity - ref["com_velocity"][None])**2, axis=-1)
        cost += w["orientation"] * np.sum(quat[..., 1:]**2, axis=-1)
        foot_error = feet - ref["feet"][None]
        cost += w["foot_xy"] * np.sum(foot_error[..., :2]**2, axis=(-1, -2))
        cost += w["foot_z"] * np.sum(foot_error[..., 2]**2, axis=-1)
        cost += w["foot_slip"] * np.sum(footvel[..., :2]**2 * ref["stance"][None, ..., None], axis=(-1, -2))
        joint_indices = 1 + self.model.jnt_qposadr[self.model.actuator_trnid[:, 0]]
        cost += w["joint"] * np.mean((s[..., joint_indices] - self._joint_reference[None])**2, axis=-1)
        cost += w["residual"] * np.mean(candidates[:, :horizon]**2, axis=-1)
        fallen = (com[..., 2] < .45) | (2 * np.sum(quat[..., 1:3]**2, axis=-1) > .5)
        cost += 1e4 * fallen
        # Divergent rollouts repeat time; they must not become low-cost candidates.
        divergent = np.any(np.diff(states[..., 0], axis=1) <= 0., axis=1)
        result = np.mean(cost, axis=1)
        result[divergent] = 1e8
        return np.nan_to_num(result, nan=1e8, posinf=1e8, neginf=1e8)

    def act(self, observation: RobotObservation):
        started = time.perf_counter()
        if not isinstance(observation, RobotObservation):
            raise TypeError("act requires RobotObservation(time, qpos, qvel)")
        t = float(observation.time)
        qpos, qvel = np.asarray(observation.qpos), np.asarray(observation.qvel)
        if (not np.isfinite(t) or t < 0 or qpos.shape != (self.model.nq,)
                or qvel.shape != (self.model.nv,) or not np.all(np.isfinite(qpos))
                or not np.all(np.isfinite(qvel))):
            raise ValueError("Robot observation must have finite time and correctly shaped finite qpos/qvel")
        future = t + np.arange(self.dial_config.horizon_steps + 1) * self.control_dt
        self._reference_controls = self.reference.batch(future)["joint_targets"]
        self._cost_reference = self.reference.batch(future[:-1])
        self._joint_reference = self._reference_controls[1:]
        # Build the state from the explicit observation, never from plant
        # force arrays, acceleration, contact forces or solver warm-start.
        mujoco.mj_resetData(self.model, self._observation_data)
        self._observation_data.time = t
        self._observation_data.qpos[:] = qpos
        self._observation_data.qvel[:] = qvel
        self._initial_state = np.zeros(mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_FULLPHYSICS))
        mujoco.mj_getState(self.model, self._observation_data, self._initial_state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        if not self._first:
            self.nodes = self.optimizer.shift(self.nodes, tail_action=np.zeros(len(self.active)))
        count = self.dial_config.init_iterations if self._first else self.dial_config.iterations
        result = self.optimizer.optimize(self.nodes, self._rollout_cost, iterations=count)
        self.nodes = result.nodes.copy()
        self._first = False
        target = self._reference_controls[0].copy()
        target[self.active] += self.action_scale * result.controls[0]
        self.last_plan_info = {"planning_s": time.perf_counter() - started,
                               "samples": self.dial_config.samples, "iterations": count,
                               "horizon_s": self.dial_config.horizon_steps * self.control_dt,
                               "diagnostics": result.diagnostics}
        return np.clip(target, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])

    def metadata(self):
        return {"method": "DIAL-MPC G1 CPU port", "config": self.config,
                "plant": "full floating-base/contact MuJoCo dynamics with position-PD actuators",
                "active_actuators": self.active.tolist(), "reference": self.reference.metadata(),
                "wind_prediction": self.wind_prediction,
                "observation": "time,qpos,qvel", "external_load_access": "none",
                "solver_warmstart": "zero; never copied from plant",
                "state_estimation": "ideal simulator pose/velocity; hardware requires an estimator",
                "future_random_disturbance_access": False, "real_time_claim": False}

    def close(self):
        self.rollout.close()
