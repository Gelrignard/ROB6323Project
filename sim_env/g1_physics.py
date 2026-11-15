"""Physical Unitree G1 environment with native joint-position actuators.

Run ``python -m sim_env.g1_physics --duration 2 --wind-scale 1`` to apply
mesh-surface aerodynamic loads during a physical standing test. Use
``--drag-model lumped`` to retain the earlier torso-force approximation.
This is genuine free-base/contact dynamics, not LQR walking.
The default holds the vendor stand targets; it is not a wind-balancing policy.
"""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from .config import DEFAULT_TORNADO
from .g1_model import load_g1_model, get_g1_profile
from .wind import local_to_world, wind_force


class G1PhysicsEnv:
    """Full G1 physics. ``step_joint_targets`` expects 29 angles in radians."""

    def __init__(self, wind_scale=0.0, control_dt=0.02, physics_dt=0.002, wind_config=None,
                 drag_model="lumped", surface_max_patches=4096, record_wind_history=False,
                 wind_field=None, disturbance_callback=None):
        if drag_model not in ("none", "lumped", "surface"):
            raise ValueError("drag_model must be 'none', 'lumped' or 'surface'")
        if not np.all(np.isfinite([wind_scale, control_dt, physics_dt])) or wind_scale < 0 or min(control_dt, physics_dt) <= 0:
            raise ValueError("time steps must be positive and wind_scale nonnegative")
        ratio = control_dt / physics_dt
        if ratio < 1 or not np.isclose(ratio, round(ratio)):
            raise ValueError("control_dt must be an integer multiple of physics_dt")
        self.model = load_g1_model()
        self.model.opt.timestep = physics_dt
        self.profile = get_g1_profile(self.model)
        self.data = mujoco.MjData(self.model)
        # Effective projected area is a modeling assumption, not a measured
        # G1 aerodynamic coefficient. Keep it shared with the G1 LIP preset.
        self.wind_config = replace(DEFAULT_TORNADO, projected_area_m2=0.25) if wind_config is None else wind_config
        if self.wind_config.patch_size_m != 10.0:
            raise ValueError("the G1 scene currently uses a 10 m test patch")
        self.drag_model = drag_model
        self.wind_field = wind_field
        self.disturbance_callback = disturbance_callback
        self.surface_drag = None
        if drag_model == "surface":
            from .surface_drag import SurfaceDrag
            self.surface_drag = SurfaceDrag(self.model, wind_config=self.wind_config,
                                            max_patches=surface_max_patches)
        self.record_wind_history = bool(record_wind_history)
        self.wind_scale = float(wind_scale)
        self.control_dt, self.physics_dt = float(control_dt), float(physics_dt)
        self.substeps = int(round(ratio))
        self.last_wind_force = np.zeros(3)
        self.last_wind_torque_com = np.zeros(3)
        self.last_wind_com = np.zeros(3)
        self.last_body_wind_forces = np.zeros((self.model.nbody, 3))
        self.last_body_wind_torques = np.zeros((self.model.nbody, 3))
        self.last_body_com = np.zeros((self.model.nbody, 3))
        self.last_disturbance_body_forces = np.zeros((self.model.nbody, 3))
        self.last_disturbance_body_torques = np.zeros((self.model.nbody, 3))
        self.last_disturbance_joint_torques = np.zeros(self.model.nv)
        self.last_disturbance_active_event_ids = ()
        self.disturbance_sample_time = 0.0
        self._joint_disturbance_mask = np.zeros(self.model.nv, dtype=bool)
        for joint_id in self.model.actuator_trnid[:, 0]:
            if joint_id >= 0 and self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE:
                self._joint_disturbance_mask[self.model.jnt_dofadr[joint_id]] = True
        self.wind_force_sample_time = 0.0
        self.max_wind_force = 0.0
        self.max_wind_torque_com = 0.0
        self.max_contacts = 0
        self.reset()

    def reset(self, base_xy=(1.0, 5.0), qpos=None):
        """Reset once; subsequent physical steps never overwrite qpos/qvel."""
        xy = np.asarray(base_xy, dtype=float)
        if xy.shape != (2,) or not np.all(np.isfinite(xy)):
            raise ValueError("base_xy must be a finite (2,) vector")
        pose = self.profile.home_qpos.copy() if qpos is None else np.asarray(qpos, dtype=float).copy()
        if pose.shape != (self.model.nq,) or not np.all(np.isfinite(pose)):
            raise ValueError(f"qpos must have shape ({self.model.nq},)")
        mujoco.mj_resetData(self.model, self.data)
        pose[:2] = xy
        self.data.qpos[:] = pose
        self.data.ctrl[:] = self.profile.home_ctrl
        self.last_wind_force[:] = 0
        self.last_wind_torque_com[:] = 0
        self.last_body_wind_forces[:] = 0
        self.last_body_wind_torques[:] = 0
        self.wind_force_sample_time = 0.0
        self.max_wind_force = 0.0
        self.max_wind_torque_com = 0.0
        self.max_contacts = 0
        self.wind_history = []
        self.disturbance_history = []
        self.last_disturbance_body_forces[:] = 0
        self.last_disturbance_body_torques[:] = 0
        self.last_disturbance_joint_torques[:] = 0
        self.last_disturbance_active_event_ids = ()
        self.disturbance_sample_time = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.last_wind_com[:] = self.data.subtree_com[self.profile.root_body_id]
        self.last_body_com[:] = self.data.xipos
        return self.observation()

    def observation(self):
        return {"qpos": self.data.qpos.copy(), "qvel": self.data.qvel.copy(),
                "com": self.data.subtree_com[self.profile.root_body_id].copy(),
                "sole_positions": self.data.site_xpos[list(self.profile.sole_site_ids)].copy(),
                "contacts": int(self.data.ncon), "time": float(self.data.time),
                "wind_force": self.last_wind_force.copy(),
                "wind_torque_com": self.last_wind_torque_com.copy(),
                "wind_com": self.last_wind_com.copy(),
                "wind_body_forces": self.last_body_wind_forces.copy(),
                "wind_body_torques": self.last_body_wind_torques.copy(),
                "wind_force_sample_time": self.wind_force_sample_time,
                "disturbance_body_forces": self.last_disturbance_body_forces.copy(),
                "disturbance_body_torques": self.last_disturbance_body_torques.copy(),
                "disturbance_joint_torques": self.last_disturbance_joint_torques.copy(),
                "disturbance_active_event_ids": self.last_disturbance_active_event_ids,
                "disturbance_sample_time": self.disturbance_sample_time}

    def _apply_wind(self):
        # mj_step integrates qpos after its forward pass. Refresh kinematics
        # before using positions/Jacobians for the next force evaluation.
        mujoco.mj_forward(self.model, self.data)
        self.last_body_wind_forces[:] = 0
        self.last_body_wind_torques[:] = 0
        self.last_body_com[:] = self.data.xipos
        self.last_wind_com[:] = self.data.subtree_com[self.profile.root_body_id]
        self.last_wind_force[:] = 0
        self.last_wind_torque_com[:] = 0
        if self.surface_drag is not None:
            result = self.surface_drag.evaluate(self.data, wind_scale=self.wind_scale, wind_field=self.wind_field)
            self.last_body_wind_forces[:] = result.body_forces
            self.last_body_wind_torques[:] = result.body_torques
            self.last_wind_force[:] = result.total_force
            self.last_wind_torque_com[:] = result.total_torque_com
            self.last_wind_com[:] = result.com
        elif self.drag_model == "lumped":
            # Apply the assumed whole projected area ONCE at torso CoM.
            # It has zero local body torque but a moment about whole-robot CoM.
            body_id = self.profile.torso_body_id
            jacobian = np.zeros((3, self.model.nv))
            mujoco.mj_jacBodyCom(self.model, self.data, jacobian, None, body_id)
            velocity = jacobian @ self.data.qvel
            position = self.data.xipos[body_id]
            if self.wind_field is None:
                force = wind_force(local_to_world(position[:2], self.wind_config), velocity[:2],
                                   self.wind_config, scale=self.wind_scale)
                self.last_wind_force[:] = [force[0], force[1], 0.0]
            else:
                world_position = position.copy()
                world_position[:2] = local_to_world(position[:2], self.wind_config)
                air = np.asarray(self.wind_field(world_position, float(self.data.time)), dtype=float) * self.wind_scale
                if air.shape != (3,) or not np.all(np.isfinite(air)):
                    raise ValueError("wind_field must return finite velocities of shape (3,)")
                relative = air - velocity
                coefficient = .5 * self.wind_config.air_density_kg_m3 * self.wind_config.drag_coefficient * self.wind_config.projected_area_m2
                self.last_wind_force[:] = coefficient * np.linalg.norm(relative) * relative
            self.last_body_wind_forces[body_id] = self.last_wind_force
            self.last_wind_torque_com[:] = np.cross(position - self.last_wind_com, self.last_wind_force)
        if not (np.all(np.isfinite(self.last_body_wind_forces)) and
                np.all(np.isfinite(self.last_body_wind_torques))):
            raise RuntimeError("nonfinite aerodynamic body loads")
        self.wind_force_sample_time = float(self.data.time)
        # xfrc_applied uses world-frame force and torque about each body CoM.
        # The surface and lumped models are exclusive, never superimposed.
        self.data.xfrc_applied[:, :3] = self.last_body_wind_forces
        self.data.xfrc_applied[:, 3:] = self.last_body_wind_torques
        self.max_wind_force = max(self.max_wind_force, float(np.linalg.norm(self.last_wind_force)))
        self.max_wind_torque_com = max(self.max_wind_torque_com, float(np.linalg.norm(self.last_wind_torque_com)))

    def drag_metadata(self):
        if self.drag_model == "none":
            return {"model": "aerodynamics explicitly disabled; external disturbance loads remain active"}
        if self.surface_drag is not None:
            return self.surface_drag.metadata()
        return {"model": "lumped relative-velocity horizontal drag at torso CoM",
                "projected_area_m2_assumption": self.wind_config.projected_area_m2,
                "drag_coefficient": self.wind_config.drag_coefficient,
                "air_density_kg_m3": self.wind_config.air_density_kg_m3}

    def _apply_disturbances(self):
        """Add external loads after wind, without changing qpos or qvel."""
        self.data.qfrc_applied[:] = 0.0
        self.last_disturbance_body_forces[:] = 0.0
        self.last_disturbance_body_torques[:] = 0.0
        self.last_disturbance_joint_torques[:] = 0.0
        self.last_disturbance_active_event_ids = ()
        self.disturbance_sample_time = float(self.data.time)
        if self.disturbance_callback is not None:
            sample = self.disturbance_callback(self.disturbance_sample_time)
            for name, shape in (("body_forces", (self.model.nbody, 3)),
                                ("body_torques", (self.model.nbody, 3)), ("joint_torques", (self.model.nv,))):
                value = np.asarray(getattr(sample, name), dtype=float)
                if value.shape != shape or not np.all(np.isfinite(value)):
                    raise ValueError(f"disturbance {name} must be finite with shape {shape}")
            if np.any(np.asarray(sample.joint_torques)[~self._joint_disturbance_mask] != 0):
                raise ValueError("joint disturbances may only target actuated hinge DOFs")
            outside_robot = self.model.body_rootid != self.profile.root_body_id
            if np.any(np.asarray(sample.body_forces)[outside_robot] != 0) or np.any(np.asarray(sample.body_torques)[outside_robot] != 0):
                raise ValueError("link disturbances may only target robot bodies")
            self.last_disturbance_body_forces[:] = sample.body_forces
            self.last_disturbance_body_torques[:] = sample.body_torques
            self.last_disturbance_joint_torques[:] = sample.joint_torques
            self.last_disturbance_active_event_ids = tuple(sample.active_event_ids)
        self.data.xfrc_applied[:, :3] += self.last_disturbance_body_forces
        self.data.xfrc_applied[:, 3:] += self.last_disturbance_body_torques
        self.data.qfrc_applied[:] = self.last_disturbance_joint_torques
        if self.record_wind_history:
            self.disturbance_history.append((self.disturbance_sample_time,
                                            self.last_disturbance_body_forces.copy(),
                                            self.last_disturbance_body_torques.copy(),
                                            self.last_disturbance_joint_torques.copy()))

    def _record_wind_sample(self):
        if self.record_wind_history:
            self.wind_history.append((self.wind_force_sample_time, self.last_wind_force.copy(),
                                      self.last_wind_torque_com.copy(), self.last_wind_com.copy(),
                                      self.last_body_wind_forces.copy(), self.last_body_wind_torques.copy(),
                                      self.last_body_com.copy()))

    def step_joint_targets(self, targets):
        """Integrate real contacts; clamp angle targets to actuator ranges."""
        targets = np.asarray(targets, dtype=float)
        if targets.shape != (self.model.nu,) or not np.all(np.isfinite(targets)):
            raise ValueError(f"targets must be finite shape ({self.model.nu},)")
        self.data.ctrl[:] = np.clip(targets, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
        expected = float(self.data.time) + self.control_dt
        for _ in range(self.substeps):
            self._apply_wind()
            self._apply_disturbances()
            self._record_wind_sample()
            mujoco.mj_step(self.model, self.data)
            self.max_contacts = max(self.max_contacts, int(self.data.ncon))
        mujoco.mj_forward(self.model, self.data)
        if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
            raise RuntimeError("G1 produced nonfinite physical state")
        if np.any(self.data.warning.number) or not np.isclose(self.data.time, expected, rtol=0, atol=1e-9):
            raise RuntimeError("G1 MuJoCo warning or unexpected simulator reset")
        return self.observation()

    def failure_reason(self):
        pelvis = self.data.xpos[self.profile.root_body_id]
        upright = self.data.xmat[self.profile.root_body_id].reshape(3, 3)[2, 2]
        if pelvis[2] < 0.45 or upright < 0.5:
            return "pelvis_height_or_tilt_limit"
        if np.any(pelvis[:2] < 0) or np.any(pelvis[:2] > 10):
            return "left_test_region"
        return None


def run_standing(duration=2.0, wind_scale=0.0, viewer=False, output=Path("sim_env/outputs/g1_physics"),
                 drag_model="lumped", surface_max_patches=4096):
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be positive and finite")
    env = G1PhysicsEnv(wind_scale=wind_scale, drag_model=drag_model,
                       surface_max_patches=surface_max_patches, record_wind_history=True)
    initial = env.data.qpos.copy()
    state_history, velocity_history, time_history = [initial], [env.data.qvel.copy()], [0.0]
    window = None
    if viewer:
        import mujoco.viewer
        window = mujoco.viewer.launch_passive(env.model, env.data)
        window.cam.azimuth = 135
        window.cam.elevation = -20
        window.cam.distance = 3.0
        window.cam.lookat[:] = [1, 5, .7]
    reason = None
    try:
        for _ in range(max(1, int(np.ceil(duration / env.control_dt)))):
            started = time.perf_counter()
            if window is not None:
                if not window.is_running():
                    reason = "viewer_closed"
                    break
                with window.lock():
                    env.step_joint_targets(env.profile.home_ctrl)
                window.sync()
                time.sleep(max(0.0, env.control_dt - (time.perf_counter() - started)))
            else:
                env.step_joint_targets(env.profile.home_ctrl)
            state_history.append(env.data.qpos.copy())
            velocity_history.append(env.data.qvel.copy())
            time_history.append(float(env.data.time))
            reason = env.failure_reason()
            if reason:
                break
    finally:
        if window is not None:
            window.close()
    summary = {"model": "Unitree G1 29-DoF, free floating base, physical contacts",
               "controller": "vendor joint-position stand targets; NOT a walking or wind-balance controller",
               "drag_model": env.drag_model, "drag_model_metadata": env.drag_metadata(),
               "wind_model": ("aerodynamics explicitly disabled" if env.drag_model == "none" else
                              "mesh-surface pressure quadrature applied as per-body forces and torques"
                              if env.surface_drag is not None else "relative-velocity horizontal drag applied once at torso CoM"),
               "wind_config": asdict(env.wind_config),
               "wind_scale": wind_scale, "elapsed_simulation_s": float(env.data.time),
               "requested_duration_s": duration,
               "projected_area_m2_assumption": env.wind_config.projected_area_m2 if env.drag_model == "lumped" else None,
               "completed_requested_duration": reason is None, "failure_reason": reason,
               "nq": env.model.nq, "nv": env.model.nv, "nu": env.model.nu,
               "mass_kg": env.profile.mass, "max_contacts": env.max_contacts,
               "max_wind_force_N": env.max_wind_force,
               "max_wind_torque_about_com_Nm": env.max_wind_torque_com,
               "last_wind_force_N": env.last_wind_force.tolist(),
               "last_wind_torque_about_com_Nm": env.last_wind_torque_com.tolist(),
               "last_wind_sample_time_s": env.wind_force_sample_time,
               "wind_load_samples": len(env.wind_history),
               "disturbance_load_samples": len(env.disturbance_history),
               "load_sample_convention": "loads sampled at START of every physics step; wind_sample_time_s is separate from state time",
               "load_coordinate_convention": "world-aligned patch coordinates; body torques about body CoM; total torque about whole-robot CoM at each load sample",
               "initial_qpos": initial.tolist(), "final_qpos": env.data.qpos.tolist(),
               "mujoco_warning_counts": env.data.warning.number.tolist()}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    sample_shapes = ((), (3,), (3,), (3,), (env.model.nbody, 3), (env.model.nbody, 3), (env.model.nbody, 3))
    sample_keys = ("wind_sample_time_s", "wind_force_N", "wind_torque_about_com_Nm", "wind_com_m",
                   "wind_body_forces_N", "wind_body_torques_Nm", "wind_body_com_m")
    loads = {key: np.asarray([sample[index] for sample in env.wind_history]).reshape((-1, *shape))
             for index, (key, shape) in enumerate(zip(sample_keys, sample_shapes))}
    disturbance_shapes = ((), (env.model.nbody, 3), (env.model.nbody, 3), (env.model.nv,))
    disturbance_keys = ("disturbance_sample_time_s", "disturbance_body_forces_N",
                        "disturbance_body_torques_Nm", "disturbance_joint_torques_Nm")
    loads.update({key: np.asarray([sample[index] for sample in env.disturbance_history]).reshape((-1, *shape))
                  for index, (key, shape) in enumerate(zip(disturbance_keys, disturbance_shapes))})
    np.savez_compressed(output / "physical_state.npz", time=np.asarray(time_history),
                        qpos=np.asarray(state_history), qvel=np.asarray(velocity_history),
                        wind_body_names=np.array([env.model.body(index).name for index in range(env.model.nbody)]),
                        **loads)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--wind-scale", type=float, default=0.0)
    parser.add_argument("--drag-model", choices=("none", "lumped", "surface"), default="surface",
                        help="surface: distributed mesh loads (default); lumped: earlier torso-force approximation")
    parser.add_argument("--surface-patches", type=int, default=4096, help="maximum deterministic surface quadrature patches")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("sim_env/outputs/g1_physics"))
    args = parser.parse_args()
    summary = run_standing(args.duration, args.wind_scale, args.viewer, args.output,
                           drag_model=args.drag_model, surface_max_patches=args.surface_patches)
    print(json.dumps(summary, indent=2))
    return 0 if summary["completed_requested_duration"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
