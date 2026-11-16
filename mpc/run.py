"""Run actual G1 DIAL-MPC dynamics in calm, impulsive-load and vortex cases.

The CPU planner runs synchronously: reported wall time is not real-time
performance. No policy checkpoint or reinforcement learning is involved.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from sim_env.disturbances import DisturbanceConfig, DisturbanceScheduler
from sim_env.g1_physics import G1PhysicsEnv
from sim_env.wind_fields import WindFieldConfig, make_wind_field
from .observation import RobotObservation


CASES = ("calm", "random", "tornado")


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _state(env):
    feet = [env.model.body(f"{side}_ankle_roll_link").id for side in ("left", "right")]
    contacts = np.zeros(2, dtype=bool)
    for contact in env.data.contact[:env.data.ncon]:
        bodies = (int(env.model.geom_bodyid[contact.geom1]), int(env.model.geom_bodyid[contact.geom2]))
        for i, foot in enumerate(feet):
            if foot in bodies and 0 in bodies:
                contacts[i] = True
    return {"time": float(env.data.time), "qpos": env.data.qpos.copy(),
            "qvel": env.data.qvel.copy(), "ctrl": env.data.ctrl.copy(),
            "sole_position_m": env.data.site_xpos[list(env.profile.sole_site_ids)].copy(),
            "foot_contact": contacts,
            "upright": float(env.data.xmat[env.profile.root_body_id].reshape(3, 3)[2, 2])}


def run_episode(env, controller, *, duration, output, scenario, seed,
                controller_config=None, disturbance_metadata=None, wind_metadata=None):
    """Integrate an already initialized physical environment; never reset it."""
    output = Path(output)
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveModel(env.model, str(output / "model.mjb"), None)
    records = [_state(env)]
    plans, timings = [], []
    failure = None
    began = time.perf_counter()
    count = int(np.ceil(duration / env.control_dt - 1e-9))
    progress_at = time.perf_counter()
    for step in range(count):
        plan_start = time.perf_counter()
        try:
            targets = np.asarray(controller.act(RobotObservation.from_data(env.data)), dtype=float)
            if targets.shape != (env.model.nu,) or not np.all(np.isfinite(targets)):
                raise ValueError("Controller returned invalid joint-position targets")
            # G1PhysicsEnv clips to hardware position bounds. The planner must
            # model those same bounds; record clipping instead of concealing it.
            clipped = np.clip(targets, env.model.actuator_ctrlrange[:, 0], env.model.actuator_ctrlrange[:, 1])
            elapsed_plan = time.perf_counter() - plan_start
            plans.append({"time_s": float(env.data.time),
                          **getattr(controller, "last_plan_info", {}),
                          "wall_seconds": elapsed_plan,
                          "target_clipped_fraction": float(np.mean(targets != clipped))})
            timings.append(elapsed_plan)
            env.step_joint_targets(targets)
            if env.data.time <= records[-1]["time"]:
                raise RuntimeError("physical time did not advance")
            records.append(_state(env))
            failure = env.failure_reason()
        except (RuntimeError, ValueError, FloatingPointError) as error:
            failure = f"integration_or_controller_error: {error}"
        if failure:
            break
        if (step + 1) % 25 == 0 or time.perf_counter() - progress_at >= 15:
            print(f"{scenario}: {env.data.time:.2f}/{duration:.2f} simulated seconds, "
                  f"x={env.data.qpos[0]:.3f} m, z={env.data.qpos[2]:.3f} m, "
                  f"last plan {elapsed_plan:.3f} s", flush=True)
            progress_at = time.perf_counter()
    wall_seconds = time.perf_counter() - began
    trajectory = {key: np.asarray([record[key] for record in records]) for key in records[0]}
    trajectory["planner_seconds"] = np.asarray(timings)
    sample_shapes = ((), (3,), (3,), (3,), (env.model.nbody, 3), (env.model.nbody, 3), (env.model.nbody, 3))
    sample_keys = ("wind_sample_time_s", "wind_force_N", "wind_torque_about_com_Nm", "wind_com_m",
                   "wind_body_forces_N", "wind_body_torques_Nm", "wind_body_com_m")
    for i, (key, shape) in enumerate(zip(sample_keys, sample_shapes)):
        trajectory[key] = np.asarray([sample[i] for sample in env.wind_history]).reshape((-1, *shape))
    disturbance_shapes = ((), (env.model.nbody, 3), (env.model.nbody, 3), (env.model.nv,))
    disturbance_keys = ("disturbance_sample_time_s", "disturbance_body_forces_N",
                        "disturbance_body_torques_Nm", "disturbance_joint_torques_Nm")
    for i, (key, shape) in enumerate(zip(disturbance_keys, disturbance_shapes)):
        trajectory[key] = np.asarray([sample[i] for sample in env.disturbance_history]).reshape((-1, *shape))
    np.savez_compressed(output / "trajectory.npz", **trajectory)
    speed = float(getattr(controller, "command_speed", (controller_config or {}).get("speed", .2)))
    duration_actual = float(trajectory["time"][-1])
    progress = float(trajectory["qpos"][-1, 0] - trajectory["qpos"][0, 0])
    velocities = trajectory["qvel"][1:, :2]
    tracking_rmse = float(np.sqrt(np.mean(np.sum((velocities - [speed, 0])**2, axis=1)))) if len(velocities) else None
    contacts = trajectory["foot_contact"]
    liftoffs = np.sum(contacts[:-1] & ~contacts[1:], axis=0) if len(contacts) > 1 else np.zeros(2, dtype=int)
    landings = np.sum(~contacts[:-1] & contacts[1:], axis=0) if len(contacts) > 1 else np.zeros(2, dtype=int)
    completed = bool(failure is None and duration_actual >= duration - 1e-8)
    walking = bool(completed and progress >= .6 * speed * duration and np.all(landings >= 1)
                   and tracking_rmse is not None and tracking_rmse < .25)
    summary = {"controller": getattr(controller, "name", "DIAL-MPC G1 CPU port"), "scenario": scenario, "seed": int(seed),
               "model": "Unitree G1 29-DoF; full floating-base/contact MuJoCo dynamics",
               "actuation": "29 force-limited joint-position actuators; targets optimized by the controller",
               "execution": "synchronous offline CPU planning; no real-time claim",
               "requested_duration_s": float(duration), "elapsed_simulation_s": duration_actual,
               "completed_requested_duration": completed, "failure_reason": failure,
               "walking_success": walking,
               "walking_success_definition": "finish without failure; >=60% commanded forward travel; both feet land at least once; planar velocity RMSE<0.25m/s",
               "command_speed_mps": speed, "forward_progress_m": progress,
               "velocity_tracking_rmse_mps": tracking_rmse,
               "left_right_liftoffs": liftoffs.tolist(), "left_right_landings": landings.tolist(),
               "left_right_max_sole_height_m": np.max(trajectory["sole_position_m"][:, :, 2], axis=0).tolist(),
               "minimum_pelvis_height_m": float(np.min(trajectory["qpos"][:, 2])),
               "minimum_upright_cosine": float(np.min(trajectory["upright"])),
               "wall_seconds": wall_seconds,
               "planner_mean_seconds": float(np.mean(timings)) if timings else None,
               "planner_p95_seconds": float(np.percentile(timings, 95)) if timings else None,
               "control_dt_s": env.control_dt, "physics_dt_s": env.physics_dt,
               "controller_config": controller_config or {},
               "controller_metadata": controller.metadata() if hasattr(controller, "metadata") else {},
               "drag_model": env.drag_model, "wind_scale": env.wind_scale,
               "wind_config": asdict(env.wind_config), "wind_field": wind_metadata,
               "disturbances": disturbance_metadata, "max_drag_force_N": env.max_wind_force,
               "max_drag_torque_about_com_Nm": env.max_wind_torque_com,
               "surface_model": env.surface_drag.metadata() if env.surface_drag else None,
               "initial_qpos": trajectory["qpos"][0], "final_qpos": trajectory["qpos"][-1],
               "mujoco_warning_counts": env.data.warning.number.tolist(),
               "state_convention": "actual integrated states, including initial and last valid physical state",
               "load_convention": "loads at START of every 0.002s physics step; separate timestamps"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False, default=json_value), encoding="utf-8")
    (output / "planner.json").write_text(json.dumps(plans, indent=2, allow_nan=False, default=json_value), encoding="utf-8")
    return summary


def run_case(args, scenario):
    from .controller import DIALController
    config = json.loads(Path(args.controller_config).read_text(encoding="utf-8")) if args.controller_config else {}
    config["duration"] = max(float(config.get("duration", 0.)), args.duration)
    native_calm = scenario == "calm"
    wind_cfg = WindFieldConfig(kind="tornado" if scenario == "tornado" else "calm")
    field = None if native_calm else make_wind_field(wind_cfg)
    env = G1PhysicsEnv(wind_scale=0. if native_calm else args.wind_scale if scenario == "tornado" else 1.,
                       drag_model="none" if native_calm else "surface", wind_field=field, record_wind_history=True,
                       surface_max_patches=args.surface_patches)
    controller = DIALController(env, config)
    try:
        env.reset(qpos=controller.initial_qpos, base_xy=controller.initial_qpos[:2])
        controller.reset(seed=args.seed)
        scheduler = DisturbanceScheduler(env.model, DisturbanceConfig(
            enabled=scenario == "random", settle_time_s=.25,
            interval_s=(.35, .65), duration_s=(.12, .25)))
        scheduler.reset(seed=args.seed + 100000, horizon_s=args.duration, strength=args.disturbance_strength)
        env.disturbance_callback = None if native_calm else scheduler.sample
        output = Path(args.output) / scenario
        summary = run_episode(env, controller, duration=args.duration, output=output,
                              scenario=scenario, seed=args.seed, controller_config=config,
                              disturbance_metadata=scheduler.metadata(),
                              wind_metadata=None if field is None else field.metadata())
    finally:
        controller.close()
    print(json.dumps({key: summary[key] for key in ("scenario", "elapsed_simulation_s", "forward_progress_m",
                      "walking_success", "failure_reason", "wall_seconds")}, indent=2), flush=True)
    if args.video:
        from .render import render_run
        render_run(output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES),
                        help="calm: native walking without aerodynamic/injected loads; other cases retain surface drag")
    parser.add_argument("--duration", type=float, default=4.)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--controller-config", default=str(Path(__file__).parent / "configs" / "g1_demo.json"),
                        help="DIALController configuration JSON (default: validated G1 demo settings)")
    parser.add_argument("--wind-scale", type=float, default=1., help="wind speed multiplier for tornado case")
    parser.add_argument("--disturbance-strength", type=float, default=.2, help="fraction of 150N/20Nm/30Nm event peaks")
    parser.add_argument("--surface-patches", type=int, default=4096)
    parser.add_argument("--output", default="outputs/dial_mpc/demo",
                        help="run directory (default: %(default)s); each case directory must be empty")
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.duration) or args.duration <= 0:
        parser.error("duration must be finite and positive")
    summaries = [run_case(args, scenario) for scenario in args.cases]
    return 0 if all(row["completed_requested_duration"] for row in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
