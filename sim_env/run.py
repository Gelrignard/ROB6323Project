"""Run the reduced MuJoCo walker: python -m sim_env.run --video --snapshot.

The default controller is finite-horizon affine LQR plus support projection.
The nominal gait is prescribed, not learned. See README for scope/assumptions.
"""

import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
import time
import platform

import numpy as np

from lqr import LIPWalkingLQR
from .config import DEFAULT_TORNADO
from .environment import ReducedWalkingEnv, SimulationConfig
from .gait import GaitConfig, WalkingReference, project_to_support
from .wind import local_to_world, tornado_velocity, wind_force


def simulate(config=None, *, duration=None, horizon_seconds=1.6,
             wind_aware=True, controller_mode="lqr", initial_offset=None,
             gait_config=None, wind_config=None):
    cfg = config or SimulationConfig()
    wind_config = DEFAULT_TORNADO if wind_config is None else wind_config
    if controller_mode not in ("lqr", "open-loop"):
        raise ValueError("controller_mode must be lqr or open-loop")
    gait = WalkingReference(gait_config or GaitConfig(com_height=cfg.height, gravity=cfg.gravity))
    if not np.isclose(gait.config.com_height, cfg.height) or not np.isclose(gait.config.gravity, cfg.gravity):
        raise ValueError("gait reference and simulated dynamics must use the same height and gravity")
    duration = gait.duration if duration is None else float(duration)
    if not np.isfinite(duration) or duration <= 0 or duration > gait.duration:
        raise ValueError(f"duration must be in (0, {gait.duration}]")
    if not np.isfinite(horizon_seconds) or horizon_seconds < cfg.control_dt:
        raise ValueError("horizon_seconds must be finite and >= control_dt")
    horizon_steps = int(round(horizon_seconds / cfg.control_dt))
    controller = LIPWalkingLQR(cfg.control_dt, cfg.height, cfg.mass, horizon_steps, gravity=cfg.gravity)
    env = ReducedWalkingEnv(cfg, wind_config)
    x0 = gait.sample(0.0)["com_ref"].copy()
    if initial_offset is not None:
        offset = np.asarray(initial_offset, dtype=float)
        if offset.shape != (4,) or not np.all(np.isfinite(offset)):
            raise ValueError("initial_offset must be finite shape (4,)")
        x0 += offset
    env.reset(x0)
    records = []
    compute_seconds = []
    requested_steps = max(1, int(np.ceil(duration / cfg.control_dt - 1e-10)))
    failure = None
    for _ in range(requested_steps):
        started = time.perf_counter()
        t = float(env.data.time)
        reference = gait.sample(t)
        preview = gait.sample_horizon(t, cfg.control_dt, horizon_steps)
        forecast = np.zeros((horizon_steps, 2))
        if wind_aware:
            positions = preview["com_ref"][:-1, [0, 2]]
            velocities = preview["com_ref"][:-1, [1, 3]]
            forecast = wind_force(local_to_world(positions, wind_config), velocities, wind_config, scale=cfg.wind_scale)
        # Same desired CoM motion requires shifted CoP to offset forecast wind.
        input_reference = preview["cop_ref"] + cfg.height / (cfg.mass * cfg.gravity) * forecast
        if controller_mode == "lqr":
            command = controller.command(env.state, preview["com_ref"], input_reference, forecast)
            raw_cop = command.unconstrained_cop
        else:
            raw_cop = input_reference[0].copy()
        held_support = gait.support_over_interval(t, cfg.control_dt)
        cop = project_to_support(raw_cop, held_support)
        compute_seconds.append(time.perf_counter() - started)
        row = {
            "time": t, "state": env.state.copy(), "reference": reference["com_ref"].copy(),
            "cop": cop.copy(), "raw_cop": raw_cop.copy(),
            "cop_reference": input_reference[0].copy(),
            "feet": reference["feet"].copy(), "phase": reference["phase"],
            "step_count": reference["step_count"],
            "projection": float(np.linalg.norm(cop - raw_cop)),
            "support_polygon": reference["support_polygon"].copy(),
            "hold_support_polygon": held_support.copy(),
        }
        # Log all samples at the stated interval-start time. Effort is
        # accumulated separately at every physics substep in the environment.
        position, velocity = row["state"][[0, 2]], row["state"][[1, 3]]
        row["wind_velocity"] = tornado_velocity(local_to_world(position, wind_config), wind_config, scale=cfg.wind_scale)
        row["wind_force"] = wind_force(local_to_world(position, wind_config), velocity, wind_config, scale=cfg.wind_scale)
        row["ground_force"] = cfg.mass * cfg.gravity / cfg.height * (position - cop)
        env.step(cop, held_support)
        records.append(row)
        failure = env.failure_reason(gait.sample(float(env.data.time)))
        if failure:
            break
    final_reference = gait.sample(float(env.data.time))
    positions = np.array([r["state"][[0, 2]] for r in records])
    refs = np.array([r["reference"][[0, 2]] for r in records])
    error = positions - refs
    final_error = float(np.linalg.norm(env.state[[0, 2]] - final_reference["com_ref"][[0, 2]]))
    summary = {
        "model": "constant-height 2D LIP; prescribed feet; visual-only legs; no articulated contact dynamics",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "controller": "affine finite-horizon LQR with support-polygon projection" if controller_mode == "lqr" else "prescribed open-loop CoP",
        "wind_information": "known spatial field evaluated along nominal preview" if wind_aware else "no wind feedforward",
        "simulation_config": asdict(cfg), "wind_config": asdict(wind_config),
        "gait_config": asdict(gait.config), "horizon_seconds": horizon_steps * cfg.control_dt,
        "requested_duration_s": duration, "elapsed_simulation_s": float(env.data.time),
        "completed_requested_duration": failure is None and len(records) == requested_steps,
        "tracking_success": failure is None and final_error < 0.15,
        "failure_reason": failure,
        "completed_footsteps": int(final_reference["step_count"]),
        "initial_state": x0.tolist(), "final_state": env.state.tolist(),
        "displacement_m": float(np.linalg.norm(env.state[[0, 2]] - x0[[0, 2]])),
        "position_rmse_m": float(np.sqrt(np.mean(np.sum(error * error, axis=1)))),
        "max_position_error_m": float(np.max(np.linalg.norm(error, axis=1))),
        "final_position_error_m": final_error,
        "projected_command_fraction": float(np.mean([r["projection"] > 1e-9 for r in records])),
        "max_command_projection_m": float(max(r["projection"] for r in records)),
        "max_friction_utilization": env.max_friction_ratio,
        "max_wind_speed_mps": env.max_wind_speed,
        "max_wind_force_N": env.max_wind_force,
        "ground_force_squared_integral_N2s": env.ground_force_squared_integral,
        "trajectory_sample_convention": "states, references, CoP and forces at control-interval START; final state stored separately here",
        "controller_median_ms": float(1000 * np.median(compute_seconds)),
        "controller_max_ms": float(1000 * max(compute_seconds)),
        "controller_timing_includes": "reference construction, known-wind forecast, LQR, and support projection; excludes physics/rendering",
        "mujoco_warning_counts": env.data.warning.number.tolist(),
        "limitations": ["No full-body fall/contact validation", "CoP projection is not constrained optimal control",
                        "Ground-force integral is an effort proxy, not electrical/mechanical energy",
                        "Prescribed horizontal vortex is not CFD or a calibrated tornado"],
    }
    return records, summary, env, gait


def save_results(records, summary, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    columns = ("time_s", "x_m", "vx_mps", "y_m", "vy_mps", "ref_x_m", "ref_vx_mps", "ref_y_m", "ref_vy_mps",
               "cop_x_m", "cop_y_m", "raw_cop_x_m", "raw_cop_y_m", "wind_x_mps", "wind_y_mps",
               "wind_force_x_N", "wind_force_y_N", "ground_force_x_N", "ground_force_y_N", "cop_projection_m", "completed_footsteps", "phase")
    with (output / "trajectory.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for r in records:
            writer.writerow([r["time"], *r["state"], *r["reference"], *r["cop"], *r["raw_cop"],
                             *r["wind_velocity"], *r["wind_force"], *r["ground_force"], r["projection"], r["step_count"], r["phase"]])
    np.savez_compressed(output / "trajectory.npz", **{
        key: np.array([r[key] for r in records]) for key in
        ("time", "state", "reference", "cop", "raw_cop", "feet", "wind_force", "wind_velocity", "ground_force", "projection", "step_count")
    })
    from .render import plot_results
    plot_results(records, summary, output / "tracking.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("g1", "lip"), default="g1", help="G1-parameterized LQR and G1 IK replay; lip retains the original diagnostic model")
    parser.add_argument("--wind-scale", type=float, default=None, help="multiply wind velocity; default .5 for G1, 1 for original LIP")
    parser.add_argument("--duration", type=float, default=None, help="seconds; default complete prescribed gait")
    parser.add_argument("--horizon", type=float, default=1.6, help="LQR preview horizon in seconds")
    parser.add_argument("--blind", action="store_true", help="disable known-wind feedforward")
    parser.add_argument("--controller", choices=("lqr", "open-loop"), default="lqr")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--video", action="store_true", help="render an annotated MP4 after simulation")
    parser.add_argument("--snapshot", action="store_true", help="render a MuJoCo preview PNG after simulation")
    parser.add_argument("--no-force-overlay", action="store_true", help="omit the drag-only overlay from rendered media")
    parser.add_argument("--drag-profile", choices=("surface", "lumped"), default=None,
                        help="drag-only video diagnostic; default surface for G1, lumped for lip; does not change reduced LQR dynamics")
    parser.add_argument("--viewer", action="store_true", help="replay the recorded run in the interactive MuJoCo viewer")
    args = parser.parse_args()
    drag_profile = args.drag_profile or ("surface" if args.robot == "g1" else "lumped")
    if drag_profile == "surface" and args.robot != "g1":
        parser.error("--drag-profile surface requires --robot g1")
    wind_scale = args.wind_scale if args.wind_scale is not None else (.5 if args.robot == "g1" else 1.0)
    args.output = args.output or Path("sim_env/outputs/g1_lqr" if args.robot == "g1" else "sim_env/outputs/lqr_lip")
    gait_config, wind_config = None, None
    if args.robot == "g1":
        from .g1_model import get_g1_profile
        profile = get_g1_profile()
        # Full-gait IK validates this crouch without violating ankle limits.
        config = SimulationConfig(mass=profile.mass, height=profile.com_height - .04, friction=.6, wind_scale=wind_scale)
        gait_config = GaitConfig(com_height=config.height, gravity=config.gravity, step_width=profile.step_width,
                                 foot_half_length=profile.sole_half_length, foot_half_width=profile.sole_half_width)
        # Explicit aerodynamic approximation, not measured G1 coefficients.
        wind_config = replace(DEFAULT_TORNADO, projected_area_m2=.25)
    else:
        config = SimulationConfig(wind_scale=wind_scale)
    records, summary, env, gait = simulate(config, duration=args.duration,
                                           horizon_seconds=args.horizon, wind_aware=not args.blind,
                                           controller_mode=args.controller, gait_config=gait_config, wind_config=wind_config)
    summary["robot"] = "Unitree G1 29-DOF" if args.robot == "g1" else "diagnostic LIP"
    summary["visualization"] = "G1 joint-limited inverse-kinematics replay; not full-body walking physics" if args.robot == "g1" else "prescribed visual legs"
    if args.robot == "g1":
        summary["ik_validation"] = "Not checked by the reduced simulation. Media generation saves diagnostics for rendered samples in g1_ik_diagnostics.json."
    summary["requested_media_drag_profile"] = drag_profile if not args.no_force_overlay else "disabled"
    summary["surface_profile_feedback"] = "Surface media is a diagnostic on prescribed G1 motion. Reduced LQR dynamics still use lumped drag."
    save_results(records, summary, args.output)
    print(json.dumps({k: summary[k] for k in ("tracking_success", "failure_reason", "elapsed_simulation_s", "completed_footsteps", "displacement_m", "position_rmse_m", "max_wind_speed_mps", "controller_median_ms")}, indent=2))
    print(f"Results: {args.output.resolve()}")
    if args.video or args.snapshot or args.viewer:
        from .render import render_recording, replay_viewer
        if args.video or args.snapshot:
            render_recording(env, gait, records, args.output, video=args.video, snapshot=args.snapshot,
                             controller_label="LQR" if args.controller == "lqr" else "Open-loop", robot=args.robot,
                             force_overlay=not args.no_force_overlay, drag_profile=drag_profile)
        if args.viewer:
            replay_viewer(env, gait, records, robot=args.robot)
    return 0 if summary["tracking_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
