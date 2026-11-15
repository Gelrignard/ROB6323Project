"""Reproducible plots and MuJoCo replay (rendering never drives dynamics)."""

from pathlib import Path
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .wind import local_to_world, tornado_velocity


def plot_results(records, summary, path):
    t = np.array([r["time"] for r in records])
    x = np.array([r["state"] for r in records])
    ref = np.array([r["reference"] for r in records])
    cop = np.array([r["cop"] for r in records])
    wind = np.array([r["wind_force"] for r in records])
    lower = np.array([np.min(r["support_polygon"], axis=0) for r in records])
    upper = np.array([np.max(r["support_polygon"], axis=0) for r in records])
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": .2}):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
        ax = axes[0, 0]
        ax.plot(ref[:, 0], ref[:, 2], "--", color="#768493", label="Nominal CoM")
        ax.plot(x[:, 0], x[:, 2], color="#007e99", label="MuJoCo CoM")
        ax.scatter(x[0, 0], x[0, 2], color="#2b956b", zorder=3, label="Start")
        ax.scatter(x[-1, 0], x[-1, 2], color="#ec7b3e", zorder=3, label="End")
        ax.set(xlabel="Patch-local x [m]", ylabel="Patch-local y [m]", title="Walking trajectory (lateral scale enlarged)")
        ax.legend(loc="upper right", fontsize=8)
        ax = axes[0, 1]
        ax.plot(t, 1000 * (x[:, 0] - ref[:, 0]), label="Forward error")
        ax.plot(t, 1000 * (x[:, 2] - ref[:, 2]), label="Lateral error")
        ax.set(xlabel="Time [s]", ylabel="CoM tracking error [mm]", title="Feedback tracking")
        ax.legend(fontsize=8)
        ax = axes[1, 0]
        ax.fill_between(t, lower[:, 1], upper[:, 1], color="#d9e9ed", label="Support y extent")
        ax.plot(t, cop[:, 1], color="#d68614", lw=1.2, label="Applied CoP y")
        ax.plot(t, x[:, 2], color="#007e99", lw=1.2, label="CoM y")
        ax.set(xlabel="Time [s]", ylabel="Patch-local y [m]", title="Alternating support and pressure control")
        ax.legend(fontsize=8, loc="upper right")
        ax = axes[1, 1]
        ax.plot(t, wind[:, 0], label="Wind force x")
        ax.plot(t, wind[:, 1], label="Wind force y")
        ax.set(xlabel="Time [s]", ylabel="Force [N]", title="Relative-velocity aerodynamic load")
        ax.legend(fontsize=8)
        status = "completed" if summary["tracking_success"] else "failed"
        label = "LQR" if summary["controller"].startswith("affine") else "Open-loop"
        fig.suptitle(f"Reduced LIP walking with {label} | {status} | {summary['completed_footsteps']} footsteps\n"
                     f"Wind scale {summary['simulation_config']['wind_scale']:g}  |  "
                     f"Position RMSE {1000*summary['position_rmse_m']:.1f} mm  |  "
                     "Prescribed feet; visual legs", fontsize=15, color="#152b40")
        fig.savefig(path, dpi=170, facecolor="white")
        plt.close(fig)


def _font(size):
    for name in ("C:/Windows/Fonts/segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _annotate(frame, row, wind_scale, controller_label="LQR", robot="lip"):
    picture = Image.fromarray(frame)
    draw = ImageDraw.Draw(picture, "RGBA")
    draw.rectangle((0, 0, picture.width, 84), fill=(12, 25, 38, 230))
    surface = row.get("drag_model") == "surface"
    title = ("Unitree G1 | Surface drag profile" if surface else
             f"Unitree G1 | {controller_label} trajectory replay" if robot == "g1" else
             f"{controller_label} walking in the tornado test region")
    draw.text((22, 10), title, font=_font(26), fill="white")
    draw.text((22, 47), f"t = {row['time']:.2f} s   |   step {row['step_count']}   |   "
              f"wind {np.linalg.norm(row['wind_velocity']):.1f} m/s   |   "
              f"drag {np.linalg.norm(row.get('drag_force', row['wind_force'])):.1f} N   |   scale {wind_scale:g}",
              font=_font(19), fill=(173, 226, 236, 255))
    draw.rectangle((0, picture.height - 43, picture.width, picture.height), fill=(12, 25, 38, 230))
    caption = ("Surface loads on prescribed G1 gait | not coupled to LQR | approximate mesh drag, no fluid solver" if surface else
               "Reduced-model LQR + G1 inverse kinematics | not full-body walking dynamics"
               if robot == "g1" else "Reduced LIP dynamics | prescribed footsteps | legs are visualization | yellow: CoP")
    draw.text((22, picture.height - 34), caption,
              font=_font(18), fill=(232, 237, 242, 255))
    return np.asarray(picture)


def _set_record(env, gait, row):
    env.data.qpos[:] = row["state"][[0, 2]]
    env.data.qvel[:] = row["state"][[1, 3]]
    env.data.time = row["time"]
    env.update_visuals(gait.sample(row["time"]), row["cop"])


def _connector(scene, kind, start, end, width, color):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, kind, np.zeros(3), np.zeros(3), np.eye(3).reshape(-1), np.array(color, dtype=np.float32))
    mujoco.mjv_connector(geom, kind, width, np.array(start, dtype=float), np.array(end, dtype=float))
    scene.ngeom += 1


def _decorate(scene, row, wind_scale, wind_config):
    c = row["state"][[0, 2]]
    for dx in (-1.0, 0.0, 1.0):
        for dy in (-1.0, 1.0):
            p = c + [dx, dy]
            w = tornado_velocity(local_to_world(p, wind_config), wind_config, scale=wind_scale)
            end = p + 0.05 * w
            _connector(scene, mujoco.mjtGeom.mjGEOM_ARROW, [*p, .04], [*end, .04], .018, [.15, .8, .9, .7])
    polygon = row["support_polygon"]
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        _connector(scene, mujoco.mjtGeom.mjGEOM_CAPSULE, [*a, .012], [*b, .012], .008, [.95, .8, .2, 1])


def render_recording(env, gait, records, output, *, video=False, snapshot=False,
                     controller_label="LQR", robot="g1", force_overlay=True, drag_profile="surface"):
    if drag_profile not in ("surface", "lumped"):
        raise ValueError("drag_profile must be surface or lumped")
    if force_overlay and drag_profile == "surface" and robot != "g1":
        raise ValueError("surface drag requires the articulated G1 mesh")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    camera = mujoco.MjvCamera()
    camera.azimuth = 135
    camera.elevation = -24
    camera.distance = 2.8 if force_overlay else 3.6
    replay = None
    if robot == "g1":
        from .g1_replay import G1Replay
        replay = G1Replay(com_height=env.config.height)
    display_model = replay.model if replay is not None else env.model
    stride = 2
    sampled = records[::stride]
    profile_poses, profile_velocities = None, None
    if force_overlay and drag_profile == "surface":
        from .visualize.surface_profile import prepare_surface_profile
        print("Evaluating surface drag along the prescribed G1 gait...", flush=True)
        sampled, profile_poses, profile_velocities = prepare_surface_profile(
            replay, sampled, env.wind_config, env.config.wind_scale, output)
    forces = None
    if force_overlay:
        from .visualize import ForceVisualizer
        from .visualize.appearance import set_robot_opacity
        forces = ForceVisualizer(sampled, env.config)
    renderer = mujoco.Renderer(display_model, height=720, width=1280)
    writer = None
    replay_times, replay_poses = [], []
    try:
        if video:
            import imageio.v2 as imageio
            writer = imageio.get_writer(str(output / "walking.mp4"), fps=1 / (env.config.control_dt * stride),
                                        codec="libx264", quality=7, macro_block_size=16)
        selected = min(len(sampled) - 1, max(0, len(sampled) // 3))
        indices = range(len(sampled)) if video else [selected]
        saved_snapshot = False
        for i in indices:
            row = sampled[i]
            if replay is None:
                _set_record(env, gait, row)
                display_data = env.data
            else:
                if profile_poses is None:
                    replay.set_state(row["state"], row["feet"], row["time"], row["cop"])
                else:
                    replay.data.qpos[:] = profile_poses[i]
                    replay.data.qvel[:] = profile_velocities[i]
                    replay.data.time = row["time"]
                    mujoco.mj_forward(replay.model, replay.data)
                display_data = replay.data
                replay_times.append(row["time"])
                replay_poses.append(replay.data.qpos.copy())
            if forces is not None:
                # A drag-only view: omit CoP/reference markers and floor-speed arrows.
                for name in ("cop_marker", "reference_marker"):
                    body = display_model.body(name).id
                    display_data.mocap_pos[display_model.body_mocapid[body], 2] = -5
                mujoco.mj_forward(display_model, display_data)
            camera.lookat[:] = [row["state"][0] + .25, row["state"][2], .65]
            renderer.update_scene(display_data, camera=camera)
            if forces is not None:
                set_robot_opacity(renderer.scene, display_model, opacity=0.5)
                forces.decorate(renderer.scene, row)
            else:
                _decorate(renderer.scene, row, env.config.wind_scale, env.wind_config)
            frame = _annotate(renderer.render(), row, env.config.wind_scale, controller_label, robot)
            if forces is not None:
                frame = forces.compose(frame, row)
            if writer is not None:
                writer.append_data(frame)
            if snapshot and not saved_snapshot and i >= selected:
                Image.fromarray(frame).save(output / "simulation_preview.png")
                saved_snapshot = True
        print(f"Rendered media: {output.resolve()}")
        if forces is not None:
            metadata = forces.metadata()
            metadata["robot_opacity"] = 0.5
            metadata["rendered_samples"] = len(indices)
            metadata["recording_sample_convention"] = "forces and poses at control-interval start"
            (output / "force_visualization.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if replay is not None:
            (output / "g1_ik_diagnostics.json").write_text(json.dumps(replay.diagnostics(), indent=2), encoding="utf-8")
            np.savez_compressed(output / "g1_kinematic_poses.npz", time=np.asarray(replay_times),
                                qpos=np.asarray(replay_poses), kinematic_only=True)
    finally:
        if writer is not None:
            writer.close()
        renderer.close()


def replay_viewer(env, gait, records, *, robot="g1"):
    import mujoco.viewer
    replay = None
    if robot == "g1":
        from .g1_replay import G1Replay
        replay = G1Replay(com_height=env.config.height)
    model, data = (replay.model, replay.data) if replay is not None else (env.model, env.data)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25
        viewer.cam.distance = 4
        for row in records:
            if not viewer.is_running():
                break
            started = time.perf_counter()
            with viewer.lock():
                if replay is None:
                    _set_record(env, gait, row)
                else:
                    replay.set_state(row["state"], row["feet"], row["time"], row["cop"])
                viewer.cam.lookat[:] = [row["state"][0] + .2, row["state"][2], .6]
                viewer.user_scn.ngeom = 0
                _decorate(viewer.user_scn, row, env.config.wind_scale, env.wind_config)
            viewer.sync()
            time.sleep(max(0.0, env.config.control_dt - (time.perf_counter() - started)))
