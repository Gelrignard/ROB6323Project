"""Render saved physical DIAL-MPC trajectories with drag-only force vectors.

State assignment happens only in this offline renderer, never in control.
Each run retains its own outcome and actual duration, including failures.
"""

import argparse
from dataclasses import fields
from functools import lru_cache
import json
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sim_env.config import TornadoConfig
from sim_env.surface_drag import SurfaceDrag
from sim_env.visualize.appearance import set_robot_opacity
from sim_env.visualize.forces import ForceVisualizer
from sim_env.wind_fields import WindFieldConfig, make_wind_field


@lru_cache(maxsize=16)
def _font(size):
    for path in ("C:/Windows/Fonts/segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _panel(frame, row, records, index, summary, arrows):
    background, text, muted, coral = (12, 25, 38), (235, 243, 248), (153, 175, 192), (255, 126, 97)
    panel = Image.new("RGB", (480, 720), background)
    draw = ImageDraw.Draw(panel)
    force = row["drag_force"]
    draw.text((24, 18), summary.get("controller", "DIAL-MPC / G1")[:29], font=_font(24), fill=text)
    draw.text((24, 60), f"{summary['scenario'].upper()}   t = {row['time']:.2f} s", font=_font(23), fill=muted)
    draw.text((24, 104), f"Forward travel: {row['progress']:+.3f} m", font=_font(22), fill=text)
    draw.text((24, 139), f"Command: {summary['command_speed_mps']:.2f} m/s", font=_font(18), fill=muted)
    for x, foot, contact in zip((24, 240), ("LEFT FOOT", "RIGHT FOOT"), row["contact"]):
        draw.text((x, 178), foot, font=_font(16), fill=muted)
        draw.text((x, 201), "CONTACT" if contact else "AIRBORNE", font=_font(20),
                  fill=(102, 225, 157) if contact else (255, 206, 84))
    draw.text((24, 251), "AERODYNAMIC DRAG ONLY", font=_font(21), fill=text)
    draw.text((24, 289), f"|F| = {np.linalg.norm(force):.2f} N", font=_font(27), fill=coral)
    draw.text((24, 332), f"Fx {force[0]:+.2f}   Fy {force[1]:+.2f}   Fz {force[2]:+.2f}",
              font=_font(18), fill=text)
    times = np.array([r["time"] for r in records])
    magnitudes = np.linalg.norm([r["drag_force"] for r in records], axis=1)
    left, right, top, bottom = 58, 449, 386, 493
    max_force = max(1., float(np.max(magnitudes)) * 1.1)
    max_time = max(float(times[-1]), 1e-9)
    for fraction in (0., .5, 1.):
        y = bottom - fraction * (bottom - top)
        draw.line((left, y, right, y), fill=(45, 65, 80))
        draw.text((left - 8, y), f"{fraction*max_force:.1f}", font=_font(13), fill=muted, anchor="rm")
    points = [(left + t/max_time*(right-left), bottom-f/max_force*(bottom-top))
              for t, f in zip(times[:index+1], magnitudes[:index+1])]
    if len(points) > 1:
        draw.line(points, fill=coral, width=3)
    draw.text((left, bottom+8), "0 s", font=_font(13), fill=muted)
    draw.text((right, bottom+8), f"{max_time:.2f} s", font=_font(13), fill=muted, anchor="rt")
    if row["active_events"]:
        draw.text((24, 546), "RANDOM LOAD ACTIVE", font=_font(22), fill=(255, 206, 84))
        draw.text((24, 579), ", ".join(row["active_events"])[:47], font=_font(15), fill=muted)
    else:
        draw.text((24, 550), "No injected pulse at this instant", font=_font(17), fill=muted)
    draw.text((24, 616), "Robot opacity 50%; drag vectors opaque", font=_font(16), fill=muted)
    draw.text((24, 644), f"Resultant arrow scale {arrows.arrow_scale:.3g} m/N", font=_font(15), fill=muted)
    draw.text((24, 675), "Synchronous CPU planning; video in sim time", font=_font(15), fill=muted)
    scene = Image.fromarray(frame)
    scene_draw = ImageDraw.Draw(scene)
    scene_draw.rectangle((0, 0, scene.width, 72), fill=background)
    scene_draw.text((20, 9), f"FULL PHYSICAL ROLLOUT  |  {summary.get('controller', 'DIAL-MPC G1 port')}"[:80],
                    font=_font(22), fill=text)
    failure = summary.get("failure_reason")
    status = f"Run outcome: FAILED ({failure})" if failure else (
        "Run outcome: walking criterion passed" if summary.get("walking_success") else
        "Run outcome: survived; walking criterion NOT passed")
    scene_draw.text((20, 43), status[:104], font=_font(17), fill=coral if failure else muted)
    return np.concatenate((np.asarray(scene), np.asarray(panel)), axis=1)


def render_run(directory, output=None, fps=25):
    directory = Path(directory)
    output = directory / "walking.mp4" if output is None else Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if output.suffix.lower() != ".mp4" or not np.isfinite(fps) or fps <= 0:
        raise ValueError("output must be .mp4 and fps positive")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    trajectory = np.load(directory / "trajectory.npz")
    model = mujoco.MjModel.from_binary_path(str(directory / "model.mjb"))
    data = mujoco.MjData(model)
    provider_fields = {field.name for field in fields(WindFieldConfig)}
    wind_spec = {key: value for key, value in (summary.get("wind_field") or {"kind": "calm"}).items()
                 if key in provider_fields}
    provider = make_wind_field(wind_spec)
    surface_meta = summary.get("surface_model") or {}
    drag = SurfaceDrag(model, wind_config=TornadoConfig(**summary["wind_config"]),
                       max_patches=surface_meta.get("quadrature_patches", 4096))
    if summary.get("drag_model") != "surface":
        raise ValueError("This renderer currently expects the surface-drag demo model")
    control_dt = float(summary["control_dt_s"])
    stride = max(1, int(round(1 / (fps * control_dt))))
    indices = list(range(0, len(trajectory["time"]), stride))
    if indices[-1] != len(trajectory["time"]) - 1:
        indices.append(len(trajectory["time"]) - 1)
    # Constant-fps playback follows the saved sampling stride. Final endpoint
    # inclusion can extend video duration by at most one rendered frame.
    effective_fps = 1 / (stride * control_dt)
    records = []
    events = (summary.get("disturbances") or {}).get("events", [])
    for i in indices:
        data.qpos[:], data.qvel[:], data.ctrl[:] = trajectory["qpos"][i], trajectory["qvel"][i], trajectory["ctrl"][i]
        data.time = float(trajectory["time"][i])
        mujoco.mj_forward(model, data)
        result = drag.evaluate(data, wind_scale=summary["wind_scale"], wind_field=provider)
        records.append({"index": i, "time": data.time, "drag_model": "surface",
                        "drag_origin": result.com.copy(), "drag_force": result.total_force.copy(),
                        "surface_drag": {"points": result.points.copy(), "forces": result.patch_forces.copy(),
                                         "pressure": result.pressure_Pa.copy()},
                        "progress": float(data.qpos[0] - trajectory["qpos"][0, 0]),
                        "contact": trajectory["foot_contact"][i],
                        "active_events": [event["kind"] for event in events
                                          if event["start_s"] <= data.time < event["start_s"] + event["duration_s"]]})
    arrows = ForceVisualizer(records, SimpleNamespace(height=.7, control_dt=1/effective_fps))
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation, camera.distance = 135, -20, 3.1
    renderer = mujoco.Renderer(model, height=720, width=960)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        writer = imageio.get_writer(str(output), fps=effective_fps, codec="libx264", quality=7,
                                   macro_block_size=16, ffmpeg_params=["-threads", "2"])
        for index, row in enumerate(records):
            i = row["index"]
            data.qpos[:], data.qvel[:], data.ctrl[:] = trajectory["qpos"][i], trajectory["qvel"][i], trajectory["ctrl"][i]
            data.time = row["time"]
            mujoco.mj_forward(model, data)
            camera.lookat[:] = [*data.qpos[:2], .65]
            renderer.update_scene(data, camera=camera)
            set_robot_opacity(renderer.scene, model, opacity=.5)
            arrows.decorate(renderer.scene, row)
            frame = _panel(renderer.render(), row, records, index, summary, arrows)
            writer.append_data(frame)
            if index == min(len(records)-1, int(effective_fps)):
                Image.fromarray(frame).save(output.with_suffix(".png"))
    finally:
        if writer is not None:
            writer.close()
        renderer.close()
        trajectory.close()
    info = {"source": str(directory.resolve()), "video": str(output.resolve()),
            "frames": len(records), "fps": effective_fps, "robot_opacity": .5,
            "drag_model": "same surface quadrature re-evaluated at saved physical states",
            "scope": "actual integrated trajectory; state replay occurs only for rendering",
            "force_arrows": "aerodynamic drag only; injected pulses are separately captioned",
            "failure_reason": summary.get("failure_reason"), "walking_success": summary.get("walking_success")}
    output.with_suffix(".render.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="run directory containing summary.json, trajectory.npz and model.mjb")
    parser.add_argument("--output")
    parser.add_argument("--fps", type=float, default=25)
    args = parser.parse_args()
    render_run(args.directory, args.output, args.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
