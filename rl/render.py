"""Render an actual physical G1 episode with aerodynamic drag vectors.

Example: python -m rl.render --stand --scenario tornado --duration 2 --output rl/evaluations/tornado.mp4
Compatible PPO bundles may replace --stand with --checkpoint PATH. Saved
states come from free-base MuJoCo integration, never a prescribed IK gait.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sim_env.rl_config import RLConfig
from sim_env.rl_env import G1RobustEnv
from sim_env.visualize.appearance import set_robot_opacity
from sim_env.visualize.forces import ForceVisualizer
from .contracts import environment_contract, load_bundle
from .evaluate import SCENARIOS, controller_reset_info, json_value, scenario_config
from .policies import PPOController, StandController, checked_action


def _font(size):
    for path in ("C:/Windows/Fonts/segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _record(env):
    """Evaluate the same drag law at this exact displayed physical state."""
    physics = env.physics
    row = {**env.get_state(), "ctrl": env.data.ctrl.copy(), "drag_model": env.config.drag_model,
           "drag_origin": env.data.subtree_com[env.root_body_id].copy()}
    if physics.surface_drag is not None:
        result = physics.surface_drag.evaluate(env.data, wind_scale=physics.wind_scale,
                                               wind_field=physics.wind_field)
        row.update(drag_force=result.total_force.copy(), drag_torque=result.total_torque_com.copy(),
                   drag_origin=result.com.copy(), surface_drag={"points": result.points.copy(),
                   "forces": result.patch_forces.copy(), "pressure": result.pressure_Pa.copy()})
    else:
        # Re-evaluation changes only applied loads; the next physical step
        # replaces them before adding its separately scheduled perturbations.
        physics._apply_wind()
        row.update(drag_force=physics.last_wind_force.copy(), drag_torque=physics.last_wind_torque_com.copy())
    return row


def _compose(frame, records, index, controller_label, scenario, arrows, failure):
    """Use fresh physical-rollout captions, not the legacy IK sidebar."""
    row = records[index]
    panel = Image.new("RGB", (480, 720), (12, 25, 38))
    draw = ImageDraw.Draw(panel)
    text, muted = (235, 243, 248), (153, 175, 192)
    force = row["drag_force"]
    draw.text((24, 20), "AERODYNAMIC DRAG ONLY", font=_font(24), fill=text)
    draw.text((24, 64), f"{scenario}  |  t = {row['time']:.2f} s", font=_font(20), fill=muted)
    draw.text((24, 104), "Fully coupled MuJoCo dynamics", font=_font(20), fill=text)
    draw.text((24, 136), f"Drag model: {row['drag_model']}", font=_font(18), fill=muted)
    draw.text((24, 180), f"|Fdrag| = {np.linalg.norm(force):.2f} N", font=_font(27), fill=(255, 126, 97))
    draw.text((24, 224), f"Fx {force[0]:+.2f}   Fy {force[1]:+.2f}   Fz {force[2]:+.2f}", font=_font(19), fill=text)
    draw.text((24, 258), "World axes; force ON robot", font=_font(17), fill=muted)
    times = np.array([record["time"] for record in records])
    magnitudes = np.linalg.norm([record["drag_force"] for record in records], axis=1)
    left, right, top, bottom = 60, 450, 336, 498
    max_force = max(1.0, float(np.max(magnitudes)) * 1.1)
    max_time = max(float(times[-1]), 1e-9)
    draw.text((24, 300), "Drag magnitude over time [N]", font=_font(19), fill=text)
    for fraction in (0., .5, 1.):
        y = bottom - fraction * (bottom - top)
        draw.line((left, y, right, y), fill=(45, 65, 80))
        draw.text((left - 8, y), f"{fraction * max_force:.1f}", font=_font(13), fill=muted, anchor="rm")
    points = [(left + t / max_time * (right - left), bottom - f / max_force * (bottom - top))
              for t, f in zip(times[:index + 1], magnitudes[:index + 1])]
    if len(points) > 1:
        draw.line(points, fill=(255, 126, 97), width=3)
    draw.text((left, bottom + 12), "0 s", font=_font(14), fill=muted)
    draw.text((right, bottom + 12), f"{max_time:.2f} s", font=_font(14), fill=muted, anchor="rt")
    if arrows is not None:
        draw.text((24, 550), f"Total arrow: {arrows.arrow_scale:.3g} m/N", font=_font(17), fill=text)
        if row["drag_model"] == "surface":
            draw.text((24, 580), f"Local arrows: {arrows.local_arrow_scale:.3g} m/N", font=_font(17), fill=text)
    draw.text((24, 622), "Robot opacity: 50%; vectors stay opaque", font=_font(16), fill=muted)
    draw.text((24, 651), "Injected joint/link loads are not drag", font=_font(16), fill=muted)
    note = "Windward pressure approximation; no CFD" if row["drag_model"] == "surface" else "Vectors use one fixed scale throughout clip"
    draw.text((24, 680), note, font=_font(15), fill=muted)
    scene = Image.fromarray(frame)
    scene_draw = ImageDraw.Draw(scene)
    scene_draw.rectangle((0, 0, scene.width, 76), fill=(12, 25, 38))
    scene_draw.text((20, 10), controller_label, font=_font(22), fill=text)
    status = f"Rollout end: {failure}" if failure else "Physical controller rollout; contacts and free-base dynamics"
    scene_draw.text((20, 43), status, font=_font(17), fill=muted)
    return np.concatenate((np.asarray(scene), np.asarray(panel)), axis=1)


def render(args):
    output = Path(args.output).resolve()
    if output.suffix.lower() != ".mp4":
        raise ValueError("output must have an .mp4 extension")
    metadata_path, states_path = output.with_suffix(".json"), output.with_suffix(".npz")
    for path in (output, metadata_path, states_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    base_config = RLConfig.from_json(args.config) if args.config else None
    if args.checkpoint and base_config is None:
        bundle = load_bundle(args.checkpoint)
        base_config = RLConfig.from_dict(bundle.manifest["environment_config"])
    config = scenario_config(args.scenario, args.duration, base_config=base_config)
    env = G1RobustEnv(config)
    controller = renderer = writer = None
    try:
        controller = PPOController(env, args.checkpoint, args.device) if args.checkpoint else StandController(env)
        label = f"PPO: {controller.name}" if args.checkpoint else "STAND DIAGNOSTIC (not a walking policy)"
        observation, info = env.reset(seed=args.seed)
        initial_info = info
        controller.reset(seed=args.seed, info=controller_reset_info(info))
        records = [_record(env)]
        terminated = truncated = False
        while not (terminated or truncated):
            action = checked_action(controller.act(observation), env)
            observation, _, terminated, truncated, info = env.step(action)
            if not info["physical_state_valid"] or env.data.time <= records[-1]["time"]:
                break
            records.append(_record(env))
        arrows = None if config.drag_model == "none" else ForceVisualizer(
            records, SimpleNamespace(height=.7, control_dt=config.control_dt))
        camera = mujoco.MjvCamera()
        camera.azimuth, camera.elevation, camera.distance = 135, -20, 2.8
        display_data = mujoco.MjData(env.model)
        renderer = mujoco.Renderer(env.model, height=720, width=960)
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(output), fps=1 / config.control_dt,
                                   codec="libx264", quality=7, macro_block_size=16)
        for index, row in enumerate(records):
            display_data.qpos[:], display_data.qvel[:], display_data.ctrl[:] = row["qpos"], row["qvel"], row["ctrl"]
            display_data.time = row["time"]
            mujoco.mj_forward(env.model, display_data)
            camera.lookat[:] = [*row["qpos"][:2], .65]
            renderer.update_scene(display_data, camera=camera)
            set_robot_opacity(renderer.scene, env.model, opacity=.5)
            if arrows is not None:
                arrows.decorate(renderer.scene, row)
            writer.append_data(_compose(renderer.render(), records, index, label,
                                        args.scenario, arrows, info.get("failure_reason")))
        writer.close()
        writer = None
        np.savez_compressed(states_path, time=[r["time"] for r in records],
                            qpos=[r["qpos"] for r in records], qvel=[r["qvel"] for r in records],
                            ctrl=[r["ctrl"] for r in records], aerodynamic_force_N=[r["drag_force"] for r in records],
                            aerodynamic_torque_about_com_Nm=[r["drag_torque"] for r in records])
        metadata = {"controller": label, "scenario": args.scenario, "seed": args.seed,
                    "environment_config": config.to_dict(), "contract": environment_contract(env),
                    "scope": "Recorded actual free-base MuJoCo controller rollout; no IK or prescribed gait",
                    "drag_sampling": "Same aerodynamic model re-evaluated at displayed physical qpos/qvel/time",
                    "force_display": "Aerodynamic loads only; injected joint/link disturbances excluded",
                    "surface_model": env.physics.surface_drag.metadata() if env.physics.surface_drag else None,
                    "opacity": .5, "frames": len(records), "fps": 1 / config.control_dt,
                    "physical_duration_s": float(info["elapsed_s"]),
                    "video_duration_s": len(records) * config.control_dt,
                    "video_sampling": "One frame per control time; includes initial and final physical state",
                    "terminated": terminated, "truncated": truncated, "failure_reason": info.get("failure_reason"),
                    "initialization": initial_info, "final_metrics": info["metrics"]}
        metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False, default=json_value), encoding="utf-8")
        print(json.dumps({"video": str(output), "metadata": str(metadata_path), "physical_states": str(states_path),
                          "failure_reason": info.get("failure_reason"), "frames": len(records)}, indent=2))
        return output
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
        if controller is not None:
            controller.close()
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument("--stand", action="store_true")
    parser.add_argument("--config", help="optional common evaluation RLConfig JSON")
    parser.add_argument("--scenario", choices=SCENARIOS, default="tornado")
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    render(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
