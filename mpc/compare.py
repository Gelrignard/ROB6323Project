"""Compare three or four DIAL-MPC recordings on one simulated-time timeline.

Usage: python -m mpc.compare RUN/calm RUN/random RUN/tornado --output comparison.mp4
Four inputs use a2x2 grid in the supplied order. Three inputs stay horizontal.
Shorter runs freeze at their actual final frame with an explicit ENDED or
FAILED label. No clip is time-stretched or presented as continued motion.
"""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from .render import _font, controller_information_label, render_run


BACKGROUND = (12, 25, 38)
TEXT = (235, 243, 248)
MUTED = (153, 175, 192)
FAILURE = (255, 126, 97)


def comparison_layout(count):
    """Return columns, rows and a codec-aligned canvas for supported inputs."""
    if count == 3:
        return 3, 1, (2160, 800)
    if count == 4:
        return 2, 2, (1440, 1520)
    raise ValueError("Three or four run directories are required")


def default_tile_label(summary):
    if summary["scenario"] == "random":
        strength = (summary.get("disturbances") or {}).get("strength", 0.)
        return f"RANDOM IMPACTS (strength {strength:.2f})"
    return summary["scenario"].upper()


def recorded_pulses(states, events, time_s, end_s, body_ids, joint_dofs):
    """Current per-target load from recorded physics samples, never forecasts.

    When several active events share a target/type, the recorded vector is the
    aggregate actually applied to that target. It is displayed once. A completed
    or failed episode has no samples after its end, so future scheduled events
    are never shown as active on a held frame.
    """
    if time_s >= end_s - 1e-9 or "disturbance_sample_time_s" not in states:
        return []
    times = states["disturbance_sample_time_s"]
    index = int(np.searchsorted(times, time_s + 1e-9, side="right") - 1)
    if index < 0:
        return []
    sample_time = float(times[index])
    active = [event for event in events
              if event["start_s"] <= sample_time < event["start_s"] + event["duration_s"]]
    result, shown = [], set()
    for event in active:
        kind, target = event["kind"], event["target"]
        if (kind, target) in shown:
            continue
        shown.add((kind, target))
        key = {"link_force": "disturbance_body_forces_N",
               "link_torque": "disturbance_body_torques_Nm",
               "joint_torque": "disturbance_joint_torques_Nm"}[kind]
        target_index = joint_dofs.get(target) if kind == "joint_torque" else body_ids.get(target)
        value = None
        if key in states and target_index is not None:
            value = np.asarray(states[key][index, target_index], dtype=float)
        result.append({"kind": kind, "target": target,
                       "magnitude": float(np.linalg.norm(value)) if value is not None else None,
                       "unit": "N" if kind == "link_force" else "N m",
                       "sample_time_s": sample_time})
    return result


def pulse_caption(pulse):
    kind = {"link_force": "Force", "link_torque": "Link torque", "joint_torque": "Joint torque"}[pulse["kind"]]
    magnitude = (f"{pulse['magnitude']:.2f} {pulse['unit']}" if pulse["magnitude"] is not None
                 else "recorded magnitude unavailable")
    return f"{kind} on {pulse['target']}: {magnitude}"


class RecordedVideo:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.summary = json.loads((self.directory / "summary.json").read_text(encoding="utf-8"))
        video_path = self.directory / "walking.mp4"
        if not video_path.is_file():
            render_run(self.directory)
        self.reader = imageio.get_reader(str(video_path))
        self.states = np.load(self.directory / "trajectory.npz")
        self.times = self.states["time"]
        self.end = float(self.times[-1])
        self.events = (self.summary.get("disturbances") or {}).get("events", [])
        self.body_ids, self.joint_dofs = {}, {}
        if self.events:
            model = mujoco.MjModel.from_binary_path(str(self.directory / "model.mjb"))
            for event in self.events:
                target = event["target"]
                if event["kind"] == "joint_torque":
                    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, target)
                    if joint >= 0:
                        self.joint_dofs[target] = int(model.jnt_dofadr[joint])
                else:
                    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target)
                    if body >= 0:
                        self.body_ids[target] = body
        fps = float(self.reader.get_meta_data()["fps"])
        control_dt = float(self.summary["control_dt_s"])
        stride = max(1, int(round(1 / (fps * control_dt))))
        indices = list(range(0, len(self.times), stride))
        if indices[-1] != len(self.times) - 1:
            indices.append(len(self.times) - 1)
        self.frame_times = self.times[indices]
        self._last_frame_index = None
        self._last_frame = None

    def at(self, time_s):
        index = int(np.clip(np.searchsorted(self.frame_times, time_s + 1e-9, side="right") - 1,
                            0, len(self.frame_times) - 1))
        if index != self._last_frame_index:
            self._last_frame = self.reader.get_data(index)
            self._last_frame_index = index
        state_index = int(np.clip(np.searchsorted(self.times, time_s + 1e-9, side="right") - 1,
                                  0, len(self.times) - 1))
        ended = bool(time_s > self.end + 1e-9)
        at_end = bool(time_s >= self.end - 1e-9)
        actual_time = min(float(time_s), self.end)
        progress = float(self.states["qpos"][state_index, 0] - self.states["qpos"][0, 0])
        contact = self.states["foot_contact"][state_index]
        loads_t = self.states["wind_sample_time_s"]
        if len(loads_t):
            load_index = int(np.clip(np.searchsorted(loads_t, actual_time + 1e-9, side="right") - 1, 0, len(loads_t) - 1))
            drag = float(np.linalg.norm(self.states["wind_force_N"][load_index]))
        else:
            drag = 0.
        pulses = recorded_pulses(self.states, self.events, time_s, self.end, self.body_ids, self.joint_dofs)
        return self._last_frame, {"time": actual_time, "ended": ended, "at_end": at_end, "progress": progress,
                                  "contact": contact, "drag": drag, "pulses": pulses}

    def close(self):
        self.reader.close()
        self.states.close()


def _tile(source, time_s, label=None):
    frame, row = source.at(time_s)
    summary = source.summary
    tile = Image.new("RGB", (720, 720), BACKGROUND)
    draw = ImageDraw.Draw(tile)
    case = summary["scenario"]
    draw.text((20, 8), label or default_tile_label(summary), font=_font(25), fill=TEXT)
    if case == "tornado":
        exposure = f"Vortex wind speed scale {summary['wind_scale']:.2f}"
    elif case == "random":
        disturbance = summary.get("disturbances") or {}
        strength = disturbance.get("strength", 0.)
        config = disturbance.get("config") or {}
        exposure = (f"Caps: {strength * config.get('link_force_max_n', 0.):g} N force; "
                    f"{strength * config.get('link_torque_max_nm', 0.):g} N m link torque; "
                    f"{strength * config.get('joint_torque_max_nm', 0.):g} N m joint torque")
    else:
        exposure = "Calm air; no injected disturbances"
    draw.text((20, 44), exposure, font=_font(18), fill=MUTED)
    draw.text((20, 76), controller_information_label(summary), font=_font(17), fill=TEXT)
    # Our source render is a960px-wide scene plus a480px quantitative panel.
    # Crop its scene, remove its duplicate caption, preserve aspect ratio.
    scene = Image.fromarray(frame).crop((0, 72, min(frame.shape[1], 960), 720))
    scene = scene.resize((674, 455), Image.Resampling.LANCZOS)
    tile.paste(scene, (23, 102))
    draw = ImageDraw.Draw(tile)
    failure = summary.get("failure_reason")
    if row["ended"]:
        message = ("FAILED" if failure else "ENDED") + f" at {source.end:.2f} s - FINAL FRAME HELD"
        draw.rectangle((0, 112, 720, 166), fill=BACKGROUND)
        draw.text((360, 124), message, font=_font(23), fill=FAILURE if failure else TEXT, anchor="mt")
    draw.text((20, 574), f"t={row['time']:.2f} s   forward={row['progress']:+.3f} m   drag={row['drag']:.2f} N",
              font=_font(20), fill=TEXT)
    contacts = " / ".join("contact" if value else "airborne" for value in row["contact"])
    draw.text((20, 607), f"Feet L/R: {contacts}", font=_font(17), fill=MUTED)
    if row["pulses"]:
        for index, pulse in enumerate(row["pulses"][:2]):
            draw.text((20, 635 + index * 22), pulse_caption(pulse), font=_font(16), fill=(255, 206, 84))
        if len(row["pulses"]) > 2:
            draw.text((698, 614), f"+{len(row['pulses']) - 2} other loads", font=_font(14), fill=MUTED, anchor="rt")
    else:
        pulse_status = "Episode ended; no further load samples" if row["at_end"] else "No active injected pulse"
        draw.text((20, 644), pulse_status, font=_font(17), fill=MUTED)
    outcome = (f"FAILED at {source.end:.2f} s: {failure}" if failure else
               "Walking criterion passed" if summary.get("walking_success") else
               "Survived; walking criterion NOT passed")
    if not row["at_end"]:
        outcome = "Run in progress; outcome evaluated at episode end"
    draw.text((20, 687), outcome[:79], font=_font(15), fill=FAILURE if failure and row["at_end"] else MUTED)
    return tile


def compare_runs(directories, output, fps=25, labels=None):
    columns, rows, canvas_size = comparison_layout(len(directories))
    if labels is not None and (len(labels) != len(directories) or any(not str(label).strip() for label in labels)):
        raise ValueError("labels must provide one nonempty title per input run")
    output = Path(output)
    sidecar = output.with_suffix(".json")
    preview = output.with_suffix(".png")
    if any(path.exists() for path in (output, sidecar, preview)):
        raise FileExistsError("Refusing to overwrite comparison output or sidecars")
    if output.suffix.lower() != ".mp4" or not np.isfinite(fps) or fps <= 0:
        raise ValueError("output must be .mp4 and fps positive")
    sources, writer = [], None
    try:
        for directory in directories:
            sources.append(RecordedVideo(directory))
        labels = list(labels) if labels is not None else [default_tile_label(source.summary) for source in sources]
        duration = max(source.end for source in sources)
        count = int(np.ceil(duration * fps - 1e-9)) + 1
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(output), fps=fps, codec="libx264", quality=7,
                                   macro_block_size=16, ffmpeg_params=["-threads", "2"])
        for index in range(count):
            time_s = index / fps
            canvas = Image.new("RGB", canvas_size, BACKGROUND)
            draw = ImageDraw.Draw(canvas)
            draw.text((24, 12), f"G1 DIAL-MPC  |  {len(sources)} PHYSICAL ROLLOUTS", font=_font(28), fill=TEXT)
            draw.text((24, 48), "Arrows: air drag only, scaled per clip. Injected loads: recorded values in text.", font=_font(16), fill=MUTED)
            draw.text((canvas_size[0]-26, 22), f"SIM TIME {time_s:.2f} s   |   offline CPU planning", font=_font(21), fill=MUTED, anchor="rt")
            for position, source in enumerate(sources):
                row, column = divmod(position, columns)
                x, y = column * 720, 72 + row * 720
                canvas.paste(_tile(source, time_s, labels[position]), (x, y))
                if column:
                    draw.line((x, y, x, y+720), fill=(65, 85, 100), width=2)
                if row:
                    draw.line((x, y, x+720, y), fill=(65, 85, 100), width=2)
            writer.append_data(np.asarray(canvas))
            if index == min(count - 1, int(fps)):
                canvas.save(preview)
        writer.close()
        writer = None
        metadata = {"runs": [{"directory": str(source.directory.resolve()), "label": label,
                               "scenario": source.summary["scenario"], "duration_s": source.end,
                               "wind_scale": source.summary.get("wind_scale"),
                               "wind_field": source.summary.get("wind_field"),
                               "wind_config": source.summary.get("wind_config"),
                               "controller_metadata": source.summary.get("controller_metadata"),
                               "controller_config": source.summary.get("controller_config"),
                               "controller_information_label": controller_information_label(source.summary),
                               "disturbance_strength": (source.summary.get("disturbances") or {}).get("strength"),
                               "disturbances": source.summary.get("disturbances"),
                               "walking_success": source.summary.get("walking_success"),
                               "failure_reason": source.summary.get("failure_reason")}
                              for source, label in zip(sources, labels)],
                    "fps": fps, "frames": count, "video": str(output.resolve()),
                    "layout": {"columns": columns, "rows": rows, "input_order": "row-major", "pixels": canvas_size},
                    "timeline": "shared simulated seconds; no time stretching",
                    "shorter_runs": "hold actual final frame with explicit ENDED/FAILED banner",
                    "force_display": "air-drag vectors only, scaled independently per source clip; injected pulses captioned with target, actual recorded magnitude and unit",
                    "injected_load_sampling": "latest recorded physics sample at or before current time; no active-load caption at or after episode end; same-target overlapping loads show recorded aggregate"}
        sidecar.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps(metadata, indent=2), flush=True)
        return output
    finally:
        if writer is not None:
            writer.close()
        for source in sources:
            source.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", help="three runs horizontally or four runs in a2x2 grid, preserving input order")
    parser.add_argument("--labels", nargs="+", help="optional tile titles, one per input run")
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=float, default=25)
    args = parser.parse_args()
    compare_runs(args.directories, args.output, args.fps, args.labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
