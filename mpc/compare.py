"""Compare three recorded DIAL-MPC cases on one simulated-time timeline.

Usage: python -m mpc.compare RUN/calm RUN/random RUN/tornado --output comparison.mp4
Shorter runs freeze at their actual final frame with an explicit ENDED or
FAILED label. No clip is time-stretched or presented as continued motion.
"""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from .render import _font, controller_information_label, render_run


BACKGROUND = (12, 25, 38)
TEXT = (235, 243, 248)
MUTED = (153, 175, 192)
FAILURE = (255, 126, 97)


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
        index = int(np.clip(np.searchsorted(self.frame_times, time_s, side="right") - 1,
                            0, len(self.frame_times) - 1))
        if index != self._last_frame_index:
            self._last_frame = self.reader.get_data(index)
            self._last_frame_index = index
        state_index = int(np.clip(np.searchsorted(self.times, time_s, side="right") - 1,
                                  0, len(self.times) - 1))
        ended = bool(time_s > self.end + 1e-9)
        actual_time = min(float(time_s), self.end)
        progress = float(self.states["qpos"][state_index, 0] - self.states["qpos"][0, 0])
        contact = self.states["foot_contact"][state_index]
        loads_t = self.states["wind_sample_time_s"]
        if len(loads_t):
            load_index = int(np.clip(np.searchsorted(loads_t, actual_time, side="right") - 1, 0, len(loads_t) - 1))
            drag = float(np.linalg.norm(self.states["wind_force_N"][load_index]))
        else:
            drag = 0.
        active = [event["kind"] for event in self.events
                  if not ended and event["start_s"] <= actual_time < event["start_s"] + event["duration_s"]]
        return self._last_frame, {"time": actual_time, "ended": ended, "progress": progress,
                                  "contact": contact, "drag": drag, "active": active}

    def close(self):
        self.reader.close()
        self.states.close()


def _tile(source, time_s):
    frame, row = source.at(time_s)
    summary = source.summary
    tile = Image.new("RGB", (720, 720), BACKGROUND)
    draw = ImageDraw.Draw(tile)
    case = summary["scenario"]
    draw.text((20, 8), case.upper(), font=_font(28), fill=TEXT)
    if case == "tornado":
        exposure = f"Vortex wind speed scale {summary['wind_scale']:.2f}"
    elif case == "random":
        disturbance = summary.get("disturbances") or {}
        strength = disturbance.get("strength", 0.)
        config = disturbance.get("config") or {}
        exposure = (f"Random pulses: strength {strength:.2f}, "
                    f"force cap {strength * config.get('link_force_max_n', 0.):g} N")
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
        message = "FAILED - FINAL FRAME HELD" if failure else "ENDED - FINAL FRAME HELD"
        draw.rectangle((0, 112, 720, 166), fill=BACKGROUND)
        draw.text((360, 124), message, font=_font(27), fill=FAILURE if failure else TEXT, anchor="mt")
    draw.text((20, 574), f"t={row['time']:.2f} s   forward={row['progress']:+.3f} m   drag={row['drag']:.2f} N",
              font=_font(20), fill=TEXT)
    contacts = " / ".join("contact" if value else "airborne" for value in row["contact"])
    draw.text((20, 610), f"Feet L/R: {contacts}", font=_font(18), fill=MUTED)
    pulse = "PULSE ACTIVE: " + ", ".join(row["active"]) if row["active"] else "No active injected pulse"
    draw.text((20, 642), pulse[:69], font=_font(18), fill=(255, 206, 84) if row["active"] else MUTED)
    outcome = (f"FAILED: {failure}" if failure else
               "Walking criterion passed" if summary.get("walking_success") else
               "Survived; walking criterion NOT passed")
    draw.text((20, 678), outcome[:74], font=_font(17), fill=FAILURE if failure else MUTED)
    return tile


def compare_runs(directories, output, fps=25):
    if len(directories) != 3:
        raise ValueError("Exactly three run directories are required")
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
        duration = max(source.end for source in sources)
        count = int(np.ceil(duration * fps - 1e-9)) + 1
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(output), fps=fps, codec="libx264", quality=7,
                                   macro_block_size=16, ffmpeg_params=["-threads", "2"])
        for index in range(count):
            time_s = index / fps
            canvas = Image.new("RGB", (2160, 800), BACKGROUND)
            draw = ImageDraw.Draw(canvas)
            draw.text((24, 12), "G1 DIAL-MPC  |  THREE PHYSICAL ROLLOUTS", font=_font(30), fill=TEXT)
            draw.text((24, 48), "Drag vectors: visualization only", font=_font(16), fill=MUTED)
            draw.text((2134, 22), f"SIM TIME {time_s:.2f} s   |   offline CPU planning", font=_font(22), fill=MUTED, anchor="rt")
            for column, source in enumerate(sources):
                canvas.paste(_tile(source, time_s), (column * 720, 72))
                if column:
                    draw.line((column * 720, 72, column * 720, 792), fill=(65, 85, 100), width=2)
            writer.append_data(np.asarray(canvas))
            if index == min(count - 1, int(fps)):
                canvas.save(preview)
        writer.close()
        writer = None
        metadata = {"runs": [{"directory": str(source.directory.resolve()),
                               "scenario": source.summary["scenario"], "duration_s": source.end,
                               "wind_scale": source.summary.get("wind_scale"),
                               "wind_field": source.summary.get("wind_field"),
                               "wind_config": source.summary.get("wind_config"),
                               "controller_metadata": source.summary.get("controller_metadata"),
                               "controller_config": source.summary.get("controller_config"),
                               "controller_information_label": controller_information_label(source.summary),
                               "disturbance_strength": (source.summary.get("disturbances") or {}).get("strength"),
                               "walking_success": source.summary.get("walking_success"),
                               "failure_reason": source.summary.get("failure_reason")}
                              for source in sources],
                    "fps": fps, "frames": count, "video": str(output.resolve()),
                    "timeline": "shared simulated seconds; no time stretching",
                    "shorter_runs": "hold actual final frame with explicit ENDED/FAILED banner",
                    "force_display": "aerodynamic vectors only; random injected pulses separately captioned"}
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
    parser.add_argument("directories", nargs=3, help="three run directories, normally calm random tornado")
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=float, default=25)
    args = parser.parse_args()
    compare_runs(args.directories, args.output, args.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
