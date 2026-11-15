"""Drag-only spatial and temporal displays, without changing physics.

The original reduced model supplies one CoM drag resultant. A surface run adds
pressure samples and integrated drag computed separately on G1 geometry.
"""

from dataclasses import dataclass
from functools import lru_cache

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


DRAG = (255, 126, 97)
COMPONENT_COLORS = ((255, 183, 111), (106, 201, 255), (189, 158, 255))
TEXT = (235, 243, 248)
MUTED = (153, 175, 192)
BACKGROUND = (12, 25, 38)


@lru_cache(maxsize=16)
def _font(size):
    for path in ("C:/Windows/Fonts/segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _vector(value, length, name):
    value = np.asarray(value, dtype=float)
    if value.shape != (length,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be finite with shape ({length},)")
    return value


def drag_vector(row):
    """Return the recorded 3D aerodynamic load ON the robot, in newtons."""
    if "drag_force" in row:
        return _vector(row["drag_force"], 3, "drag_force")
    return np.array([*_vector(row["wind_force"], 2, "wind_force"), 0.0])


def surface_samples(row):
    """Validated quadrature locations, local forces and normal pressures."""
    surface = row.get("surface_drag")
    if surface is None:
        return None
    points = np.asarray(surface["points"], dtype=float)
    forces = np.asarray(surface["forces"], dtype=float)
    pressure = np.asarray(surface["pressure"], dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or forces.shape != points.shape or pressure.shape != points.shape[:1]:
        raise ValueError("surface points/forces must have shape (N,3), pressure (N,)")
    if not all(np.all(np.isfinite(value)) for value in (points, forces, pressure)) or np.any(pressure < 0):
        raise ValueError("surface samples must be finite, with nonnegative pressure")
    return points, forces, pressure


def _pressure_color(fraction):
    knots = np.array([[42, 110, 167], [62, 201, 211], [255, 206, 84], [255, 86, 68]], dtype=float)
    position = np.clip(fraction, 0.0, 1.0) * (len(knots) - 1)
    lower = min(int(position), len(knots) - 2)
    return tuple(np.rint(knots[lower] + (position - lower) * (knots[lower + 1] - knots[lower])).astype(int))


@dataclass(frozen=True)
class ForceArrow:
    name: str
    origin: np.ndarray
    force: np.ndarray
    metres_per_newton: float
    color: tuple = DRAG

    @property
    def endpoint(self):
        return self.origin + self.metres_per_newton * self.force


class ForceVisualizer:
    """Integrated and local drag vectors with a 480 px quantitative sidebar.

    The arrow scale and pressure-color range are fixed across the entire clip.
    Local vectors are display subsampled only; force plots use all samples.
    """

    panel_width = 480
    max_surface_vectors = 96
    local_max_display_length = .26
    local_min_display_length = .03

    def __init__(self, records, config):
        if not records:
            raise ValueError("drag visualization requires at least one record")
        self.config = config
        if not np.isfinite(config.height) or config.height <= 0:
            raise ValueError("height must be finite and positive")
        self.times = np.array([float(row["time"]) for row in records])
        if not np.all(np.isfinite(self.times)) or np.any(np.diff(self.times) <= 0):
            raise ValueError("record times must be finite and strictly increasing")
        self.history = np.array([drag_vector(row) for row in records])
        self.magnitudes = np.linalg.norm(self.history, axis=1)
        self.start = float(self.times[0])
        sample_dt = float(np.median(np.diff(self.times))) if len(self.times) > 1 else config.control_dt
        self.end = float(self.times[-1] + sample_dt)
        self.force_limit = self._axis_limit(np.max(np.abs(self.history)))
        self.magnitude_limit = self._axis_limit(np.max(self.magnitudes))
        self.arrow_scale = .85 / max(float(np.max(self.magnitudes)), 1.0)
        self.models = sorted({row.get("drag_model", "lumped") for row in records})
        if any(model not in ("lumped", "surface") for model in self.models):
            raise ValueError("drag_model must be lumped or surface")
        surfaces = [surface_samples(row) for row in records]
        available = [sample for sample in surfaces if sample is not None and len(sample[0])]
        self.pressure_limit = max(1.0, max((float(np.max(s[2])) for s in available), default=0.0))
        maximum_height = max((float(np.max(s[0][:, 2])) for s in available), default=1.4)
        minimum_height = min((float(np.min(s[0][:, 2])) for s in available), default=0.0)
        lower_height = min(0.0, np.floor(minimum_height / .2) * .2)
        self.height_edges = np.linspace(lower_height, max(lower_height + .2, np.ceil(maximum_height / .2) * .2), 8)
        self.profile_limit = max(1.0, max((float(np.max(self._height_profile(s))) for s in available), default=0.0))
        self.surface_indices = np.array([], dtype=int)
        self.surface_sample_count = 0
        self.local_arrow_scale = 1.0
        if available:
            self.surface_sample_count = len(available[0][0])
            if any(len(sample[0]) != self.surface_sample_count for sample in available):
                raise ValueError("surface sample count must stay constant across a clip")
            peak_local_forces = np.max(np.stack([np.linalg.norm(sample[1], axis=1) for sample in available]), axis=0)
            peak_force = float(np.max(peak_local_forces))
            self.local_arrow_scale = self.local_max_display_length / peak_force if peak_force > 1e-12 else 1.0
            self.surface_indices = self._select_surface_vectors(available[0][0], peak_local_forces)

    def _select_surface_vectors(self, positions, peak_forces):
        """Stable IDs with spatial coverage, biased toward visible local loads.

        Selection is made once from first-frame locations and the maximum load
        of each sample over the clip, so an arrow does not jump between nearby
        surface samples as the robot moves. All samples still enter integrals.
        """
        candidates = np.flatnonzero(peak_forces * self.local_arrow_scale >= self.local_min_display_length)
        if len(candidates) <= self.max_surface_vectors:
            return candidates
        points = positions[candidates]
        weights = .25 + .75 * peak_forces[candidates] / np.max(peak_forces[candidates])
        selected = []
        nearest_distance_squared = np.full(len(candidates), np.inf)
        index = int(np.argmax(peak_forces[candidates]))
        for _ in range(self.max_surface_vectors):
            selected.append(index)
            squared_distance = np.sum((points - points[index]) ** 2, axis=1)
            nearest_distance_squared = np.minimum(nearest_distance_squared, squared_distance)
            scores = nearest_distance_squared * weights
            scores[selected] = -1.0
            index = int(np.argmax(scores))
        return candidates[np.array(selected)]

    @staticmethod
    def _axis_limit(peak):
        peak = max(1.0, float(peak))
        increment = 10 ** np.floor(np.log10(peak))
        return float(np.ceil(peak * 1.12 / increment) * increment)

    def _height_profile(self, surface):
        return np.histogram(surface[0][:, 2], bins=self.height_edges,
                            weights=np.linalg.norm(surface[1], axis=1))[0]

    def arrow_specs(self, row):
        if "drag_origin" in row:
            origin = _vector(row["drag_origin"], 3, "drag_origin")
        else:
            state = _vector(row["state"], 4, "state")
            origin = np.array([state[0], state[2], self.config.height])
        return [ForceArrow("drag", origin, drag_vector(row), self.arrow_scale)]

    def local_arrow_specs(self, row):
        """Local force arrows point INTO the surface; heads mark application.

        Tail = sample position - scale*dF, head = sample position. Each arrow
        has its actual force direction and length linear in the sample force;
        negligible lengths are omitted, never stretched to a minimum length.
        """
        samples = surface_samples(row)
        if samples is None:
            return []
        points, forces, pressure = samples
        if len(points) != self.surface_sample_count:
            raise ValueError("surface sample count differs from visualization history")
        result = []
        for index in self.surface_indices:
            length = self.local_arrow_scale * np.linalg.norm(forces[index])
            if length < self.local_min_display_length:
                continue
            origin = points[index] - self.local_arrow_scale * forces[index]
            color = _pressure_color(pressure[index] / self.pressure_limit)
            result.append(ForceArrow(f"surface_drag_{index}", origin, forces[index], self.local_arrow_scale, color))
        return result

    @staticmethod
    def _draw_arrow(scene, arrow, width):
        if np.linalg.norm(arrow.force) <= 1e-10 or scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        rgba = np.array([*(np.asarray(arrow.color) / 255), 1.0], dtype=np.float32)
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3),
                           np.zeros(3), np.eye(3).ravel(), rgba)
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, width,
                            arrow.origin, arrow.endpoint)
        scene.ngeom += 1

    def decorate(self, scene, row):
        """Append only drag-related geometry to the render/viewer scene."""
        self._draw_arrow(scene, self.arrow_specs(row)[0], .022)
        for arrow in self.local_arrow_specs(row):
            self._draw_arrow(scene, arrow, .009)

    def _chart(self, draw, title, top, values, limit, current_time, signed=True):
        left, right, y_top, bottom = 65, 456, top + 28, top + 94
        draw.text((24, top), title, font=_font(17), fill=TEXT)
        if signed:
            for x, label, color in zip((342, 373, 404), ("x", "y", "z"), COMPONENT_COLORS):
                draw.text((x, top), label, font=_font(15), fill=color)
        draw.text((432, top), "[N]", font=_font(13), fill=MUTED)
        lower = -limit if signed else 0.0
        ticks = (lower, 0, limit) if signed else (0, limit / 2, limit)
        for value in ticks:
            y = bottom - (value - lower) / (limit - lower) * (bottom - y_top)
            draw.line((left, y, right, y), fill=(45, 65, 80), width=1)
            draw.text((left - 8, y), f"{value:g}", font=_font(12), fill=MUTED, anchor="rm")
        xs = left + (self.times - self.start) / (self.end - self.start) * (right - left)
        values = np.asarray(values).reshape(len(self.times), -1)
        ys = bottom - (values - lower) / (limit - lower) * (bottom - y_top)
        elapsed = self.times <= current_time + 1e-9
        colors = COMPONENT_COLORS if signed else (DRAG,)
        for column, color in enumerate(colors):
            points = list(zip(xs.tolist(), ys[:, column].tolist()))
            faint = tuple(int(.35 * channel) for channel in color)
            if len(points) > 1:
                draw.line(points, fill=faint, width=1)
            passed = [point for point, present in zip(points, elapsed) if present]
            if len(passed) > 1:
                draw.line(passed, fill=color, width=2)
            if passed:
                x, y = passed[-1]
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
        cursor = left + np.clip((current_time - self.start) / (self.end - self.start), 0, 1) * (right - left)
        draw.line((cursor, y_top, cursor, bottom), fill=(225, 237, 243), width=1)
        if not signed:
            for fraction in (0, .5, 1):
                draw.text((left + fraction * (right - left), bottom + 8),
                          f"{self.start + fraction * (self.end - self.start):g}",
                          font=_font(12), fill=MUTED, anchor="mt")
            draw.text((255, bottom + 21), "time [s]", font=_font(12), fill=MUTED, anchor="mt")

    def _spatial_profile(self, draw, row):
        samples = surface_samples(row)
        if samples is None:
            draw.rounded_rectangle((18, 505, 462, 635), radius=8, fill=(27, 46, 62))
            draw.text((30, 517), "Lumped model", font=_font(19), fill=DRAG)
            draw.text((30, 550), "One aerodynamic resultant at CoM.", font=_font(16), fill=TEXT)
            draw.text((30, 576), "No per-surface pressure or force", font=_font(16), fill=MUTED)
            draw.text((30, 600), "distribution is computed in this mode.", font=_font(16), fill=MUTED)
            return
        draw.text((24, 505), "Surface load by height", font=_font(17), fill=TEXT)
        draw.text((24, 530), "Sum of local |dF| per band [N]", font=_font(14), fill=MUTED)
        profile = self._height_profile(samples)
        for index in range(len(profile)):
            y = 553 + (len(profile) - 1 - index) * 12
            center = (self.height_edges[index] + self.height_edges[index + 1]) / 2
            draw.text((75, y + 5), f"{center:.1f} m", font=_font(11), fill=MUTED, anchor="rm")
            length = 310 * profile[index] / self.profile_limit
            draw.rectangle((88, y, 88 + max(.1, length), y + 8), fill=DRAG)
            draw.text((min(430, 96 + length), y + 4), f"{profile[index]:.2f}", font=_font(11), fill=MUTED, anchor="lm")

    def compose(self, frame, row):
        """Attach a drag-only sidebar synchronized to this logged record."""
        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8 or frame.shape[0] != 720:
            raise ValueError("expected an RGB uint8 frame with height 720")
        current_time = float(row["time"])
        if not np.isfinite(current_time):
            raise ValueError("time must be finite")
        force = drag_vector(row)
        is_surface = row.get("drag_model", "lumped") == "surface"
        panel = Image.new("RGB", (self.panel_width, 720), BACKGROUND)
        draw = ImageDraw.Draw(panel)
        draw.line((0, 0, 0, 720), fill=(71, 96, 115), width=2)
        draw.text((24, 18), "DRAG ON ROBOT", font=_font(25), fill=TEXT)
        draw.text((456, 24), f"{current_time:.2f} s", font=_font(18), fill=MUTED, anchor="rt")
        draw.text((24, 56), "World x / y / z  |  force in newtons", font=_font(16), fill=MUTED)
        draw.rounded_rectangle((18, 87, 462, 149), radius=8, fill=(27, 46, 62))
        draw.text((30, 93), "Surface pressure integration" if is_surface else "Single lumped CoM resultant",
                  font=_font(18), fill=DRAG)
        draw.text((30, 121), "Prescribed G1 gait; not coupled to LQR" if is_surface else "No distributed surface pressure model",
                  font=_font(15), fill=MUTED)
        draw.text((24, 167), f"|Fdrag| = {np.linalg.norm(force):.2f} N", font=_font(23), fill=DRAG)
        draw.text((24, 202), f"Fx {force[0]:+.2f}    Fy {force[1]:+.2f}    Fz {force[2]:+.2f}",
                  font=_font(17), fill=TEXT)
        draw.text((24, 232), f"Total arrow at CoM: {self.arrow_scale:.3g} m/N", font=_font(14), fill=MUTED)
        if is_surface:
            draw.text((24, 251), f"Local: {self.local_arrow_scale:.3g} m/N; arrowheads at surface", font=_font(13), fill=MUTED)
        self._chart(draw, "Drag components", 269, self.history, self.force_limit, current_time)
        self._chart(draw, "Resultant magnitude", 377, self.magnitudes, self.magnitude_limit, current_time, signed=False)
        self._spatial_profile(draw, row)
        if is_surface:
            draw.text((24, 643), "Local vector color: normal pressure [Pa]", font=_font(14), fill=MUTED)
            for x in range(24, 360):
                draw.line((x, 666, x, 676), fill=_pressure_color((x - 24) / 335))
            draw.text((24, 680), "0", font=_font(11), fill=MUTED)
            draw.text((360, 680), f"{self.pressure_limit:.1f}", font=_font(11), fill=MUTED, anchor="rt")
            draw.text((24, 701), "Windward pressure only; no CFD or shielding.", font=_font(12), fill=MUTED)
        else:
            draw.text((24, 658), "Drag uses local relative air velocity.", font=_font(15), fill=MUTED)
            draw.text((24, 685), "Arrow is a force vector; no other loads shown.", font=_font(14), fill=MUTED)
        return np.concatenate((frame, np.asarray(panel)), axis=1)

    def metadata(self):
        return {"quantity": "aerodynamic drag only, force ON robot in world axes, N",
                "drag_models": self.models,
                "sample_time": "synchronized to each displayed state",
                "arrow_origin": "drag_origin from record, otherwise reduced whole-robot CoM; surface resultant is illustrative, not a single physical application point",
                "arrow_metres_per_newton": self.arrow_scale,
                "arrow_scaling": "one isotropic scale fixed over the clip",
                "pressure_color_max_Pa": self.pressure_limit,
                "pressure_colors": "normal pressure on display-subsampled local drag arrows; fixed range for entire clip",
                "local_arrow_metres_per_newton": self.local_arrow_scale,
                "local_arrow_geometry": "head=surface sample, tail=sample-scale*dF; direction equals force ON robot",
                "local_arrow_min_display_length_m": self.local_min_display_length,
                "local_arrow_scaling": "linear in local force magnitude with one clipwide scale; arrows below the display-length threshold are omitted, never normalized or stretched",
                "surface_vector_limit": self.max_surface_vectors,
                "surface_sample_count": self.surface_sample_count,
                "surface_vector_sampling": "stable sample IDs selected once by load-weighted spatial coverage; only drawing is subsampled; all surface samples remain in force and torque integration",
                "height_profile": "sum of local force magnitudes in each height band; includes all samples; not equal to magnitude of integrated force when directions differ",
                "surface_scope": "evaluated on prescribed G1 kinematics; does not prove coupled walking stability; no CFD, wake or shielding model",
                "chart_convention": "signed x/y/z drag and resultant magnitude; cursor=current time; dim trace=future logged values",
                "panel_width_px": self.panel_width,
                "sample_count": len(self.times)}
