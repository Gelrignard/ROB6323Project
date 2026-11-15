"""Prescribed footsteps and a bounded reference for a reduced LIP walker.

This module plans kinematics, not joint torques or physical contact dynamics.
The reference CoM is the bounded solution of c'' = (g/h) (c - z_ref).
It is constructed by a two-sided exponential filter of the planned CoP.
Thus it includes the anticipatory lateral sway needed before a support switch.
Foot positions are sole centers; their stance height is exactly zero.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class GaitConfig:
    start_x: float = 1.0
    start_y: float = 5.0
    forward_speed: float = 0.35
    walk_start: float = 1.0
    step_duration: float = 0.65
    n_steps: int = 20
    double_support_fraction: float = 0.20
    step_width: float = 0.18
    foot_half_length: float = 0.14
    foot_half_width: float = 0.07
    swing_height: float = 0.08
    com_height: float = 0.90
    gravity: float = 9.81
    duration: float = 16.0
    reference_dt: float = 0.005

    def __post_init__(self):
        values = [value for value in self.__dict__.values()]
        if not np.all(np.isfinite(values)):
            raise ValueError("gait parameters must be finite")
        positive = (self.forward_speed, self.step_duration, self.step_width,
                    self.foot_half_length, self.foot_half_width, self.swing_height,
                    self.com_height, self.gravity, self.duration, self.reference_dt)
        if min(positive) <= 0 or self.walk_start < 0:
            raise ValueError("gait lengths, rates and durations must be positive")
        if not isinstance(self.n_steps, int) or self.n_steps < 1:
            raise ValueError("n_steps must be a positive integer")
        if not 0 < self.double_support_fraction < 1:
            raise ValueError("double_support_fraction must be between zero and one")
        end = self.walk_start + (self.n_steps + 1) * self.step_duration
        if self.duration < end + self.double_support_fraction * self.step_duration:
            raise ValueError("duration must include all steps and the final transfer")


def _smoothstep(s):
    s = np.clip(s, 0.0, 1.0)
    return s * s * s * (10.0 + s * (-15.0 + 6.0 * s))


def _convex_hull(points):
    """Monotone-chain hull, counterclockwise without a repeated endpoint."""
    ordered = sorted(set(map(tuple, np.asarray(points, dtype=float))))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def point_in_support(point, polygon, tolerance=1e-10):
    """Test membership in a counterclockwise convex support polygon."""
    point, polygon = np.asarray(point, dtype=float), np.asarray(polygon, dtype=float)
    edges = np.roll(polygon, -1, axis=0) - polygon
    relative = point - polygon
    cross = edges[:, 0] * relative[:, 1] - edges[:, 1] * relative[:, 0]
    return bool(np.all(cross >= -tolerance))


def project_to_support(point, polygon):
    """Euclidean projection onto the actual convex support polygon.

    Projecting to its axis-aligned bounding box is generally unsafe when the
    feet are staggered: the box includes unsupported corners.
    """
    point, polygon = np.asarray(point, dtype=float), np.asarray(polygon, dtype=float)
    if point.shape != (2,) or polygon.ndim != 2 or polygon.shape[1] != 2:
        raise ValueError("expected point (2,) and polygon (n,2)")
    if polygon.shape[0] < 3 or not np.all(np.isfinite(point)) or not np.all(np.isfinite(polygon)):
        raise ValueError("support projection requires finite point and polygon")
    if point_in_support(point, polygon):
        return point.copy()
    edge = np.roll(polygon, -1, axis=0) - polygon
    denominator = np.sum(edge * edge, axis=1)
    weights = np.divide(np.sum((point - polygon) * edge, axis=1), denominator,
                        out=np.zeros(len(edge)), where=denominator > 0)
    candidates = polygon + np.clip(weights, 0.0, 1.0)[:, None] * edge
    return candidates[np.argmin(np.sum((candidates - point) ** 2, axis=1))].copy()


def intersect_support_polygons(polygons, tolerance=1e-10):
    """Intersect convex polygons and return a counterclockwise polygon.

    This is useful for a zero-order-held CoP command across a contact switch:
    the command must belong to every support polygon during its hold interval.
    Empty or zero-area intersections raise ``ValueError``. Clockwise inputs
    are accepted, but nonconvex or invalid polygons are rejected.
    """
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")

    def cross(a, b):
        return a[0] * b[1] - a[1] * b[0]

    def normalized(polygon):
        polygon = np.asarray(polygon, dtype=float)
        if (polygon.ndim != 2 or polygon.shape[1] != 2 or
                polygon.shape[0] < 3 or not np.all(np.isfinite(polygon))):
            raise ValueError("expected finite convex support polygons of shape (n,2)")
        if np.linalg.norm(polygon[-1] - polygon[0]) < tolerance:
            polygon = polygon[:-1]
        if len(polygon) < 3:
            raise ValueError("support polygon must have positive area")
        shifted = polygon - polygon[0]
        area2 = sum(cross(shifted[i], shifted[(i + 1) % len(shifted)])
                    for i in range(len(shifted)))
        if abs(area2) <= tolerance:
            raise ValueError("support polygon must have positive area")
        if area2 < 0:
            polygon = polygon[::-1]
        edges = np.roll(polygon, -1, axis=0) - polygon
        turns = [cross(edges[i], edges[(i + 1) % len(edges)]) for i in range(len(edges))]
        if min(turns) < -tolerance:
            raise ValueError("support polygons must be convex")
        return polygon.copy()

    polygons = list(polygons)
    if not polygons:
        raise ValueError("at least one support polygon is required")
    result = normalized(polygons[0])
    for other in polygons[1:]:
        clipper = normalized(other)
        for index, edge_start in enumerate(clipper):
            edge = clipper[(index + 1) % len(clipper)] - edge_start
            if len(result) == 0:
                raise ValueError("support polygons have empty intersection")
            clipped = []
            previous = result[-1]
            previous_side = cross(edge, previous - edge_start)
            for current in result:
                current_side = cross(edge, current - edge_start)
                previous_inside, current_inside = previous_side >= -tolerance, current_side >= -tolerance
                if previous_inside != current_inside:
                    denominator = previous_side - current_side
                    if abs(denominator) > np.finfo(float).eps:
                        fraction = np.clip(previous_side / denominator, 0.0, 1.0)
                        clipped.append(previous + fraction * (current - previous))
                if current_inside:
                    clipped.append(current)
                previous, previous_side = current, current_side
            result = np.asarray(clipped, dtype=float).reshape(-1, 2)
        if len(result) < 3:
            raise ValueError("support polygons have empty or zero-area intersection")
        # Drop collinear/duplicate vertices generated by clipping coincident edges.
        result = _convex_hull(result)
        result = normalized(result)
    return result


class WalkingReference:
    """Deterministic alternating feet, followed by a closing step and standing.

    ``sample_horizon`` returns N control references and N+1 state references.
    ``support_lower/upper`` describe a conservative *inscribed* rectangle.
    The supplied support polygons expose the complete double-support hull.
    """

    def __init__(self, config=None):
        self.config = config or GaitConfig()
        cfg = self.config
        self.duration = cfg.duration
        self.step_length = cfg.forward_speed * cfg.step_duration
        self.walk_end = cfg.walk_start + (cfg.n_steps + 1) * cfg.step_duration
        feet = np.array([[cfg.start_x, cfg.start_y + cfg.step_width / 2, 0.0],
                         [cfg.start_x, cfg.start_y - cfg.step_width / 2, 0.0]])
        self.initial_feet = feet.copy()
        self._starts, self._ends = [], []
        for event in range(cfg.n_steps + 1):
            self._starts.append(feet.copy())
            swing = 1 - event % 2
            feet[swing, 0] = cfg.start_x + min(event + 1, cfg.n_steps) * self.step_length
            self._ends.append(feet.copy())
        self.final_feet = feet.copy()
        self._starts = np.asarray(self._starts)
        self._ends = np.asarray(self._ends)
        self._build_com_reference()

    def _foot_schedule(self, time):
        cfg = self.config
        ds_time = cfg.double_support_fraction * cfg.step_duration
        if time < cfg.walk_start:
            feet = self.initial_feet.copy()
            return feet, feet[:, :2].mean(axis=0), (0, 1), "standing", 0
        if time >= self.walk_end:
            feet = self.final_feet.copy()
            # The final closing step leaves the opposite foot as stance foot.
            final_stance = cfg.n_steps % 2
            s = _smoothstep((time - self.walk_end) / ds_time)
            cop = (1 - s) * feet[final_stance, :2] + s * feet[:, :2].mean(axis=0)
            phase = "double_support" if time < self.walk_end + ds_time else "standing"
            return feet, cop, (0, 1), phase, cfg.n_steps + 1
        event_float = (time - cfg.walk_start) / cfg.step_duration
        event = min(int(math.floor(event_float)), cfg.n_steps)
        elapsed = time - (cfg.walk_start + event * cfg.step_duration)
        feet = self._starts[event].copy()
        swing, stance = 1 - event % 2, event % 2
        if elapsed < ds_time:
            if event == 0:
                previous_cop = feet[:, :2].mean(axis=0)
            else:
                previous_cop = feet[swing, :2]
            s = _smoothstep(elapsed / ds_time)
            cop = (1 - s) * previous_cop + s * feet[stance, :2]
            return feet, cop, (0, 1), "double_support", event
        fraction = (elapsed - ds_time) / (cfg.step_duration - ds_time)
        s = _smoothstep(fraction)
        feet[swing] = (1 - s) * self._starts[event, swing] + s * self._ends[event, swing]
        feet[swing, 2] = cfg.swing_height * 16.0 * fraction ** 2 * (1 - fraction) ** 2
        phase = "left_support" if stance == 0 else "right_support"
        return feet, feet[stance, :2].copy(), (stance,), phase, event

    def _build_com_reference(self):
        """Bounded LIP solution using exact filtering of a linear CoP interpolant."""
        cfg = self.config
        omega = math.sqrt(cfg.gravity / cfg.com_height)
        padding = 6.0
        times = np.arange(-padding, cfg.duration + padding + cfg.reference_dt / 2,
                          cfg.reference_dt)
        z = np.asarray([self._foot_schedule(t)[1] for t in times])
        dt = cfg.reference_dt
        decay = math.exp(-omega * dt)
        slope_weight = dt - (1 - decay) / omega
        past, future = np.empty_like(z), np.empty_like(z)
        past[0], future[-1] = z[0], z[-1]
        for i in range(1, len(times)):
            slope = (z[i] - z[i - 1]) / dt
            past[i] = decay * past[i - 1] + (1 - decay) * z[i - 1] + slope_weight * slope
        for i in range(len(times) - 2, -1, -1):
            slope = (z[i + 1] - z[i]) / dt
            future[i] = decay * future[i + 1] + (1 - decay) * z[i + 1] - slope_weight * slope
        position = (past + future) / 2
        velocity = omega * (future - past) / 2
        self._times = times
        self._com = np.column_stack((position[:, 0], velocity[:, 0],
                                     position[:, 1], velocity[:, 1]))

    def _support(self, feet, stance, cop):
        cfg = self.config
        half = np.array([cfg.foot_half_length, cfg.foot_half_width])
        corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * half
        vertices = np.concatenate([feet[index, :2] + corners for index in stance])
        polygon = _convex_hull(vertices)
        edges = np.roll(polygon, -1, axis=0) - polygon
        normals = np.column_stack((edges[:, 1], -edges[:, 0]))
        margins = np.sum(normals * (polygon - cop), axis=1)
        denominators = np.abs(normals) @ half
        scale = min(1.0, float(np.min(margins / denominators)))
        scale = max(scale, 0.0)
        return polygon, cop - scale * half, cop + scale * half

    def sample(self, time):
        if not np.isfinite(time):
            raise ValueError("time must be finite")
        feet, cop, stance, phase, count = self._foot_schedule(float(time))
        polygon, lower, upper = self._support(feet, stance, cop)
        state = np.array([np.interp(time, self._times, self._com[:, i]) for i in range(4)])
        return {"com_ref": state, "cop_ref": cop, "support_lower": lower,
                "support_upper": upper, "support_polygon": polygon,
                "feet": feet, "stance_feet": stance, "phase": phase,
                "step_count": count}

    def sample_horizon(self, time, dt, N):
        if not np.isfinite(dt) or dt <= 0 or not isinstance(N, (int, np.integer)) or N < 1:
            raise ValueError("dt must be positive and N must be a positive integer")
        samples = [self.sample(time + i * dt) for i in range(N + 1)]
        return {"com_ref": np.asarray([item["com_ref"] for item in samples]),
                "cop_ref": np.asarray([item["cop_ref"] for item in samples[:-1]]),
                "support_lower": np.asarray([item["support_lower"] for item in samples[:-1]]),
                "support_upper": np.asarray([item["support_upper"] for item in samples[:-1]]),
                "support_polygons": [item["support_polygon"] for item in samples[:-1]],
                "feet": np.asarray([item["feet"] for item in samples]),
                "phase": [item["phase"] for item in samples[:-1]],
                "step_count": np.asarray([item["step_count"] for item in samples])}

    def support_over_interval(self, time, dt):
        """Support intersection throughout the half-open interval [time,time+dt).

        Prescribed stance feet are fixed within each contact phase. Enumerating
        all phase boundaries therefore gives the exact intersection, including
        switches that do not coincide with a controller update or physics tick.
        A long interval spanning disjoint left/right supports raises ValueError.
        """
        if not np.isfinite(time) or not np.isfinite(dt) or dt <= 0:
            raise ValueError("time must be finite and dt must be finite and positive")
        cfg = self.config
        end = time + dt
        events = []
        for event in range(cfg.n_steps + 2):
            base = cfg.walk_start + event * cfg.step_duration
            events.extend((base, base + cfg.double_support_fraction * cfg.step_duration))
        cuts = [time] + sorted(set(event for event in events if time < event < end)) + [end]
        probes = [time] + [(left + right) / 2 for left, right in zip(cuts[:-1], cuts[1:])]
        polygons = [self.sample(probe)["support_polygon"] for probe in probes]
        return intersect_support_polygons(polygons)
