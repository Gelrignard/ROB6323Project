"""Deterministic, time-dependent test fields in world coordinates and SI units.

These prescribed fields do not solve fluid dynamics. Field evaluation has no
random draws: replaying the same positions and times produces identical wind,
independently of policy actions or the number/order of previous calls.
"""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class WindFieldConfig:
    kind: str = "calm"
    velocity_mps: tuple = (8.0, 0.0, 0.0)
    scale: float = 1.0
    center_m: tuple = (50.0, 50.0)
    core_radius_m: float = 12.0
    tangential_speed_mps: float = 12.0
    inflow_speed_mps: float = 1.0
    center_velocity_mps: tuple = (0.15, 0.0)
    gust_amplitude_mps: tuple = (4.0, 0.0, 0.0)
    gust_period_s: float = 3.0
    gust_phase_rad: float | None = None
    seed: int = 0
    shear_mps_per_m: tuple = (2.0, 0.0, 0.0)
    reference_height_m: float = 1.0

    def __post_init__(self):
        if self.kind not in ("calm", "tornado", "uniform", "gust", "shear", "moving_vortex"):
            raise ValueError(f"unsupported wind kind: {self.kind}")
        for name, size in (("velocity_mps", 3), ("center_m", 2), ("center_velocity_mps", 2),
                           ("gust_amplitude_mps", 3), ("shear_mps_per_m", 3)):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (size,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain {size} finite numbers")
            object.__setattr__(self, name, tuple(map(float, value)))
        values = [self.scale, self.core_radius_m, self.tangential_speed_mps,
                  self.inflow_speed_mps, self.gust_period_s, self.reference_height_m]
        if not np.all(np.isfinite(values)) or self.scale < 0 or self.core_radius_m <= 0 or self.gust_period_s <= 0:
            raise ValueError("wind parameters must be finite; scale nonnegative, radius and period positive")
        if self.inflow_speed_mps < 0:
            raise ValueError("inflow_speed_mps must be nonnegative")
        if self.gust_phase_rad is not None and not np.isfinite(self.gust_phase_rad):
            raise ValueError("gust_phase_rad must be finite or None")
        if not isinstance(self.seed, (int, np.integer)) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")


class WindField:
    def __init__(self, config):
        self.config = config
        self.phase = (float(np.random.default_rng(config.seed).uniform(0, 2 * np.pi))
                      if config.gust_phase_rad is None else float(config.gust_phase_rad))

    def __call__(self, world_points, time_s):
        points = np.asarray(world_points, dtype=float)
        if points.ndim == 0 or points.shape[-1] != 3 or not np.all(np.isfinite(points)):
            raise ValueError("world_points must have finite shape (...,3)")
        if not np.isscalar(time_s) or not np.isfinite(time_s):
            raise ValueError("time_s must be a finite scalar")
        c = self.config
        wind = np.zeros_like(points)
        if c.kind == "calm":
            return wind
        if c.kind in ("tornado", "moving_vortex"):
            center = np.array(c.center_m)
            if c.kind == "moving_vortex":
                center += float(time_s) * np.array(c.center_velocity_mps)
            r = points[..., :2] - center
            tangent = np.stack((-r[..., 1], r[..., 0]), axis=-1)
            factor = 2 * c.core_radius_m / (c.core_radius_m**2 + np.sum(r * r, axis=-1))
            wind[..., :2] = factor[..., None] * (c.tangential_speed_mps * tangent - c.inflow_speed_mps * r)
        else:
            wind[...] = c.velocity_mps
            if c.kind == "gust":
                wind += np.sin(2 * np.pi * float(time_s) / c.gust_period_s + self.phase) * np.array(c.gust_amplitude_mps)
            elif c.kind == "shear":
                wind += (points[..., 2:3] - c.reference_height_m) * np.array(c.shear_mps_per_m)
        return c.scale * wind

    def metadata(self):
        return {**asdict(self.config), "effective_gust_phase_rad": self.phase,
                "coordinate_frame": "world Cartesian xyz, m and m/s", "time_unit": "seconds",
                "model": "prescribed field; no fluid feedback or CFD"}


def make_wind_field(config=None):
    config = WindFieldConfig() if config is None else config
    if isinstance(config, dict):
        config = WindFieldConfig(**config)
    if not isinstance(config, WindFieldConfig):
        raise TypeError("config must be a dict or WindFieldConfig")
    return WindField(config)
