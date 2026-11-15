"""Configuration for the synthetic wind map and selected walking region.

These parameters describe an idealized horizontal velocity field, not a CFD
solution or a calibrated disaster-weather model. SI units are used throughout.
"""

from dataclasses import dataclass
import math
from typing import Tuple


@dataclass(frozen=True)
class TornadoConfig:
    """A smooth vortex on a 100 m map with a 10 m walking test region.

    ``peak_tangential_speed_mps`` and ``peak_inflow_speed_mps`` are the
    respective component maxima, both reached at ``core_radius_m``. Positive
    tangential speed gives counterclockwise circulation. The inflow is only a
    horizontal modeling term; vertical flow and mass conservation are not
    resolved by this two-dimensional illustration.
    """

    map_size_m: float = 100.0
    center_m: Tuple[float, float] = (50.0, 50.0)
    core_radius_m: float = 12.0
    peak_tangential_speed_mps: float = 12.0
    peak_inflow_speed_mps: float = 1.0
    patch_origin_m: Tuple[float, float] = (68.0, 45.0)
    patch_size_m: float = 10.0
    air_density_kg_m3: float = 1.225
    drag_coefficient: float = 1.0
    projected_area_m2: float = 0.5

    def __post_init__(self):
        positive = ("map_size_m", "core_radius_m", "patch_size_m",
                    "air_density_kg_m3", "projected_area_m2")
        nonnegative = ("peak_inflow_speed_mps", "drag_coefficient")
        for name in positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in nonnegative:
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isfinite(self.peak_tangential_speed_mps):
            raise ValueError("peak_tangential_speed_mps must be finite")
        for name in ("center_m", "patch_origin_m"):
            value = getattr(self, name)
            if len(value) != 2 or not all(math.isfinite(v) for v in value):
                raise ValueError(f"{name} must contain two finite coordinates")
        if any(v < 0 or v > self.map_size_m for v in self.center_m):
            raise ValueError("center_m must lie inside the map")
        if any(v < 0 or v + self.patch_size_m > self.map_size_m
               for v in self.patch_origin_m):
            raise ValueError("the walking patch must lie inside the map")

    @property
    def patch_bounds_m(self):
        """Return world-coordinate bounds as (xmin, xmax, ymin, ymax)."""
        x, y = self.patch_origin_m
        return x, x + self.patch_size_m, y, y + self.patch_size_m


DEFAULT_TORNADO = TornadoConfig()
