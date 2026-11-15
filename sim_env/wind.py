"""Vectorized idealized tornado velocity and relative-wind drag.

For displacement r=(dx,dy), core radius a, tangential maximum V and radial
inflow maximum U, the planar field is

    w = 2 a / (a**2 + ||r||**2) * (V * [-dy, dx] - U * r).

It is smooth and zero at the center; component magnitudes peak at radius a
and decay as 1/r far from the center. The flow is prescribed and does not
react to the robot. No turbulence, vertical flow, lift, or aerodynamic moments
are resolved. All public functions accept a single (2,) vector or (...,2)
arrays. Position arguments are in WORLD coordinates unless named "local".
"""

import numpy as np

from .config import DEFAULT_TORNADO, TornadoConfig


def _vectors(value, name):
    array = np.asarray(value, dtype=float)
    if array.ndim == 0 or array.shape[-1] != 2:
        raise ValueError(f"{name} must have shape (2,) or (..., 2)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def tornado_velocity(position, config=None, scale=1.0):
    """Return world-frame horizontal wind velocity in m/s.

    ``scale`` is a nonnegative scalar multiplying the entire velocity field.
    It can be varied by a runner to introduce an explicitly prescribed gust.
    The analytical field remains defined outside the plotted map.
    """
    cfg = DEFAULT_TORNADO if config is None else config
    points = _vectors(position, "position")
    if not np.isscalar(scale) or not np.isfinite(scale) or scale < 0:
        raise ValueError("scale must be a finite nonnegative scalar")
    displacement = points - np.asarray(cfg.center_m)
    radius_squared = np.sum(displacement * displacement, axis=-1)
    tangent = np.stack((-displacement[..., 1], displacement[..., 0]), axis=-1)
    factor = 2.0 * cfg.core_radius_m / (cfg.core_radius_m ** 2 + radius_squared)
    return float(scale) * factor[..., None] * (
        cfg.peak_tangential_speed_mps * tangent
        - cfg.peak_inflow_speed_mps * displacement)


def aerodynamic_force(wind_velocity, body_velocity, config=None):
    """Return planar drag force ON the body in N, using relative velocity.

    F = 0.5 * rho * Cd * A * ||w-v|| * (w-v).
    Positive force points along wind relative to the body. This assumes a
    constant projected area and drag coefficient; it is not an energy model.
    """
    cfg = DEFAULT_TORNADO if config is None else config
    wind = _vectors(wind_velocity, "wind_velocity")
    velocity = _vectors(body_velocity, "body_velocity")
    relative = wind - velocity
    speed = np.linalg.norm(relative, axis=-1, keepdims=True)
    coefficient = (0.5 * cfg.air_density_kg_m3 * cfg.drag_coefficient
                   * cfg.projected_area_m2)
    return coefficient * speed * relative


def wind_force(position, robot_velocity, config=None, scale=1.0):
    """Evaluate the wind field and return planar aerodynamic force in N."""
    return aerodynamic_force(tornado_velocity(position, config, scale),
                             robot_velocity, config)


def local_to_world(local_position, config=None):
    """Translate coordinates from the 10 m patch to the 100 m map."""
    cfg = DEFAULT_TORNADO if config is None else config
    return _vectors(local_position, "local_position") + np.asarray(cfg.patch_origin_m)


def world_to_local(world_position, config=None):
    """Translate coordinates from the 100 m map to the walking patch."""
    cfg = DEFAULT_TORNADO if config is None else config
    return _vectors(world_position, "world_position") - np.asarray(cfg.patch_origin_m)
