"""Stateless scalar schedules for controlled SAC curriculum training."""


def linear_wind_limit(step: int, start: float, finish: float, duration: int) -> float:
    """Maximum sampled wind speed in m/s at a vectorized environment step.

    The sampler uses this ceiling on each environment's next wind resample;
    existing wind vectors are not abruptly rewritten.
    """
    if duration <= 0 or not 0.0 <= start <= finish:
        raise ValueError('duration > 0 and 0 <= start <= finish required')
    fraction = min(1.0, max(0.0, float(step) / float(duration)))
    return start + (finish - start) * fraction
