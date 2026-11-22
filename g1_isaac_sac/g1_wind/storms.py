"""Meteorology-informed *wind-speed envelopes*, not full weather simulations.

NOAA NHC SSHWS: maximum 1-minute sustained near 10 m, mph converted to m/s.
NWS severe thunderstorm: damaging gust >=58 mph, NOT sustained.
Tornado EF0..EF3: damage-based estimated 3-second gusts, NOT 1-minute sustained.

The curriculum ceilings are approximate and rounded; they are engineering targets,
not meteorological storm labels for a 20-second simulated episode.
"""
from dataclasses import dataclass

MPH_TO_MPS = 0.44704

@dataclass(frozen=True)
class StormClass:
    name: str
    lower_mph: float
    upper_mph: float | None
    averaging: str

    @property
    def lower_mps(self) -> float:
        return self.lower_mph * MPH_TO_MPS

    @property
    def upper_mps(self) -> float | None:
        return None if self.upper_mph is None else self.upper_mph * MPH_TO_MPS


REFERENCE_CLASSES = {
    "tropical_storm": StormClass("Tropical storm", 39, 73, "1-minute sustained"),
    "hurricane_cat1": StormClass("Category 1 hurricane", 74, 95, "1-minute sustained"),
    "hurricane_cat2": StormClass("Category 2 hurricane", 96, 110, "1-minute sustained"),
    "hurricane_cat3": StormClass("Category 3 hurricane", 111, 129, "1-minute sustained"),
    "hurricane_cat4": StormClass("Category 4 hurricane", 130, 156, "1-minute sustained"),
    "hurricane_cat5": StormClass("Category 5 hurricane", 157, None, "1-minute sustained"),
    "ef0_gust": StormClass("EF0 tornado equivalent gust", 65, 85, "estimated 3-second gust"),
    "ef1_gust": StormClass("EF1 tornado equivalent gust", 86, 110, "estimated 3-second gust"),
    "ef2_gust": StormClass("EF2 tornado equivalent gust", 111, 135, "estimated 3-second gust"),
    "ef3_gust": StormClass("EF3 tornado equivalent gust", 136, 165, "estimated 3-second gust"),
}

# Target = rounded SI stress-test ceiling; not a storm's measured maximum.
# hurricane_cat5 and EF3 are open/high ranges; our 80/74 m/s are chosen FINITE tests.
STORM_TARGETS_MPS = {
    'high_wind': 18.0,
    'severe_gust': 26.0,
    'tropical_storm': 33.0,
    'hurricane_cat1': 42.0,  # <=95 mph; old 43 m/s preset mislabeled Category 1
    'hurricane_cat2': 50.0,
    'hurricane_cat3': 58.0,
    'hurricane_cat4': 70.0,
    'hurricane_cat5': 80.0,
    'ef0_gust': 38.0,
    'ef1_gust': 49.0,
    'ef2_gust': 60.0,
    'ef3_gust': 74.0,
}


def storm_curriculum_ceiling(step: int, start: float, target: float, duration: int) -> float:
    """Piecewise-linear staged upper bound, with anchor points from wind hazards.

    The same global step-to-ceiling schedule applies to every environment;
    actual wind vectors only change on each environment's wind resample.
    """
    if duration <= 0 or not 0 <= start <= target:
        raise ValueError('duration>0 and 0<=start<=target required')
    knots = [start]
    for anchor in (4, 8, 12, 18, 26, 33, 43, 50, 58, 70, 80):
        if start < anchor < target:
            knots.append(float(anchor))
    if not knots or knots[-1] != target:
        knots.append(float(target))
    if len(knots) == 1:
        return target
    fraction = min(1.0, max(0.0, step / duration))
    segment_position = fraction * (len(knots)-1)
    segment = min(int(segment_position), len(knots)-2)
    weight = segment_position - segment
    return knots[segment] * (1-weight) + knots[segment+1] * weight


def storm_min_wind(ceiling: float, fraction: float) -> float:
    if ceiling < 0 or not 0 <= fraction <= 1:
        raise ValueError('Require ceiling>=0, min_fraction in [0,1]')
    return ceiling * fraction
