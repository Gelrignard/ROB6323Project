"""Robot state available to the controller; external loads stay in the plant."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RobotObservation:
    """Ideal state-estimator output, without force or simulator solver state.

    Base pose and velocity currently come from simulation. Real hardware
    would need a state estimator; no external-force sensor is assumed.
    """

    time: float
    qpos: np.ndarray
    qvel: np.ndarray

    @classmethod
    def from_data(cls, data):
        return cls(float(data.time), data.qpos.copy(), data.qvel.copy())
