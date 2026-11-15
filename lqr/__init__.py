"""Homework-style LQR for a wind-disturbed, reduced-order walking model."""

from .controller import ControlResult, LIPWalkingLQR, lip_discrete_dynamics
from .riccati import affine_lqr

__all__ = ["affine_lqr", "lip_discrete_dynamics", "LIPWalkingLQR", "ControlResult"]
