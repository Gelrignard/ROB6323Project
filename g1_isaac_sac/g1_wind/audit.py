"""Auditable deterministic wind scheduling, mutually exclusive episode flags and gated curriculum.

No simulator dependencies; unit-testable on CPU.  A gate is a TRAINING HEURISTIC,
not a substitute for offline matched evaluation of frozen policies.
"""
import math
from dataclasses import dataclass
import torch


def exclusive_outcomes(terminated, truncated):
    """Fall takes precedence over time-limit when both are true."""
    falls=terminated.bool()
    timeouts=truncated.bool() & ~falls
    return falls,timeouts,falls|timeouts


def prescribed_uniform_wind(time_s, num_envs, speed_mps, heading_deg=0.0,
                            gust_fraction=0.0, gust_period_s=8.0, device='cpu'):
    """Deterministic world wind, independent of robot actions/resets/RNG usage.

    A fixed phase is assigned to each env and reused for both controllers;
    this is a reproducible sinusoidal perturbation, not meteorological turbulence.
    """
    if num_envs < 1 or speed_mps < 0 or gust_fraction < 0 or gust_period_s <= 0:
        raise ValueError('Invalid prescribed wind parameters')
    phase=2*math.pi*torch.remainder(torch.arange(num_envs,device=device)*0.6180339887498948,1.0)
    amplitude=(1.0+gust_fraction*torch.sin(2*math.pi*time_s/gust_period_s+phase)).clamp_min(0.0)
    magnitude=speed_mps*amplitude
    theta=math.radians(heading_deg)
    return torch.stack((magnitude*math.cos(theta),magnitude*math.sin(theta),torch.zeros_like(magnitude)),dim=-1)


@dataclass
class GateDecision:
    tested_episodes: int
    survival_fraction: float | None
    ceiling_before: float
    ceiling_after: float
    advanced: bool
    qualified_windows: int


class SuccessGatedWindCurriculum:
    """Step the ceiling up only after repeated success on *exposed* completed episodes.

    Caller must reject insufficiently-exposed episodes.  A completed episode
    with high wind in only one frame must not count as evidence of robustness.
    """
    def __init__(self, start, target, increment=2.0, min_episodes=64,
                 survival_threshold=0.85, required_windows=2):
        if not 0 <= start <= target or increment<=0 or min_episodes<1 or required_windows<1:
            raise ValueError('Invalid gated curriculum settings')
        if not 0<=survival_threshold<=1:
            raise ValueError('survival_threshold must be in [0,1]')
        self.ceiling=float(start); self.target=float(target); self.increment=float(increment)
        self.min_episodes=int(min_episodes); self.survival_threshold=float(survival_threshold)
        self.required_windows=int(required_windows)
        self.episodes=0; self.falls=0; self.consecutive_successes=0

    def add_completed(self, falls, eligible):
        n=int(eligible.sum().item()); f=int((eligible & falls).sum().item())
        self.episodes+=n; self.falls+=f

    def check(self):
        if self.episodes < self.min_episodes: return None
        survival=1-self.falls/self.episodes
        before=self.ceiling
        if survival>=self.survival_threshold:
            self.consecutive_successes+=1
            if self.consecutive_successes>=self.required_windows:
                self.ceiling=min(self.target,self.ceiling+self.increment)
                self.consecutive_successes=0
        else:
            self.consecutive_successes=0
        result=GateDecision(self.episodes,survival,before,self.ceiling,self.ceiling>before,
                            self.consecutive_successes)
        self.episodes=0; self.falls=0
        return result
