"""NumPy/SciPy port of the official DIAL-MPC annealed sampling optimizer.

Algorithm source: LeCAR-Lab/dial-mpc, core/dial_core.py, revision
871c84f1cefbe9fa9d83a06b05ab4f0243c1a05c (Apache-2.0).
See UPSTREAM.md and LICENSE-DIAL-MPC for attribution and port differences.

The dynamics and objective are supplied by a batched rollout callback. No
policy, dynamics model, diffusion network or gradient is learned here.
"""

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
from scipy.interpolate import make_interp_spline


@dataclass(frozen=True)
class DialConfig:
    samples: int = 512
    horizon_steps: int = 20
    knot_intervals: int = 5
    iterations: int = 2
    init_iterations: int = 10
    temperature: float = .05
    sigma: float = 1.0
    horizon_decay: float = .9
    trajectory_decay: float = .5
    freeze_first: bool = True

    def __post_init__(self):
        for name in ("samples", "horizon_steps", "knot_intervals", "iterations", "init_iterations"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.knot_intervals < 2 or self.horizon_steps < self.knot_intervals:
            raise ValueError("quadratic interpolation requires 2 <= knot_intervals <= horizon_steps")
        for name in ("temperature", "sigma", "horizon_decay", "trajectory_decay"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.horizon_decay > 1 or self.trajectory_decay > 1:
            raise ValueError("annealing factors must lie in (0,1]")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class DIALResult:
    nodes: np.ndarray
    controls: np.ndarray
    diagnostics: tuple


def cost_weights(costs, temperature):
    """Official standardized-reward softmax, expressed with costs.

    Subtracting the incumbent cost gives the same normalized weights as
    subtracting any finite baseline. Infinite/NaN rollout costs are rejected
    as failed samples. A constant-cost batch retains its incumbent exactly
    instead of producing the upstream implementation's 0/0.
    """
    costs = np.asarray(costs, dtype=float)
    if costs.ndim != 1 or costs.size < 2:
        raise ValueError("costs must be a vector including sampled candidates and incumbent")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    finite = np.isfinite(costs)
    if not np.any(finite):
        raise RuntimeError("Every sampled MPC rollout has invalid cost")
    valid = costs[finite]
    spread = float(np.std(valid))
    scale = max(1., float(np.max(np.abs(valid))))
    weights = np.zeros(costs.size)
    if spread <= np.finfo(float).eps * scale:
        index = costs.size - 1 if finite[-1] else int(np.flatnonzero(finite)[0])
        weights[index] = 1.
        return weights
    baseline = costs[-1] if finite[-1] else float(np.min(valid))
    logits = -(valid - baseline) / spread / temperature
    logits -= np.max(logits)
    unnormalized = np.exp(logits)
    weights[finite] = unnormalized / np.sum(unnormalized)
    return weights


class DIALOptimizer:
    """Dual-annealing MPPI optimizer over normalized control knots.

    The callback receives (samples+1, horizon_steps+1, action_dim) controls
    and returns one accumulated cost per trajectory. The last trajectory is
    always the incumbent. Quadratic interpolation can overshoot [-1,1]
    between clipped knots, exactly as upstream; actuator-range clipping must
    therefore be represented identically in rollout and real-plant decoding.

    Defaults freeze the first knot, following the official pipelined loop.
    Set freeze_first=False explicitly for synchronous zero-delay replanning.
    This adaptation changes action timing, not the dual annealing update.
    """

    def __init__(self, action_dim, config=None, seed=0):
        if not isinstance(action_dim, int) or isinstance(action_dim, bool) or action_dim < 1:
            raise ValueError("action_dim must be a positive integer")
        self.action_dim = action_dim
        self.config = DialConfig() if config is None else config
        if not isinstance(self.config, DialConfig):
            raise TypeError("config must be DialConfig")
        self.rng = np.random.default_rng(seed)
        c = self.config
        # Unit spacing is equivalent to upstream's seconds; both grids scale
        # by the same control interval and spline values are invariant to it.
        self.knot_times = np.linspace(0., c.horizon_steps, c.knot_intervals + 1)
        self.control_times = np.arange(c.horizon_steps + 1, dtype=float)
        self._node_to_control = make_interp_spline(
            self.knot_times, np.eye(c.knot_intervals + 1), k=2)(self.control_times)
        self._control_to_node = make_interp_spline(
            self.control_times, np.eye(c.horizon_steps + 1), k=2)(self.knot_times)

    def _nodes(self, nodes):
        value = np.asarray(nodes, dtype=float)
        shape = (self.config.knot_intervals + 1, self.action_dim)
        if value.ndim < 2 or value.shape[-2:] != shape or not np.all(np.isfinite(value)):
            raise ValueError(f"nodes must have finite trailing shape {shape}")
        return value

    def interpolate(self, nodes):
        """Quadratic spline interpolation, preserving arbitrary leading batch axes."""
        return np.einsum("hj,...ja->...ha", self._node_to_control, self._nodes(nodes))

    def shift(self, nodes, n_steps=1, tail_action=None):
        """Recede the dense trajectory and interpolate back to knots.

        The official implementation appends normalized zero at the horizon;
        tail_action makes any different terminal initialization explicit.
        """
        if not isinstance(n_steps, int) or not 1 <= n_steps <= self.config.horizon_steps:
            raise ValueError("n_steps must lie between 1 and horizon_steps")
        controls = self.interpolate(nodes)
        shifted = np.roll(controls, -n_steps, axis=-2)
        tail = np.zeros(self.action_dim) if tail_action is None else np.asarray(tail_action, dtype=float)
        if tail.shape != (self.action_dim,) or not np.all(np.isfinite(tail)):
            raise ValueError("tail_action must be a finite action vector")
        shifted[..., -n_steps:, :] = tail
        return np.einsum("jh,...ha->...ja", self._control_to_node, shifted)

    def noise_schedule(self, iterations=None):
        """Standard deviations sigma * gamma_h**(K-j) * gamma_i**i."""
        count = self.config.iterations if iterations is None else iterations
        if not isinstance(count, int) or count <= 0:
            raise ValueError("iterations must be a positive integer")
        c = self.config
        horizons = c.horizon_decay ** np.arange(c.knot_intervals, -1, -1)
        trajectories = c.trajectory_decay ** np.arange(count)
        return c.sigma * trajectories[:, None] * horizons[None, :]

    def optimize(self, initial_nodes, rollout_costs: Callable, iterations=None):
        """Refine a control trajectory using full-model sampled rollout costs."""
        mean = self._nodes(initial_nodes).copy()
        if mean.ndim != 2:
            raise ValueError("initial_nodes must describe one trajectory")
        mean = np.clip(mean, -1., 1.)
        c = self.config
        diagnostics = []
        for index, noise_scale in enumerate(self.noise_schedule(iterations)):
            noise = self.rng.normal(size=(c.samples, c.knot_intervals + 1, self.action_dim))
            samples = mean[None, :, :] + noise * noise_scale[None, :, None]
            if c.freeze_first:
                samples[:, 0, :] = mean[0]
            candidates = np.concatenate((samples, mean[None]), axis=0)
            candidates = np.clip(candidates, -1., 1.)
            costs = np.asarray(rollout_costs(self.interpolate(candidates)), dtype=float)
            if costs.shape != (c.samples + 1,):
                raise ValueError(f"rollout_costs must return shape {(c.samples + 1,)}")
            weights = cost_weights(costs, c.temperature)
            mean = np.einsum("n,nja->ja", weights, candidates)
            finite = costs[np.isfinite(costs)]
            diagnostics.append({"iteration": index,
                                "noise_std_first": float(noise_scale[0]),
                                "noise_std_last": float(noise_scale[-1]),
                                "incumbent_cost": float(costs[-1]) if np.isfinite(costs[-1]) else None,
                                "best_sample_cost": float(np.min(finite)),
                                "mean_finite_cost": float(np.mean(finite)),
                                "std_finite_cost": float(np.std(finite)),
                                "effective_samples": float(1 / np.sum(weights**2)),
                                "invalid_samples": int(np.count_nonzero(~np.isfinite(costs)))})
        return DIALResult(mean, self.interpolate(mean), tuple(diagnostics))
