"""Seeded finite-duration external loads, independent of controller state.

Events are sampled once per episode. Evaluating the schedule never changes
the random generator or simulator state. Link forces/torques are expressed in
world axes at each link COM. Joint loads are generalized joint torques; they
never write floating-base velocity or position.
"""

from dataclasses import asdict, dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class DisturbanceConfig:
    enabled: bool = False
    interval_s: tuple = (1.0, 3.0)
    duration_s: tuple = (0.1, 0.35)
    settle_time_s: float = 0.5
    link_force_max_n: float = 150.0
    link_torque_max_nm: float = 20.0
    joint_torque_max_nm: float = 30.0
    kind_probabilities: tuple = (0.5, 0.2, 0.3)
    pulse_shape: str = "half_sine"
    body_names: tuple = ()
    joint_names: tuple = ()

    def __post_init__(self):
        for name in ("interval_s", "duration_s"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (2,) or not np.all(np.isfinite(values)) or not 0 < values[0] <= values[1]:
                raise ValueError(f"{name} must contain ordered positive finite bounds")
            object.__setattr__(self, name, tuple(map(float, values)))
        for name in ("settle_time_s", "link_force_max_n", "link_torque_max_nm", "joint_torque_max_nm"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        probabilities = np.asarray(self.kind_probabilities, dtype=float)
        if probabilities.shape != (3,) or not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1):
            raise ValueError("kind_probabilities must be three nonnegative probabilities summing to one")
        object.__setattr__(self, "kind_probabilities", tuple(map(float, probabilities)))
        if self.pulse_shape not in ("half_sine", "constant"):
            raise ValueError("pulse_shape must be half_sine or constant")
        object.__setattr__(self, "body_names", tuple(self.body_names))
        object.__setattr__(self, "joint_names", tuple(self.joint_names))


@dataclass(frozen=True)
class DisturbanceSample:
    body_forces: np.ndarray
    body_torques: np.ndarray
    joint_torques: np.ndarray
    active_event_ids: tuple = ()


@dataclass(frozen=True)
class DisturbanceEvent:
    kind: str
    target: str
    start_s: float
    duration_s: float
    amplitude: tuple
    pulse_shape: str = "half_sine"

    def __post_init__(self):
        if self.kind not in ("link_force", "link_torque", "joint_torque"):
            raise ValueError("unsupported disturbance kind")
        length = 1 if self.kind == "joint_torque" else 3
        values = np.asarray(self.amplitude, dtype=float)
        if values.shape != (length,) or not np.all(np.isfinite(values)):
            raise ValueError(f"amplitude must have {length} finite components")
        if not np.all(np.isfinite([self.start_s, self.duration_s])) or self.start_s < 0 or self.duration_s <= 0:
            raise ValueError("event start must be nonnegative and duration positive")
        if self.pulse_shape not in ("half_sine", "constant"):
            raise ValueError("pulse_shape must be half_sine or constant")
        object.__setattr__(self, "amplitude", tuple(map(float, values)))

    def factor(self, time_s):
        phase = (time_s - self.start_s) / self.duration_s
        if not 0 <= phase < 1:
            return 0.0
        return float(np.sin(np.pi * phase)) if self.pulse_shape == "half_sine" else 1.0

    def impulse(self):
        """Exact force/torque time integral: N*s or N*m*s."""
        factor = 2 / np.pi if self.pulse_shape == "half_sine" else 1.0
        return np.asarray(self.amplitude) * self.duration_s * factor


class DisturbanceScheduler:
    def __init__(self, model, config=None):
        self.model = model
        self.config = DisturbanceConfig() if config is None else config
        if isinstance(self.config, dict):
            self.config = DisturbanceConfig(**self.config)
        if not isinstance(self.config, DisturbanceConfig):
            raise TypeError("config must be a dict or DisturbanceConfig")
        root = model.body("pelvis").id
        self._body_map = {model.body(i).name: i for i in range(model.nbody)
                          if model.body_rootid[i] == root and model.body_mass[i] > 0}
        joints = np.unique(model.actuator_trnid[:, 0])
        self._joint_map = {model.joint(int(j)).name: int(model.jnt_dofadr[j]) for j in joints
                           if j >= 0 and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE}
        self.body_names = self.config.body_names or tuple(self._body_map)
        self.joint_names = self.config.joint_names or tuple(self._joint_map)
        if not self.body_names or any(name not in self._body_map for name in self.body_names):
            raise ValueError("body_names must name robot links with positive mass")
        if not self.joint_names or any(name not in self._joint_map for name in self.joint_names):
            raise ValueError("joint_names must name actuated scalar hinge joints")
        self._events = []
        self.seed = None
        self.horizon_s = 0.0
        self.strength = 0.0

    @property
    def events(self):
        return [asdict(event) for event in self._events]

    def reset(self, seed=0, horizon_s=20.0, strength=1.0, events=None):
        if not np.isfinite(horizon_s) or horizon_s <= 0 or not np.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("horizon must be positive and strength in [0,1]")
        self.seed, self.horizon_s, self.strength = int(seed), float(horizon_s), float(strength)
        self._events = []
        if events is not None:
            for specification in events:
                event = specification if isinstance(specification, DisturbanceEvent) else DisturbanceEvent(**specification)
                mapping = self._joint_map if event.kind == "joint_torque" else self._body_map
                if event.target not in mapping:
                    raise ValueError(f"invalid disturbance target {event.target}")
                if event.start_s + event.duration_s > horizon_s + 1e-9:
                    raise ValueError("fixed event extends beyond the episode horizon")
                self._events.append(event)
            return self.events
        if not self.config.enabled or strength == 0:
            return self.events
        rng = np.random.default_rng(seed)
        c = self.config
        time_s = c.settle_time_s
        while True:
            time_s += float(rng.uniform(*c.interval_s))
            duration = float(rng.uniform(*c.duration_s))
            if time_s + duration > horizon_s:
                break
            kind = str(rng.choice(("link_force", "link_torque", "joint_torque"), p=c.kind_probabilities))
            if kind == "joint_torque":
                target = str(rng.choice(self.joint_names))
                amplitude = (float(rng.choice((-1, 1)) * rng.uniform(.3, 1) * c.joint_torque_max_nm * strength),)
            else:
                target = str(rng.choice(self.body_names))
                direction = rng.normal(size=3)
                direction /= max(np.linalg.norm(direction), 1e-12)
                maximum = c.link_force_max_n if kind == "link_force" else c.link_torque_max_nm
                amplitude = tuple(direction * rng.uniform(.3, 1) * maximum * strength)
            self._events.append(DisturbanceEvent(kind, target, time_s, duration, amplitude, c.pulse_shape))
        return self.events

    def sample(self, time_s):
        if not np.isfinite(time_s):
            raise ValueError("time_s must be finite")
        forces = np.zeros((self.model.nbody, 3))
        torques = np.zeros_like(forces)
        joints = np.zeros(self.model.nv)
        active = []
        for index, event in enumerate(self._events):
            factor = event.factor(time_s)
            if factor == 0:
                continue
            active.append(index)
            value = factor * np.asarray(event.amplitude)
            if event.kind == "joint_torque":
                joints[self._joint_map[event.target]] += value[0]
            elif event.kind == "link_force":
                forces[self._body_map[event.target]] += value
            else:
                torques[self._body_map[event.target]] += value
        return DisturbanceSample(forces, torques, joints, tuple(active))

    def metadata(self):
        return {"config": asdict(self.config), "seed": self.seed, "horizon_s": self.horizon_s,
                "strength": self.strength, "events": self.events,
                "link_frame": "world axes, forces and torques applied at each link COM",
                "joint_frame": "generalized hinge-joint torque; floating-base DOFs excluded",
                "schedule": "sampled once at reset; independent of controller state and sampling order"}
