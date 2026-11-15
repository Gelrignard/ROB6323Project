"""Articulated G1 inverse-kinematics replay of reduced-model trajectories.

This class calls mj_forward, never mj_step. It does NOT demonstrate contact
stability or joint-torque walking. The 12 leg joints and floating-base
translation match two flat sole poses and the actual whole-robot CoM;
pelvis orientation and the upper-body joints remain at the nominal posture.
"""

from dataclasses import asdict, dataclass

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .g1_model import get_g1_profile, load_g1_model


@dataclass(frozen=True)
class IKReport:
    success: bool
    max_foot_error_m: float
    com_error_m: float
    max_foot_orientation_error_rad: float
    evaluations: int
    kinematic_only: bool = True


def _skew(vector):
    x, y, z = vector
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)


def _log_jacobian_inverse(vector):
    """Map a world angular-velocity increment to the SO(3) log increment."""
    angle = float(np.linalg.norm(vector))
    cross = _skew(vector)
    coefficient = 1 / 12 if angle < 1e-5 else (1 - .5 * angle / np.tan(.5 * angle)) / angle**2
    return np.eye(3) - .5 * cross + coefficient * (cross @ cross)


class G1Replay:
    """Bounded IK on genuine G1 geometry; the reduced simulation stays separate.

    ``com_height`` must match the reduced-model reference. The default is a
    declared 4 cm lowering from the vendor standing CoM. Excessive lowering
    can violate ankle pitch limits during swing, even when legs can reach.
    Invalid/unreachable targets raise instead of silently stretching limbs.
    """

    def __init__(self, com_height=None, *, position_tolerance=.0015,
                 orientation_tolerance=.015, max_evaluations=60):
        self.model = load_g1_model()
        self.profile = get_g1_profile(self.model)
        self.data = mujoco.MjData(self.model)
        self.com_height = float(self.profile.com_height - .04 if com_height is None else com_height)
        if not np.isfinite(self.com_height) or self.com_height <= 0:
            raise ValueError("com_height must be finite and positive")
        self.position_tolerance = float(position_tolerance)
        self.orientation_tolerance = float(orientation_tolerance)
        self.max_evaluations = int(max_evaluations)
        if min(self.position_tolerance, self.orientation_tolerance, self.max_evaluations) <= 0:
            raise ValueError("IK tolerances and evaluation limit must be positive")
        self.root_body_id = int(self.profile.root_body_id)
        self.sole_site_ids = tuple(map(int, self.profile.sole_site_ids))
        suffixes = ("hip_pitch_joint", "hip_roll_joint", "hip_yaw_joint",
                    "knee_joint", "ankle_pitch_joint", "ankle_roll_joint")
        self.leg_joint_names = tuple(f"{side}_{suffix}" for side in ("left", "right") for suffix in suffixes)
        self.leg_joint_ids = np.array([self.model.joint(name).id for name in self.leg_joint_names])
        self.leg_qpos_indices = self.model.jnt_qposadr[self.leg_joint_ids].astype(int)
        self.dof_indices = np.concatenate((np.arange(3), self.model.jnt_dofadr[self.leg_joint_ids])).astype(int)
        self.qpos_indices = np.concatenate((np.arange(3), self.leg_qpos_indices)).astype(int)
        self.home_qpos = np.array(self.profile.home_qpos, dtype=float).copy()
        self.data.qpos[:] = self.home_qpos
        self.lower = np.concatenate(([-np.inf, -np.inf, .25], self.model.jnt_range[self.leg_joint_ids, 0]))
        self.upper = np.concatenate(([np.inf, np.inf, 1.2], self.model.jnt_range[self.leg_joint_ids, 1]))
        self._variables = self.data.qpos[self.qpos_indices].copy()
        # A bent-knee seed avoids the singular fully extended leg posture.
        for offset in (3, 9):
            self._variables[offset] = -.20
            self._variables[offset + 3] = .40
            self._variables[offset + 4] = -.20
        self._variables[2] -= .04
        self._cached_variables = None
        self._orientation_scale = .15
        self._initialized = False
        self.max_foot_error = 0.0
        self.max_com_error = 0.0
        self.max_orientation_error = 0.0
        self.solved_frames = 0
        self.last_report = None

    def _forward(self, variables):
        if self._cached_variables is None or not np.array_equal(variables, self._cached_variables):
            self.data.qpos[:] = self.home_qpos
            self.data.qpos[self.qpos_indices] = variables
            self.data.qvel[:] = 0
            mujoco.mj_forward(self.model, self.data)
            self._cached_variables = variables.copy()

    def _residual(self, variables, target_com, target_feet):
        self._forward(variables)
        result = [self.data.subtree_com[self.root_body_id] - target_com]
        for index, site in enumerate(self.sole_site_ids):
            result.append(self.data.site_xpos[site] - target_feet[index])
            rotation_error = Rotation.from_matrix(self.data.site_xmat[site].reshape(3, 3)).as_rotvec()
            result.append(self._orientation_scale * rotation_error)
        return np.concatenate(result)

    def _jacobian(self, variables, target_com, target_feet):
        self._forward(variables)
        com_jac = np.zeros((3, self.model.nv))
        mujoco.mj_jacSubtreeCom(self.model, self.data, com_jac, self.root_body_id)
        rows = [com_jac[:, self.dof_indices]]
        for site in self.sole_site_ids:
            position_jac = np.zeros((3, self.model.nv))
            rotation_jac = np.zeros_like(position_jac)
            mujoco.mj_jacSite(self.model, self.data, position_jac, rotation_jac, site)
            rotation_error = Rotation.from_matrix(self.data.site_xmat[site].reshape(3, 3)).as_rotvec()
            rows.append(position_jac[:, self.dof_indices])
            rows.append(self._orientation_scale * _log_jacobian_inverse(rotation_error) @ rotation_jac[:, self.dof_indices])
        return np.vstack(rows)

    def set_state(self, state, feet, time, cop):
        """Set a kinematic replay pose; return an explicit IK residual report.

        ``state`` is [cx,vx,cy,vy] in patch coordinates. ``feet`` are flat sole
        centers with shape (2,3), left then right; z=0 denotes ground contact.
        Velocities are not interpreted as physical generalized velocities.
        """
        state, feet, cop = (np.asarray(value, dtype=float) for value in (state, feet, cop))
        if state.shape != (4,) or feet.shape != (2, 3) or cop.shape != (2,):
            raise ValueError("expected state(4,), feet(2,3), cop(2,)")
        if not all(np.all(np.isfinite(value)) for value in (state, feet, cop)) or not np.isfinite(time):
            raise ValueError("replay inputs must be finite")
        target_com = np.array([state[0], state[2], self.com_height])
        if not self._initialized:
            self._forward(self._variables)
            self._variables[:3] += target_com - self.data.subtree_com[self.root_body_id]
            self._variables = np.clip(self._variables, self.lower + 1e-10, self.upper - 1e-10)
        solution = least_squares(self._residual, self._variables,
                                 jac=self._jacobian, args=(target_com, feet),
                                 bounds=(self.lower, self.upper), max_nfev=self.max_evaluations,
                                 ftol=1e-9, xtol=1e-9, gtol=1e-9, x_scale="jac")
        self._variables = solution.x.copy()
        self._forward(self._variables)
        errors = [np.linalg.norm(self.data.site_xpos[site] - feet[index]) for index, site in enumerate(self.sole_site_ids)]
        angle_errors = [np.linalg.norm(Rotation.from_matrix(self.data.site_xmat[site].reshape(3, 3)).as_rotvec())
                        for site in self.sole_site_ids]
        com_error = float(np.linalg.norm(self.data.subtree_com[self.root_body_id] - target_com))
        foot_error, angle_error = float(max(errors)), float(max(angle_errors))
        success = foot_error <= self.position_tolerance and com_error <= self.position_tolerance and angle_error <= self.orientation_tolerance
        self.last_report = IKReport(success, foot_error, com_error, angle_error, int(solution.nfev))
        self.max_foot_error = max(self.max_foot_error, foot_error)
        self.max_com_error = max(self.max_com_error, com_error)
        self.max_orientation_error = max(self.max_orientation_error, angle_error)
        if not success:
            raise RuntimeError(f"G1 kinematic mapping failed at t={time:.3f}: {asdict(self.last_report)}")
        self._initialized = True
        self.solved_frames += 1
        self.data.time = float(time)
        for name, position in (("cop_marker", [cop[0], cop[1], .02]),
                               ("reference_marker", target_com)):
            body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body >= 0 and self.model.body_mocapid[body] >= 0:
                self.data.mocap_pos[self.model.body_mocapid[body]] = position
        mujoco.mj_forward(self.model, self.data)
        return self.last_report

    def diagnostics(self):
        return {"kinematic_only": True, "solved_frames": self.solved_frames,
                "max_foot_error_m": self.max_foot_error,
                "max_com_error_m": self.max_com_error,
                "max_foot_orientation_error_rad": self.max_orientation_error,
                "com_height_m": self.com_height}
