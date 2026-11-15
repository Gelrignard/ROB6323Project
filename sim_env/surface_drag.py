"""Windward pressure integrated over the articulated robot's visual surface.

This is a local surface-load approximation, not a fluid solver. For outward
normal n, panel area A, and relative air velocity u=w-v, the model is

    p = 0.5*rho*Cp*max(0, -u.n)**2,       dF = -p*A*n.

Only incident faces are loaded; there is no suction, skin friction, shielding,
wake interaction, or flow deflection. Internal/overlapping visual shells can
therefore overestimate exposed area. Cp is an explicit uncalibrated constant.
Compiled MuJoCo mesh vertices already include asset scale/recentering; their
matching geom_xmat/geom_xpos maps them to world-aligned patch coordinates.
Do not apply mesh_pos/mesh_quat a second time.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from .config import DEFAULT_TORNADO
from .wind import local_to_world, tornado_velocity


def panel_pressure_forces(normals, areas, relative_velocity, *, density=1.225,
                          pressure_coefficient=1.0):
    """Return per-panel force [N] and pressure [Pa], with unit outward normals."""
    normals = np.asarray(normals, dtype=float)
    areas = np.asarray(areas, dtype=float)
    velocity = np.asarray(relative_velocity, dtype=float)
    if normals.ndim != 2 or normals.shape[1] != 3 or velocity.shape != normals.shape or areas.shape != (len(normals),):
        raise ValueError("expected normals/relative_velocity (N,3) and areas (N,)")
    if not all(np.all(np.isfinite(value)) for value in (normals, areas, velocity)):
        raise ValueError("panel inputs must be finite")
    if np.any(areas <= 0) or not np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-6):
        raise ValueError("panel areas must be positive and outward normals unit length")
    if not np.isfinite(density) or density <= 0 or not np.isfinite(pressure_coefficient) or pressure_coefficient < 0:
        raise ValueError("density must be positive; pressure_coefficient nonnegative")
    incident_speed = np.maximum(0.0, -np.einsum("ij,ij->i", velocity, normals))
    pressure = .5 * float(density) * float(pressure_coefficient) * incident_speed**2
    return -(pressure * areas)[:, None] * normals, pressure


@dataclass(frozen=True)
class SurfaceDragResult:
    points: np.ndarray
    normals: np.ndarray
    areas: np.ndarray
    patch_forces: np.ndarray
    pressure_Pa: np.ndarray
    body_ids: np.ndarray
    body_forces: np.ndarray
    body_torques: np.ndarray
    total_force: np.ndarray
    total_torque_com: np.ndarray
    com: np.ndarray
    relative_velocities: np.ndarray


class SurfaceDrag:
    """Integrate genuine mesh panels and preserve their force-and-moment sum.

    ``max_patches=None`` uses every nondegenerate triangle. The default uses
    fixed, deterministic area-stratified quadrature, distributed by mesh area;
    total area is preserved exactly, but load integration is approximate.
    Geometry group 2 selects G1's visual mesh geoms once, excluding the duplicate
    collision meshes. ``geom_ids`` can explicitly select other mesh models.
    Call mj_forward before evaluate. Evaluation does not change MuJoCo state.
    """

    def __init__(self, model, wind_config=None, max_patches=4096,
                 pressure_coefficient=1.0, *, geom_ids=None):
        self.model = model
        self.wind_config = DEFAULT_TORNADO if wind_config is None else wind_config
        self.pressure_coefficient = float(pressure_coefficient)
        if not np.isfinite(self.pressure_coefficient) or self.pressure_coefficient < 0:
            raise ValueError("pressure_coefficient must be finite and nonnegative")
        if max_patches is not None and (not isinstance(max_patches, (int, np.integer)) or max_patches <= 0):
            raise ValueError("max_patches must be a positive integer or None")
        self.max_patches = max_patches
        selected = np.flatnonzero((model.geom_type == mujoco.mjtGeom.mjGEOM_MESH)
                                  & (model.geom_group == 2) & (model.geom_bodyid > 0)) if geom_ids is None else np.asarray(geom_ids, dtype=int)
        if selected.ndim != 1 or not len(selected) or len(np.unique(selected)) != len(selected):
            raise ValueError("select at least one unique robot mesh geom")
        if np.any(selected < 0) or np.any(selected >= model.ngeom) or np.any(model.geom_type[selected] != mujoco.mjtGeom.mjGEOM_MESH):
            raise ValueError("geom_ids must identify valid mesh geoms")
        self.geom_ids = selected.copy()
        roots = np.unique(model.body_rootid[model.geom_bodyid[selected]])
        if len(roots) != 1 or roots[0] == 0:
            raise ValueError("selected meshes must belong to one articulated robot root")
        self.root_body_id = int(roots[0])
        surfaces = []
        self.flipped_meshes = []
        self.source_triangle_count = 0
        self.degenerate_triangle_count = 0
        for geom_id in selected:
            mesh_id = int(model.geom_dataid[geom_id])
            va, vn = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
            fa, fn = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
            vertices = np.asarray(model.mesh_vert[va:va + vn], dtype=float)
            triangles = vertices[model.mesh_face[fa:fa + fn]]
            twice_area_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            twice_area = np.linalg.norm(twice_area_normals, axis=1)
            valid = twice_area > 1e-14
            self.source_triangle_count += int(fn)
            self.degenerate_triangle_count += int(np.count_nonzero(~valid))
            if not np.any(valid):
                raise ValueError(f"mesh geom {geom_id} has no nondegenerate triangles")
            triangles, twice_area_normals, twice_area = triangles[valid], twice_area_normals[valid], twice_area[valid]
            # The signed-volume check reverses globally inward winding; all
            # supplied G1 meshes are positive-volume, consistently oriented.
            signed_volume = float(np.sum(triangles[:, 0] * np.cross(triangles[:, 1], triangles[:, 2])) / 6)
            if signed_volume < -1e-15:
                twice_area_normals *= -1
                self.flipped_meshes.append(int(geom_id))
            surfaces.append((triangles.mean(axis=1), twice_area_normals / twice_area[:, None], twice_area / 2))
        counts = np.array([len(surface[2]) for surface in surfaces], dtype=int)
        area_totals = np.array([surface[2].sum() for surface in surfaces])
        self.surface_area_m2 = float(area_totals.sum())
        if max_patches is None or max_patches >= int(counts.sum()):
            allocated = counts
        else:
            if max_patches < len(surfaces):
                raise ValueError("max_patches must allow at least one sample per mesh geom")
            # Reserve one point per mesh, then allocate the budget by area.
            weights = area_totals / area_totals.sum() * (max_patches - len(surfaces))
            allocated = np.floor(weights).astype(int) + 1
            remainder = max_patches - int(allocated.sum())
            allocated[np.argsort(-(weights - np.floor(weights)), kind="stable")[:remainder]] += 1
        positions, normals, areas, geoms, bodies = [], [], [], [], []
        for geom_id, surface, budget in zip(selected, surfaces, allocated):
            center, normal, area = surface
            if budget >= len(area):
                chosen = np.arange(len(area))
                quadrature_area = area.copy()
            else:
                # Every equal-area stratum contributes one midpoint sample.
                # Facets larger than a stratum may repeat with smaller weights.
                total = float(area.sum())
                chosen = np.searchsorted(np.cumsum(area), (np.arange(budget) + .5) * total / budget)
                quadrature_area = np.full(budget, total / budget)
            positions.append(center[chosen])
            normals.append(normal[chosen])
            areas.append(quadrature_area)
            geoms.append(np.full(len(chosen), geom_id, dtype=int))
            bodies.append(np.full(len(chosen), model.geom_bodyid[geom_id], dtype=int))
        self.local_points = np.concatenate(positions)
        self.local_normals = np.concatenate(normals)
        self.areas = np.concatenate(areas)
        self.patch_geom_ids = np.concatenate(geoms)
        self.body_ids = np.concatenate(bodies)
        self.active_body_ids = np.unique(self.body_ids)
        for array in (self.areas, self.patch_geom_ids, self.body_ids):
            array.setflags(write=False)
        self._jac_position = np.zeros((3, model.nv))
        self._jac_rotation = np.zeros((3, model.nv))

    def evaluate(self, data, wind_scale=1.0, wind_field=None):
        """Evaluate loads on current state; refresh mj_forward beforehand.

        Body linear/angular velocities are evaluated at each body's COM via
        MuJoCo Jacobians; point velocity includes omega cross its lever arm.
        Returned torques are about body COM for xfrc_applied, while the total
        moment is about the overall articulated robot COM.
        """
        rotations = data.geom_xmat[self.patch_geom_ids].reshape(-1, 3, 3)
        points = np.einsum("nij,nj->ni", rotations, self.local_points) + data.geom_xpos[self.patch_geom_ids]
        normals = np.einsum("nij,nj->ni", rotations, self.local_normals)
        linear = np.zeros((self.model.nbody, 3))
        angular = np.zeros_like(linear)
        for body_id in self.active_body_ids:
            mujoco.mj_jacBodyCom(self.model, data, self._jac_position, self._jac_rotation, int(body_id))
            linear[body_id] = self._jac_position @ data.qvel
            angular[body_id] = self._jac_rotation @ data.qvel
        lever = points - data.xipos[self.body_ids]
        velocity = linear[self.body_ids] + np.cross(angular[self.body_ids], lever)
        if not np.isfinite(wind_scale) or wind_scale < 0:
            raise ValueError("wind_scale must be finite and nonnegative")
        air = np.zeros_like(points)
        if wind_field is None:
            air[:, :2] = tornado_velocity(local_to_world(points[:, :2], self.wind_config), self.wind_config, scale=wind_scale)
        else:
            world_points = points.copy()
            world_points[:, :2] = local_to_world(points[:, :2], self.wind_config)
            air = np.asarray(wind_field(world_points, float(data.time)), dtype=float) * wind_scale
            if air.shape != points.shape or not np.all(np.isfinite(air)):
                raise ValueError("wind_field must return finite velocities with shape (N,3)")
        relative = air - velocity
        patch_forces, pressure = panel_pressure_forces(normals, self.areas, relative,
                                                      density=self.wind_config.air_density_kg_m3,
                                                      pressure_coefficient=self.pressure_coefficient)
        body_forces = np.zeros((self.model.nbody, 3))
        body_torques = np.zeros_like(body_forces)
        patch_torque = np.cross(lever, patch_forces)
        for component in range(3):
            body_forces[:, component] = np.bincount(self.body_ids, weights=patch_forces[:, component], minlength=self.model.nbody)
            body_torques[:, component] = np.bincount(self.body_ids, weights=patch_torque[:, component], minlength=self.model.nbody)
        com = data.subtree_com[self.root_body_id].copy()
        return SurfaceDragResult(points, normals, self.areas, patch_forces, pressure,
                                 self.body_ids, body_forces, body_torques,
                                 body_forces.sum(axis=0), np.cross(points - com, patch_forces).sum(axis=0),
                                 com, relative)

    def apply(self, data, result):
        """Assign this model's body-COM forces and torques, replacing old loads."""
        data.xfrc_applied[:, :3] = result.body_forces
        data.xfrc_applied[:, 3:] = result.body_torques

    def metadata(self):
        return {"model": "windward normal-pressure surface integration",
                "pressure_law": "p_i=0.5*rho*Cp*max(0,-(w_i-v_i).n_i)^2; dF_i=-p_i*A_i*n_i",
                "pressure_coefficient": self.pressure_coefficient,
                "air_density_kg_m3": self.wind_config.air_density_kg_m3,
                "surface_area_m2": self.surface_area_m2,
                "source_triangles": self.source_triangle_count,
                "degenerate_triangles_skipped": self.degenerate_triangle_count,
                "quadrature_patches": len(self.areas),
                "quadrature": "all triangle centroids" if len(self.areas) == self.source_triangle_count - self.degenerate_triangle_count else "deterministic area-stratified midpoint sampling per visual geom; exact total area",
                "visual_mesh_geoms": len(self.geom_ids),
                "globally_reversed_geom_ids": self.flipped_meshes,
                "force_frame": "world-aligned test-patch axes",
                "body_torque_origin": "each body center of mass (xipos), for xfrc_applied",
                "total_torque_origin": "whole-robot center of mass",
                "legacy_projected_area_used": False,
                "limitations": ["not CFD", "uncalibrated pressure coefficient", "no wake or shielding; overlapping/interior visual shells can overestimate exposed load", "no suction or skin friction", "prescribed flow does not react to the robot"]}
