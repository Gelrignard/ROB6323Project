"""Evaluate mesh drag on a prescribed articulated gait, independently of LQR.

This diagnostic does not replace the reduced simulation's wind loads. Physical
coupling is provided by G1PhysicsEnv(drag_model='surface'). Replay velocities
are obtained from pose differences; treating all moving links as stationary
would give incorrect relative wind.
"""

import csv
import json
from pathlib import Path

import mujoco
import numpy as np

from ..surface_drag import SurfaceDrag


def pose_velocities(model, times, poses):
    """Central differences on MuJoCo configuration space; one-sided at ends."""
    times, poses = np.asarray(times, dtype=float), np.asarray(poses, dtype=float)
    if times.ndim != 1 or poses.shape != (len(times), model.nq) or len(times) == 0:
        raise ValueError("times/poses must have shapes (N,) and (N,nq), N>0")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(poses)) or np.any(np.diff(times) <= 0):
        raise ValueError("finite poses and strictly increasing times required")
    velocities = np.zeros((len(times), model.nv))
    if len(times) == 1:
        return velocities
    for i in range(len(times)):
        a, b = max(0, i - 1), min(len(times) - 1, i + 1)
        mujoco.mj_differentiatePos(model, velocities[i], times[b] - times[a], poses[a], poses[b])
    return velocities


def prepare_surface_profile(replay, records, wind_config, wind_scale, output):
    """Return decorated records and replay poses, and save auditable forces.

    Caller supplies the video sampling times. Surface force/torque and the
    original lumped force are compared on this same prescribed gait, not on
    two independently controlled dynamic trajectories.
    """
    times = np.array([row["time"] for row in records])
    poses = []
    for row in records:
        replay.set_state(row["state"], row["feet"], row["time"], row["cop"])
        poses.append(replay.data.qpos.copy())
    poses = np.array(poses)
    velocities = pose_velocities(replay.model, times, poses)
    surface = SurfaceDrag(replay.model, wind_config=wind_config)
    prepared, totals, moments, body_forces, body_torques = [], [], [], [], []
    pressures, points, patch_forces = [], [], []
    last_result = None
    for row, qpos, qvel in zip(records, poses, velocities):
        replay.data.qpos[:] = qpos
        replay.data.qvel[:] = qvel
        replay.data.time = row["time"]
        mujoco.mj_forward(replay.model, replay.data)
        result = surface.evaluate(replay.data, wind_scale=wind_scale)
        prepared.append({**row, "drag_model": "surface", "drag_force": result.total_force.copy(),
                         "drag_torque": result.total_torque_com.copy(), "drag_origin": result.com.copy(),
                         "surface_drag": {"points": result.points.copy(),
                                          "forces": result.patch_forces.copy(),
                                          "pressure": result.pressure_Pa.copy()}})
        totals.append(result.total_force.copy())
        moments.append(result.total_torque_com.copy())
        body_forces.append(result.body_forces.copy())
        body_torques.append(result.body_torques.copy())
        # float32 storage is sufficient for display; integration uses float64.
        pressures.append(result.pressure_Pa.astype(np.float32))
        points.append(result.points.astype(np.float32))
        patch_forces.append(result.patch_forces.astype(np.float32))
        last_result = result
    total, torque = np.asarray(totals), np.asarray(moments)
    lumped = np.array([[*row["wind_force"], 0.0] for row in records])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "surface_drag_profile.npz", time=times,
                        force=total, torque_about_com=torque, lumped_force=lumped,
                        body_forces=body_forces, body_torques_about_body_com=body_torques,
                        points=points, patch_forces=patch_forces, pressure_Pa=pressures,
                        patch_areas=last_result.areas, body_ids=last_result.body_ids,
                        qpos=poses, qvel=velocities, kinematic_only=True)
    with (output / "drag_profile.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", "surface_Fx_N", "surface_Fy_N", "surface_Fz_N",
                         "surface_magnitude_N", "Mx_about_com_Nm", "My_about_com_Nm", "Mz_about_com_Nm",
                         "lumped_Fx_N", "lumped_Fy_N", "lumped_Fz_N", "lumped_magnitude_N"])
        for t, force, moment, old_force in zip(times, total, torque, lumped):
            writer.writerow([t, *force, np.linalg.norm(force), *moment, *old_force, np.linalg.norm(old_force)])
    report = {"scope": "surface loads evaluated on prescribed G1 IK gait; NOT coupled to reduced LQR dynamics",
              "velocity_method": "MuJoCo pose central differences at video sample times; one-sided endpoints",
              "sample_count": len(times), "wind_scale": wind_scale,
              "surface_mean_force_magnitude_N": float(np.mean(np.linalg.norm(total, axis=1))),
              "surface_max_force_magnitude_N": float(np.max(np.linalg.norm(total, axis=1))),
              "surface_max_torque_magnitude_Nm": float(np.max(np.linalg.norm(torque, axis=1))),
              "lumped_mean_force_magnitude_N": float(np.mean(np.linalg.norm(lumped, axis=1))),
              "lumped_effective_area_m2": wind_config.projected_area_m2,
              "surface_model": surface.metadata()}
    (output / "surface_drag_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Surface drag evaluated on {len(times)} prescribed poses: "
          f"mean |F| {report['surface_mean_force_magnitude_N']:.3f} N", flush=True)
    return prepared, poses, velocities
