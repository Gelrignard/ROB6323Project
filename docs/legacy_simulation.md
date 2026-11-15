# Unitree G1 wind-control course project

Environment, physics, wind, robot assets, and rendering are in **sim_env/**.
The homework-style Riccati solver and LQR controller are in **lqr/**.
The default displayed robot is the authentic Unitree G1 29-DOF model.

## Run

From the project directory using the installed Python environment:

```powershell
# Default: G1-parameterized LQR trajectory, shown on G1 using inverse kinematics.
.\.venv\Scripts\python.exe -m lqr --viewer

# Generate the full 16-second G1 replay, snapshot, plots and logs.
.\.venv\Scripts\python.exe -m lqr --video --snapshot

# Real floating-base G1 physics with foot contacts; hold the vendor standing pose.
.\.venv\Scripts\python.exe -m sim_env.g1_physics --duration 2 --wind-scale 0 --viewer

# Integrate surface drag on G1 links during a physical standing test.
.\.venv\Scripts\python.exe -m sim_env.g1_physics --duration 2 --wind-scale 1 --viewer

# Regenerate the 100 m map and 10 m test-region images.
.\.venv\Scripts\python.exe -m sim_env.plot_field

# Proposal figure: black vectors on white; arrow length proportional to speed.
.\.venv\Scripts\python.exe -m sim_env.plot_field --style arrows-bw

.\.venv\Scripts\python.exe -m unittest discover -v
```

The two simulation modes have different purposes:

| Entry point | What evolves | What is verified |
| --- | --- | --- |
| `python -m lqr` | Dynamic reduced CoM; optional articulated G1 IK replay | LQR tracking and reduced support feasibility; IK reachability is checked only for rendered/replayed samples |
| `python -m sim_env.g1_physics` | Full floating-base G1, joint actuators, gravity and physical foot contacts | Model loading, standing integration, and response to wind |

**Whole-body walking control remains to be connected.** The current LQR acts
on CoP, not G1 motor commands. The walking video is explicitly labeled
kinematic replay. The physics command holds joint-position stand targets;
it is not a walking or wind-balance controller. This separation prevents a
rendered gait from being mistaken for a validated physical gait.

## G1 assets and geometry

The unmodified G1 MJCF, 35 STL meshes, and license are vendored in
`sim_env/models/unitree_g1/` from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1)
at commit `370c43965323c0944b433b202aac5058b5f907a9`.
`ASSET_MANIFEST.json` records upstream file hashes; `fetch_model.py` can
reproduce the download. `MODEL_PROFILE.json` documents the original model.

The local loader adds the test floor and sole frames while preserving the
floating base, inertias, collision geometry, actuator force limits and joint
limits. Native actuators accept **29 joint-position targets in radians**,
not direct torque commands. No Unitree SDK is needed.

Measured G1 mass is 33.341142 kg. Grounded standing CoM height is approximately
0.693716 m. The reduced walking preset uses 0.653716 m, leaving leg-bending
room while respecting ankle limits throughout the gait. Foot spacing is
approximately 0.237013 m; conservative sole half-dimensions are 0.085 m by
0.025 m. Reduced friction coefficient is 0.6, matching the physical foot
model. The legacy lumped model assumes projected area 0.25 square metres
and drag coefficient 1. The new surface model obtains areas from the mesh
and uses a separate, uncalibrated pressure coefficient `Cp=1`.

`G1Replay` solves for base translation and all 12 leg joints to match both
flat sole poses and the **actual whole-robot CoM**, not the pelvis position.
It retains upper-body posture, enforces joint limits, and rejects excessive
IK errors. It uses `mj_forward`; it never integrates or claims full-body
walking dynamics. Saved joint poses are marked `kinematic_only`.
The full video checks 400 samples. A surface-profile snapshot also evaluates
the full sampled gait to build its history and estimate moving-link velocities;
a lumped snapshot checks one pose. A run without media or the viewer does not
execute IK.

## Wind field and coordinates

The map spans `[0,100] x [0,100]` metres with vortex center `(50,50)` m.
The walking patch is **world x=[68,78], y=[45,55]** m:

```text
world_position = local_position + (68,45)
```

The nominal local walking segment is `(1,5)` to `(5.55,5)`. The smooth,
finite-core horizontal velocity field is

```text
w(r) = 2a / (a^2 + ||r||^2) * (V * [-r_y,r_x] - U*r)
a = 12 m, V = 12 m/s, U = 1 m/s
F_w = 0.5 * rho * Cd * area * ||w-v|| * (w-v)
rho = 1.225 kg/m^3
```

This is a prescribed synthetic vortex, not CFD or a calibrated tornado.
There is no resolved turbulence, vertical airflow, or robot-flow coupling.
Horizontal inflow implies omitted vertical flow. The unscaled test-patch
wind is approximately 8.6-11.1 m/s.

The **G1 LQR starter run defaults to wind scale 0.5**, approximately 4.3-5.6
m/s in the patch, because G1 has much narrower feet than the original
placeholder. Set `--wind-scale 1` to use the full plotted field. Scale
multiplies velocity, not force. A failed run is recorded and exits with code
1; it is not replaced with an animation that hides the failure.

The reduced model applies a lumped horizontal force at overall CoM and omits
external moments. The physical G1 CLI defaults to integrated surface loads
applied to each link; `--drag-model lumped` restores the single torso load.
Python `G1PhysicsEnv` and `run_standing` retain their legacy lumped default,
so select `drag_model="surface"` explicitly when calling them directly.

## Drag-only visualization and surface integration

`sim_env/visualize/` now displays **only aerodynamic drag**: integrated
Fx/Fy/Fz, magnitude history, and a surface-load profile with height. Ground
reaction, gravity, normal-force, and total external-force plots are removed.
Surface arrows show local drag vectors: arrowheads touch the loaded surface
and tails extend outward, so their direction points into the robot. Arrow
length is proportional to the sampled patch force; color shows modeled
pressure in Pa. A larger coral arrow shows the integrated force in N.
Local and resultant arrows use separate fixed scales, labeled in the sidebar.
The robot is rendered at 50% opacity in drag views. Surface-vector display
subsampling does not affect the full force integration. The height profile sums local
force magnitudes within each bin, so its sum can exceed the magnitude of
the resultant when components cancel. Pressure and arrow scales are fixed
throughout a clip and recorded in `force_visualization.json`.

```powershell
# Default G1 video: drag-only surface diagnostic on the prescribed gait.
.\.venv\Scripts\python.exe -m lqr --video --snapshot
# Earlier single-resultant model, still with a drag-only overlay.
.\.venv\Scripts\python.exe -m lqr --video --snapshot --drag-profile lumped --output sim_env/outputs/g1_lumped_drag
# Surface forces actually coupled to physical G1 joint/contact dynamics.
.\.venv\Scripts\python.exe -m sim_env.g1_physics --drag-model surface --wind-scale 1 --duration 2 --viewer
```

The surface algorithm in `sim_env/surface_drag.py` selects the 35 visual
mesh geometries once, excluding duplicate collision geometries. The meshes
contain 393,270 triangles. The default uses 4,096 deterministic area-weighted
quadrature samples, preserving total mesh area; `max_patches=None` evaluates
every nondegenerate triangle. Compiled mesh coordinates, normals and areas
are transformed with each moving link. At each surface sample:

```text
u_i = wind(position_i) - [v_body_COM + omega_body x (position_i - body_COM)]
s_i = max(0, -u_i dot outward_normal_i)
p_i = 0.5 * rho * Cp * s_i^2
dF_i = -p_i * area_i * outward_normal_i
F_body = sum(dF_i)
torque_body = sum((position_i - body_COM) x dF_i)
```

The physical environment applies each body's force and moment through
MuJoCo `xfrc_applied` at every 0.002 s step. It does not add the old lumped
drag on top. Summing link wrenches reproduces the force and moment of the
surface distribution. The displayed resultant arrow is placed at robot
CoM for reference; the saved moment is required to represent the full load.

**The walking surface profile is evaluated on prescribed G1 motion and is
not fed back into the reduced LQR dynamics.** Link velocities are estimated
from central differences of articulated poses, including angular motion.
`drag_profile.csv` compares surface and legacy lumped loads along that same
motion; `surface_drag_profile.npz` saves patch pressure, positions, forces,
body loads, total force/moment, and kinematic qpos/qvel. Physical standing
tests are the separate coupled experiments.

This is an uncalibrated windward normal-pressure approximation, not CFD or
resolved boundary-layer friction. It omits suction, skin friction, wakes,
shielding between links and fluid feedback; overlapping or internal visual
shells can overestimate exposure. Its full-vector force can have transverse
components. MuJoCo's own built-in fluid models also use geometric
approximations rather than resolving the flow; this implementation uses
mesh surface quadrature instead of those box/ellipsoid models.
See the [official fluid-force documentation](https://mujoco.readthedocs.io/en/stable/computation/fluid.html).

For the default 16 s prescribed gait at wind scale 0.5, mean resultant load
is 5.776 N from surface quadrature versus 4.125 N from the fixed-area model.
Five moving-pose checks against all 393,270 triangle centroids give at most
1.54% resultant-force error and 0.020 N·m absolute moment error for the 4,096
sample rule. These quantify numerical integration error, not physical model
accuracy; see `sim_env/outputs/g1_lqr/surface_drag_validation.json`.
Separate 2 s physical standing tests with surface drag completed in calm and
full-scale wind; the windy run reached 25.64 N and 1.68 N·m maximum load.

## LQR and experiment options

```text
state = [c_x, v_x, c_y, v_y], action = [CoP_x, CoP_y]
c_ddot = (g/h)*(c - CoP) + F_w/m
```

MuJoCo integrates the reduced model at 500 Hz; LQR updates at 50 Hz with a
1.6-second preview. The HW2 Riccati recursion is extended to affine wind
increments and reference tracking. Actual force is recalculated from the
simulated state. Known-wind preview is evaluated along the nominal path;
`--blind` removes it. See `lqr/README.md` for equations and policy signs.

Footsteps are prescribed: 0.65-second step period, 20% double support,
0.08 m swing clearance, twenty advancing steps and one closing step.
The nominal CoM path is derived from the LIP dynamics, not imposed on the
simulated state. CoP is projected onto the true support-polygon intersection
throughout each control hold interval. Projection is a baseline heuristic,
not constrained MPC. Friction, patch exit and a reduced reach proxy are
checked; they do not replace full-body stability validation.

```powershell
.\.venv\Scripts\python.exe -m lqr --wind-scale 0 --output sim_env/outputs/g1_calm
.\.venv\Scripts\python.exe -m lqr --wind-scale 1 --output sim_env/outputs/g1_full_wind
.\.venv\Scripts\python.exe -m lqr --blind --output sim_env/outputs/g1_blind
.\.venv\Scripts\python.exe -m lqr --duration 4 --horizon 1.0

# Original numerical/box diagnostic, retained for regression checks only.
.\.venv\Scripts\python.exe -m lqr --robot lip --output sim_env/outputs/lqr_lip
```

## Outputs and source layout

- `sim_env/assets/tornado_field.png` and `.svg`: full map.
- `sim_env/assets/walking_test_region.png` and `.svg`: selected test patch.
- `sim_env/outputs/g1_lqr/walking.mp4` and `simulation_preview.png`: G1 replay.
- `g1_ik_diagnostics.json` and `g1_kinematic_poses.npz`: explicit mapping errors
  and articulated kinematic poses.
- Each reduced run saves `summary.json`, `trajectory.csv`, `trajectory.npz`,
  and `tracking.png`. States, controls, and forces use interval-start times;
  the final state is in the summary. Effort is a force-squared integral, not
  physical/electrical energy.
- Physical standing runs save `summary.json` and `physical_state.npz`.
  Wind forces and moments have their own physics-step timestamps, distinct
  from the less frequent qpos/qvel samples.
- Surface-profile media additionally save `drag_profile.csv`,
  `surface_drag_profile.npz`, and `surface_drag_summary.json`.
- Older `lqr_wind`/`lqr_calm` outputs retain the original reduced-box baseline.

```text
sim_env/
  config.py, wind.py, plot_field.py     Map, drag, frames, images
  gait.py, environment.py              Reduced walking physics and references
  g1_model.py                         Authentic G1 scene and measured profile
  g1_physics.py                       Real contact dynamics and joint-target API
  surface_drag.py                     Mesh quadrature and distributed aerodynamic wrenches
  g1_replay.py                        Joint-limited CoM/feet inverse kinematics
  run.py, render.py                   Experiments, logs, plots, video, viewer
  visualize/                         Drag-only video plots and prescribed-gait surface profiling
  models/unitree_g1/                  Pinned G1 assets and license
  tests/                             Environment, geometry, physics and IK checks
lqr/
  riccati.py, controller.py           Homework-style solver and CoP controller
  __main__.py, tests/                 CLI and independent numerical checks
```

To recreate the Python 3.12 environment:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

The tests compare LQR with an independent dense KKT solve, verify reduced
integration and feedback, and check actual G1 free-base/contact dynamics,
wind response, joint bounds, and IK Jacobians and full-gait reachability.
The next algorithmic layer is whole-body tracking of the CoM and footsteps
in the physical G1 environment.
