# G1 walking robustness under disturbances and wind

ROB6323 Optimal Control & Reinforcement Learning course project: physical G1
walking under joint/link disturbances and wind. The current executable
controller is a G1 port of **DIAL-MPC**, with no learned policy required.
Selecting and fine-tuning an RL walking policy remains a separate experiment.

## Current status

- Actual floating-base **Unitree G1, 29 actuators**, MuJoCo contacts and a
  10 m x 10 m flat test patch. Actions move physical joints.
- Gymnasium task, seeded disturbance schedules, training-only domain
  randomization, SB3 PPO checkpoint continuation and paired-seed evaluation.
- Surface-integrated aerodynamic loads, spatial/time-dependent winds, and
  the existing drag-only visualization with 50% robot transparency.
- **A pretrained walking policy has not yet been selected or integrated.**
  The PPO harness uses a project-native interface. Smoke training verifies
  execution, not walking ability. SONIC and Unitree RSL-RL weights require
  their original training stack and a tested adapter.
- **DIAL-MPC is implemented in `mpc/`.** It uses dual annealing, native
  full-order MuJoCo rollouts, and a G1-specific gait objective. The CPU
  demonstration runs synchronously offline; it does not claim real-time
  performance. [Method and port details](docs/dial_mpc.md).

The earlier [policy-selection notes](docs/policy_selection.md) document
candidate frameworks; no checkpoint choice is final. Current terrain is flat.
Neither rough-terrain performance nor hardware transfer has been established.
The [experiment protocol](docs/experiment_protocol.md) separates the required
PPO comparison, optional ablations and later research claims.
See [validation results](docs/validation.md) for the earlier RL integration
checks. The expanded suite now passes 90 tests, including DIAL optimizer,
full-dynamics prediction parity, gait references and physical recording.

## DIAL-MPC physical walking demonstrations

[Watch the three-case comparison](mpc/assets/comparison.mp4).
Individual videos: [calm](mpc/assets/calm.mp4),
[random disturbances](mpc/assets/random_disturbances.mp4),
[wind field](mpc/assets/wind_field.mp4).

![Physical G1 DIAL-MPC comparison](mpc/assets/comparison.png)

Measured with the same controller, initial state and seed (7):

| Case | Duration | Forward travel | Walking check |
| --- | --- | --- | --- |
| Calm | 4 s | 0.509 m | Passed |
| Six random joint/link pulses | 4 s | 0.502 m | Passed |
| Full configured vortex | 4 s | 0.495 m | Passed |

Peak injected link force was 27.57 N; peak modeled aerodynamic resultant in
the vortex was 26.48 N. These are short single-seed demonstrations. Planning
took approximately 1.28-1.45 s per 0.02 s control interval, so the videos play
simulated time rather than wall time. [Machine-readable results](mpc/results/g1_demo.json)
include configurations, event schedules, foot contacts, errors and timing.

The DIAL code uses only `requirements.txt`; no Torch, JAX or RL checkpoint is
needed. After installing those dependencies, run:

```powershell
.\.venv\Scripts\python.exe -m mpc --cases calm random tornado --duration 4 --controller-config mpc/configs/g1_demo.json --wind-scale 1 --disturbance-strength 0.2 --output mpc/outputs/demo --video
.\.venv\Scripts\python.exe -m mpc.compare mpc/outputs/demo/calm mpc/outputs/demo/random mpc/outputs/demo/tornado --output mpc/outputs/demo/comparison.mp4
```

Allow several minutes per case on a CPU. `--video` renders the actual
integrated trajectory with 50% robot opacity and aerodynamic drag vectors.
Random injected loads are separately captioned. Each case records the exact
compiled model, states, physical loads, contact events, planning latency and
failure status; existing output directories are never overwritten.

The demo config uses 64 sampled candidates per refinement, three refinements,
and a 0.4 s horizon. It optimizes the 12 leg-joint references around an IK gait
prior while integrating the entire floating-base 29-actuator G1. Upper-body
targets hold their reference posture. The prior is never imposed on the
physical state. This is an explicit G1 adaptation of the paper's optimizer,
not a reproduction of the upstream H1 benchmark or a kinematic walking replay.

`wind_prediction="current_wrench"` holds the currently available aerodynamic
link wrenches constant during prediction. This is ideal current-load access,
not a future-wind oracle or a validated real-robot estimator. Future random
disturbance events are never supplied to the planner. At disturbance strength
0.2, random peak limits are 30 N link force, 4 N m link torque and 6 N m joint
torque; the actual sampled event schedule is saved with every run.

## Install and verify

Python **3.12.14** is the tested interpreter, pinned in `.python-version`.
From the repository root, using [uv](https://docs.astral.sh/uv/):

```powershell
uv python install 3.12.14
uv venv --python 3.12.14 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements-rl.txt
.\.venv\Scripts\python.exe -m unittest discover -v
```

Alternatively create a venv with an installed Python 3.12.14 and use pip.
Avoid Python 3.12.0: local testing reproduced a nested ZIP reader failure
when reloading SB3 checkpoints, resolved with 3.12.14. A save/load regression
test covers this path. On Linux/macOS use `.venv/bin/python` in the commands.
The CPU environment needs no Unitree
SDK. `requirements.txt` alone installs simulation/visualization dependencies.
The selected upstream GPU training framework will need a separate install.

These short runs verify training, saving/loading and continuation. They do
**not** produce a competent walking policy. Use new or empty output folders:

```powershell
.\.venv\Scripts\python.exe -m rl.train --from-scratch --timesteps 512 --n-steps 128 --batch-size 64 --epochs 2 --output rl/runs/smoke
.\.venv\Scripts\python.exe -m rl.train --checkpoint rl/runs/smoke --timesteps 256 --n-steps 128 --batch-size 64 --epochs 2 --output rl/runs/smoke_continued
.\.venv\Scripts\python.exe -m rl.evaluate --checkpoint rl/runs/smoke --checkpoint rl/runs/smoke_continued --stand --scenarios calm tornado --episodes 1 --duration 2 --output rl/evaluations/smoke
```

For a **compatible, validated walking checkpoint**, begin fine-tuning at low
disturbance strength, then increase it while checking nominal walking:

```powershell
.\.venv\Scripts\python.exe -m rl.train --checkpoint rl/runs/walking_baseline --preset train_random --curriculum-strength 0.1 --timesteps 1000000 --output rl/runs/robust_stage1
```

`--from-scratch` must be explicit. Continuation loads PPO weights/optimizer
state and observation normalization, leaving the source bundle unchanged.
Bundles contain `model.zip`, `vecnormalize.pkl` when used, and `manifest.json`
with interface, hashes, settings, seed, versions and robot provenance.
Incompatible or incomplete bundles fail before training starts.

## Environment and loads

All environment and physics code lives in `sim_env/`:

```python
import numpy as np
from sim_env.rl_config import RLConfig
from sim_env.rl_env import G1RobustEnv

env = G1RobustEnv(RLConfig.preset("train_random"))
observation, info = env.reset(seed=42)
observation, reward, terminated, truncated, info = env.step(
    np.zeros(env.action_space.shape, dtype=np.float32)
)
env.close()
```

The **99-value observation** contains body-frame base linear/angular velocity,
projected gravity, commanded velocity/yaw rate, joint positions relative to
nominal, joint velocities and the previous requested action. Wind and
disturbance labels are excluded. Base velocity comes from simulation; real
deployment needs an estimator. The **29 normalized actions** specify joint
position offsets. `env.schema()` records exact joint order, nominal targets,
scales and limits. Physics runs at 500 Hz; control at 50 Hz.

Training randomizes mass/inertia, friction, actuator gains, sensor noise and
action delay. Nominal model parameters are restored at reset. The manual
`curriculum_strength` setting in [0,1] scales randomization and disturbances;
it does not automatically advance based on performance. Evaluation disables
training domain/noise randomization.

Schedules are sampled once at reset with independent RNG streams. The same
seed produces the same events regardless of policy actions or survival.
Default full-strength half-sine pulses last 0.1-0.35 s, with peaks up to
150 N link force, 20 N m link torque or 30 N m joint torque. These are
configurable stress-test bounds, not calibrated tornado loads. Link wrenches
use world axes at link CoM; joint torques act only on actuated hinge DOFs.
Loads are additive with wind and applied at every physics step. They expire
without resetting velocity.

`train_random` disables aerodynamics (`drag_model="none"`) for throughput.
Wind evaluation integrates surface drag. Calm air still resists moving
surfaces, so it differs physically from disabling drag.

## Evaluation protocol

```powershell
.\.venv\Scripts\python.exe -m rl.evaluate --checkpoint rl/runs/walking_baseline --checkpoint rl/runs/robust_stage1 --episodes 20 --seed 10000 --save-rollouts --output rl/evaluations/comparison
```

The default suite includes calm, uniform, gust, original tornado, shifted
vortex, reversed vortex and a smaller core. `--scenarios` also accepts
`shear`, `moving_vortex` and `random_push`. Controllers receive identical
scenario configurations and seed lists; evaluation normalization is frozen.
Small initial joint offsets (0.005 rad standard deviation by default) vary
across seeds, with identical initial states for each compared controller.
These trials measure initial-state uncertainty, not a distribution of all
possible storms. Fields use the declared scenario parameters. Keep training,
validation and final-test seeds separate. Use `--config` to select a common
RLConfig JSON; otherwise the first checkpoint supplies the common interface.
Held-out scenarios use full perturbation strength regardless of the training
curriculum stage.

Reports measure survival, failure, progress, velocity/yaw tracking error,
walking success and controller computation time. Success requires finishing
without failure, at least 60% of commanded travel, velocity RMSE below
0.25 m/s and yaw-rate RMSE below 0.5 rad/s. Standing still does not pass.
These are initial project thresholds, not a robotics benchmark standard.

Outputs: `results.json` with settings/event schedules, `episodes.csv`,
`summary.csv`, and optional NPZ states/actions. External controllers use
`--plugin module:factory` and declare their information access. Planning
time excludes physics/rendering; offline success does not prove real-time
operation. See [DIAL-MPC integration](docs/dial_mpc.md).

## Wind and visualization

![Black-and-white wind velocity field](sim_env/assets/tornado_vector_field_bw.png)

The map spans 100 m x 100 m. The test patch is at world
`x=[68,78], y=[45,55]`; local positions are translated by `(68,45)`.
Arrow length is proportional to speed.

```powershell
.\.venv\Scripts\python.exe -m sim_env.plot_field --style arrows-bw
.\.venv\Scripts\python.exe -m sim_env.g1_physics --duration 2 --wind-scale 1 --drag-model surface --viewer
.\.venv\Scripts\python.exe -m rl.render --stand --scenario tornado --duration 2 --output rl/evaluations/tornado.mp4
```

`rl.render` records actual physical motion and draws aerodynamic vectors with
50% robot opacity. Replace `--stand` with `--checkpoint <bundle>` to render a
compatible PPO policy. MP4/JSON/NPZ outputs preserve physical states, forces
and experiment settings; injected joint/link perturbations are not mislabeled
as drag. Existing outputs are never overwritten.

Surface drag integrates windward normal-pressure loads over moving G1
meshes, with 4,096 area-weighted samples by default. Relative wind includes
each sample's translational/rotational velocity; forces and moments act on
each link. This is an **uncalibrated aerodynamic approximation, not CFD**:
no wake shielding, suction, resolved turbulence or fluid feedback. The
synthetic vortex is not a measured tornado.

The command above is a physical standing diagnostic. Earlier LQR walking
videos are kinematic replays. Their commands and the full drag visualization
method are preserved in [legacy simulation notes](docs/legacy_simulation.md).

## Source layout and provenance

| Path | Purpose |
| --- | --- |
| `sim_env/rl_env.py`, `rl_config.py` | Full-body Gymnasium task/settings |
| `sim_env/disturbances.py` | Reproducible joint/link perturbations |
| `sim_env/wind_fields.py`, `surface_drag.py` | Wind and distributed loads |
| `sim_env/visualize/` | Existing drag-only rendering tools |
| `rl/` | PPO, checkpoint contracts, paired evaluation |
| `docs/` | Policy selection and baseline integration |
| `lqr/` | Preserved reduced-model homework experiment; outside current comparison |

G1 assets and upstream license are vendored from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/370c43965323c0944b433b202aac5058b5f907a9/unitree_g1).
`sim_env/models/unitree_g1/ASSET_MANIFEST.json` records source hashes.
Generated runs, logs, videos and checkpoints are ignored by Git. Required
robot assets and proposal figures are included.
