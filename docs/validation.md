# Local validation, 2026-11-15

The current DIAL-MPC baseline excludes all external-load information. Its
only runtime input is `RobotObservation(time, qpos, qvel)`. The controller
retains no environment object, and its predictions use zero external loads
and zero plant-solver warm starts. Physical drag, injected loads, logging
and force visualization remain active in the environment.

The full test suite passes **92 tests in 79.68 s**. A force-poisoning
regression keeps the allowed robot observation and planner seed fixed while
replacing simulator force/torque buffers with extreme values, invalidating
solver warm starts, and making wind/event providers raise on access. The
controller returns identical actions and rollout states, uses zero prediction
warm starts, and retains no environment reference. Additional checks ensure
the observation reads only time/qpos/qvel and the former privileged-load mode
is rejected.

Three new physical trials use the same 64 candidates, three refinement
iterations, requested 4-second episodes and seed 7 across conditions. Their
prediction horizon is 0.4 s. **The unknown-wind trial failed after 2.40 s;
the calm and random-pulse trials completed their declared walking checks.**

| Case | Executed time | Forward travel | Planar velocity RMSE | Outcome |
| --- | --- | --- | --- | --- |
| Calm | 4.00 s | 0.502 m | 0.245 m/s | Completed, walking check passed |
| Six random joint/link pulses | 4.00 s | 0.510 m | 0.230 m/s | Completed, walking check passed |
| Full configured vortex | 2.40 s | 0.109 m | 0.880 m/s | Pelvis-height termination |

The wind trial reached pelvis height 0.440 m, below the 0.45 m threshold;
the tilt threshold was not crossed. Lateral drift was 1.337 m and peak drag
was 25.58 N. Initial robot state, controller settings and seed were held
fixed across the three cases, and the random event schedule matches the
earlier demonstration. The
[result bundle](../mpc/results/g1_unknown_disturbances.json) and
[comparison video](../mpc/assets/unknown_disturbances/comparison.mp4) preserve
the new outcomes, including the wind failure. Mean planning times were
1.19-1.32 s per 0.02 s control interval, so these are offline CPU results.
This short, single-seed failure is evidence about the tested G1 configuration
and planning budget, not a general failure claim about the DIAL-MPC method.

## Additional large-impact condition

The unchanged controller was run once more at seed 7 for a requested 4 s,
with disturbance strength 1.0 instead of 0.2. Event times, kinds, targets,
durations and directions match; every prescribed amplitude is fivefold.
The recorded positions, velocities and actions before the first impact
match the smaller-impact trial bit for bit.

The large trial terminated at **3.40 s**, with pelvis height 0.431 m and
upright cosine 0.403, crossing both thresholds (0.45 m and 0.5). Forward
displacement was 1.954 m, lateral drift -0.807 m, and planar velocity RMSE
0.871 m/s; the large displacement is not evidence of successful walking.
Actual recorded peak injected loads were 137.835 N link force, 15.609 N m
link torque and 17.267 N m joint torque. The last joint-torque pulse was
cut short by termination, so its prescribed full-pulse peak was not reached.

Raw data and video are in `outputs/dial_mpc/large_impacts/random/`.
The four-case comparison and results are under
`outputs/dial_mpc/unknown_disturbances/comparison_2x2*`; all remain Git-ignored.
The previous calm, smaller-impact and wind trajectories are reused.
Arrows remain aerodynamic only and are scaled per source clip. Recorded
impact targets and magnitudes are shown separately; failed runs retain their
actual final frame and stop time, with no future pulse shown as applied.
The focused MPC suite passes **23 tests**, including five comparison tests
covering three/four-case layouts, recorded load magnitudes and units, missing
load data, and suppression of future or post-termination events.

The earlier DIAL implementation passed **90 total tests** and completed
three physical 4-second trials with privileged current aerodynamic-wrench
access. The [historical result bundle](../mpc/results/g1_demo.json) and
[historical comparison video](../mpc/assets/comparison.mp4) retain that
experiment's measurements. They are **not unknown-disturbance results**.
Those videos were decoded and their rendered frames inspected; physics was
never replaced by the reference trajectory. The earlier RL checks below
also retain their original scope.

Validated on Windows with CPython 3.12.14, MuJoCo 3.17.0, Gymnasium 1.4.0,
Stable-Baselines3 2.9.2 and PyTorch 2.9.1 CPU. This validates software behavior,
not learned walking competence or sim-to-real performance.

- `python -m unittest discover -q`: **74 tests passed**. Coverage includes
  physical force application, wind frames, surface integration, reproducible
  schedules/resets, Gymnasium API, action limits, failure handling, checkpoint
  identity/normalization, save/load, and exclusion of future disturbance data.
- PPO scratch smoke: 512 transitions, followed by 256 additional transitions
  using two subprocess environments. The continued checkpoint reports 768
  total transitions, with optimizer progress retained. The source model and
  normalization file still match their original manifest hashes.
- All ten evaluation scenarios ran with a standing diagnostic for two
  2-second episodes each. All were correctly unsuccessful at walking.
- Paired evaluation ran scratch PPO, continued PPO and standing on calm and
  tornado fields, with two seeds and a 2-second horizon: 12 episodes total.
  Initial states and event schedules matched across controllers. None of
  these smoke policies achieved walking success.
- A physical tornado rollout was encoded to H.264 video, decoded and visually
  inspected. Local drag arrows, resultant force and 50% robot opacity were
  present; the caption explicitly identifies a standing diagnostic.
- Vendored upstream asset hashes were checked against the Git index. Raw
  generated logs, policies and evaluation runs are excluded from commits;
  selected demonstration videos and compact result bundles are included.

## Reproduce the integration checks

Use the interpreter selected by `.python-version` and install
`requirements-rl.txt`. In the commands below, `python` means that virtual
environment's interpreter (`.\.venv\Scripts\python.exe` on Windows).
Choose new output folders on each run:

```powershell
python -m unittest discover -q
python -m rl.train --from-scratch --timesteps 512 --n-steps 128 --batch-size 64 --epochs 2 --output rl/runs/check_base
python -m rl.train --checkpoint rl/runs/check_base --timesteps 256 --n-envs 2 --subproc --n-steps 64 --batch-size 64 --epochs 2 --output rl/runs/check_resumed
python -m rl.evaluate --checkpoint rl/runs/check_base --checkpoint rl/runs/check_resumed --stand --scenarios calm tornado --episodes 2 --duration 2 --save-rollouts --output rl/evaluations/check_paired
python -m rl.render --stand --scenario tornado --duration 0.2 --output rl/evaluations/check_drag.mp4
```

The existing Python 3.12.0 interpreter failed to load SB3's nested checkpoint
archive despite valid ZIP CRCs. The same checkpoint loads under 3.12.14; the
repository now pins that tested runtime and includes a save/load regression.
No custom unsafe checkpoint loader or global patch was introduced.

## Still unvalidated

A selected external pretrained walking policy and its adapter, full PPO
training/convergence, long-horizon or multi-seed disturbance robustness,
rough terrain, GPU training throughput, aerodynamic fidelity and hardware
transfer. The tested G1 controller did not pass the full-strength unknown-wind
case. Simulator
base pose/velocity are still noiseless; a real robot needs a state estimator.
These capabilities are not implied by passing the integration checks above.
