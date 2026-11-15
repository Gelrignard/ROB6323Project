# Local validation, 2026-11-15

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
- Vendored upstream asset hashes were checked against the Git index. Generated
  logs, policies, videos and evaluation results are excluded from commits.

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
training/convergence, long-horizon disturbance robustness, G1 DIAL-MPC,
rough terrain, GPU training throughput, aerodynamic fidelity and hardware
transfer. These are not implied by passing the integration checks above.
