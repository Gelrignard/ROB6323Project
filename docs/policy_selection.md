# Walking policy choice

Source review: 2026-11-15. These are upstream options, not checkpoints already
integrated or validated in this repository.

## Recommendation for the course project

Start with a **velocity-command PPO policy for the 29-DoF Unitree G1**.
Use one frozen walking checkpoint as the primary baseline, then fine-tune a
copy under randomized joint torques and link wrenches. Keep the command,
observation interface, robot, low-level controller and evaluation scenarios
identical. This isolates the benefit of robustness training.

[Unitree RL Mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) is our
preferred upstream starting point because it supplies G1 velocity tracking,
MuJoCo-based simulation, PPO through RSL-RL, and a deployment pipeline.
Its documented flat-terrain task is `Unitree-G1-Flat`. The repository also
contains rough-terrain configuration code; terrain generalization should be
a separate experiment after flat-ground disturbance rejection works.

The upstream tree includes a
[G1 velocity ONNX policy](https://github.com/unitreerobotics/unitree_rl_mjlab/tree/main/deploy/robots/g1/config/policy/velocity/v0)
and its deployment parameters. **An exported ONNX actor is not a resumable
PPO training checkpoint.** We did not find a `.pt`/`.pth` training checkpoint
in that repository's main tree during this review. Obtain a corresponding
RSL-RL training checkpoint and its configuration, or first train the nominal
walking policy in that upstream stack. Validate calm-ground walking before
starting disturbance fine-tuning. Availability outside the main tree has not
been established.

[Unitree RL Lab](https://github.com/unitreerobotics/unitree_rl_lab) is an
alternative if the team's Linux GPU machine already runs Isaac Lab. Its G1
velocity task avoids introducing whole-body motion-reference training merely
to study walking robustness. Choose between these two stacks based on the
training machine and a reproducible checkpoint, not on simulator branding.

## Where SONIC fits

[SONIC / GR00T WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl)
is a serious option for later rescue behaviors involving coordinated
whole-body motion. It is not inference-only: the official project released
training code and PyTorch checkpoints in April 2026, and documents
[PPO training and fine-tuning](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/docs/source/user_guide/training.md).
Released models are available from
[NVIDIA GEAR-SONIC](https://huggingface.co/nvidia/GEAR-SONIC).

However, SONIC is a motion-conditioned whole-body tracking system. Motion
references, the reference encoder/planner, policy history/normalization,
joint conventions, and the Isaac Lab training stack become part of the
experiment. That is more setup than a velocity-command robustness study.
Published large-scale training resources are not a demonstrated minimum
for a small fine-tuning run, but we have not benchmarked such a run on the
team's hardware. The local Windows CPU smoke tests do not establish its
training feasibility.

For a one-month project with one primary implementer, use SONIC as the
stretch option unless a teammate already has its complete training and
simulation pipeline working. Complex-terrain capability and wind robustness
must each be measured; neither follows merely from selecting SONIC.

## Interface boundary in this repository

`sim_env.rl_env.G1RobustEnv` currently defines a **project-native 99-value
observation / 29-action interface**. `rl.train` is a functional SB3 PPO
training/fine-tuning harness for checkpoints created with that contract.
Neither SONIC nor Unitree RSL-RL weights can be passed directly to it.

For the selected upstream policy, preserve its original actor observations,
history, normalization, joint names/order, nominal pose, action scales,
control rate, PD gains and actuator limits. Retain privileged critic inputs
only in training. Port the disturbance model into the upstream training
environment and implement an explicitly tested inference adapter for this
CPU evaluation environment. The upstream Unitree ONNX configuration, for
example, includes gait phase and different gains/action scales; matching
the number of joints is insufficient.

The current SB3 smoke examples verify the simulator and checkpoint plumbing.
They do not supply a pretrained walking policy or demonstrate robustness.

## Comparison and claims

The mandatory comparison is **frozen nominal PPO vs disturbance-fine-tuned
PPO**. Add DIAL-MPC after a G1 controller works under the same dynamics and
task; see [the integration status](dial_mpc.md). Do not present an H1 result
as a G1 baseline. Keep held-out wind fields/seeds out of checkpoint selection.

This project can demonstrate simulation robustness and preparation for
sim-to-real transfer. A sim-to-real robustness claim requires hardware
evidence. Flat-ground wind tests do not establish rescue-terrain capability.
