# DIAL-MPC comparison: interface and current status

The experiment should compare the original walking PPO policy, the same policy
after disturbance fine-tuning, and a working DIAL-MPC controller on the same G1
plant and held-out scenarios. No DIAL-MPC score is currently reported by this
repository. `--stand` is only an integration diagnostic and is never labelled MPC.

The [official DIAL-MPC implementation](https://github.com/LeCAR-Lab/dial-mpc)
optimizes full-order legged control through sampled simulator rollouts. Its
documented humanoid example is Unitree H1, and the upstream recommended setup is
Ubuntu with CUDA. A G1 controller must be ported and validated; changing the robot
name in an H1 configuration is not an established baseline. The
[paper](https://arxiv.org/abs/2409.15610) describes the algorithm being compared.

## Common evaluation protocol

`python -m rl.evaluate` runs each selected controller with the same scenario
definitions, commanded velocity, episode horizon, and seed list. Test seeds
default to 10000 onward; do not reuse those seeds for training or tuning.
Outputs include episode metrics, grouped scores, exact environment
configurations, policy information access, and optional state/action rollouts.
Physical settings are inherited from the first PPO bundle; all other bundles
must match that common plant. Use `--config path.json` to set a shared base
configuration explicitly (for example for a plugin-only comparison).
Each scenario then overrides wind and disturbance settings, training
randomization, the straight walking command, and curriculum strength (always
1.0 in tests). A checkpoint from an early curriculum stage therefore does not
receive weaker evaluation pushes.
Success requires completing the horizon with forward progress and low tracking
error; merely surviving upright or receiving a large reward is insufficient.

```powershell
python -m rl.evaluate --checkpoint rl/runs/original --checkpoint rl/runs/robust --episodes 10 --save-rollouts
python -m rl.evaluate --plugin my_dial_adapter:create_controller --episodes 10 --save-rollouts
```

The currently supported plant uses 29 joint-position targets with the vendor's
force-limited PD actuators. PPO's normalized actions map to the exact joint
ordering, nominal angles, and action scale recorded in `env.schema()`. An MPC
controller must predict these same actuator dynamics. Raw torques must **not**
be relabelled as target angles. A torque-level comparison requires an explicit
torque actuator mode and retraining/adapting PPO to that common plant, or a
validated MPC formulation that optimizes position targets. Neither adaptation
has been claimed complete here.

## External adapter protocol

The evaluator imports a factory in your Python module:

```python
def create_controller(env, config):
    return Controller(env, config)

class Controller:
    name = "dial_mpc_g1"
    action_mode = "joint_targets"  # or "normalized"
    information_access = "current full simulated state; no future disturbance oracle"
    wind_access = "none"  # alternatively current_measurement or known_field

    def __init__(self, env, config):
        self.env = env
        # Construct the actual upstream DIAL-MPC planner and matching G1 model.
        # This documentation does not substitute a different controller.

    def reset(self, *, seed, info):
        # Seed/reinitialize the planner's own sampling and receding-horizon state.
        pass

    def act(self, observation, *, state):
        # state contains current qpos, qvel, and time from the physical simulator.
        # Return 29 finite angles in radians in env.schema() actuator order.
        raise NotImplementedError("Connect and validate the real DIAL-MPC G1 port")
```

`joint_targets` are converted through `env.action_from_joint_targets(...,
clip=False)`. Out-of-range targets raise explicitly instead of silently changing
the planned action. `normalized` outputs must already be inside [-1,1]. The
factory receives the environment for model construction, so a plugin author is
responsible for declaring any privileged information used. The standard
adapter forwards only the instantaneous state, not the future disturbance
schedule. Reset receives only the command and current time; full episode
initialization stays in evaluator logs. The adapter is not a security sandbox:
plugins must not read the future random event schedule from the environment or
reconstruct it from the paired seed. Declare wind access separately; a known
wind field is an information advantage and must be labelled in results.
Report state information advantages beside PPO, whose actor receives
only its declared proprioceptive observation (including an ideal simulated
base-velocity estimate in the native scaffold).

## Before calling this a measured DIAL-MPC baseline

1. Port the exact G1 geometry, inertias, joints, contacts, actuators, control
   interval, and command interpretation into DIAL-MPC's rollout environment.
2. Check one-step and multi-step predicted dynamics against this MuJoCo plant.
3. Establish calm walking before adding held-out perturbations and wind fields.
4. Match action limits and report controller information, planning horizon,
   rollout count, compute hardware, and inference/planning latency.
5. Run the paired evaluation suite. Report failures and boundary terminations,
   not just successful videos, and distinguish synchronous evaluation from
   meeting a real-time control deadline.

The CPU MuJoCo environment is a reproducible course-project starting point. It
is not GPU-vectorized MJX/Isaac training and does not establish sim-to-real
transfer. Hardware claims require the selected checkpoint's estimator, timing,
motor, and deployment interfaces to be reproduced and then validated.
