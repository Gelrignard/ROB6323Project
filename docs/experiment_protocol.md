# Course experiment protocol

## Question and comparisons

Does fine-tuning a nominal G1 walking PPO policy with randomized external
loads improve walking under previously unseen wind fields, while retaining
calm-ground performance?

1. Frozen nominal walking PPO checkpoint.
2. The same checkpoint fine-tuned with random joint/link perturbations.
3. If time permits, the same amount of continued PPO training without
   external perturbations, to separate extra training from disturbance exposure.
4. A validated G1 DIAL-MPC port once available; this is a separate integration
   milestone, not an already working baseline.

Use the same robot, actuator model, commanded task, action limits, timestep
and evaluation conditions. For a fair fine-tuning ablation, retain the same
domain randomization and training budget, changing only the load distribution.
Record checkpoint selection criteria before examining the final test set.

## Training

Establish real calm-ground walking first. A video made by prescribing joint
poses, a high standing reward, or a completed PPO optimizer run is insufficient.
Record the initial policy's walking performance before making a copy for
fine-tuning. A failed pretrained-policy adapter is an integration problem,
not evidence that the nominal policy lacks robustness.

Use separate train/validation/test seed ranges. Start perturbations at low
strength, and increase only if the policy retains nominal walking and
recovers from the previous load range. The implementation supports manual
strength stages; it does not implement an automatic curriculum. Multiple
independent fine-tuning seeds are needed to distinguish training variability
from evaluation initial-state variability.

Default disturbances are short half-sine pulses, sampled on robot links and
actuated joints. Large random impulses and sustained distributed wind are
different load distributions. Transfer is the hypothesis to test, not a
guarantee. Useful ablations include:

- joint torques only vs link wrenches only vs their mixture;
- short pulses vs longer-duration loads (configure `duration_s`);
- disturbance-only vs disturbance plus domain randomization;
- nominal/calm performance before and after fine-tuning.

The default 150 N / 20 N m / 30 N m limits are stress-test maxima and should
be tuned with validation scenarios. Applying the same force to a wrist and
the pelvis does not imply the same difficulty. Report target selection,
pulse duration and impulse, not just the largest force.

## Held-out evaluation

The evaluator uses a straight command of 0.35 m/s for 20 s by default, which
fits within the 10 m patch from its nominal start. Report each field
separately: calm, uniform, gust, tornado, shifted/reversed/tighter vortex,
and optionally shear/moving vortex. Additional magnitudes or terrain cases
must have named, saved configurations. A 2-second smoke test is not the
20-second walking benchmark.

Every compared controller receives identical initial states for a given
seed and identical disturbance schedules. The default 0.005 rad joint-pose
uncertainty creates distinct trials without enabling training domain
randomization. Deterministic wind scenario parameters remain fixed across
those trials. These seeds do not sample a population of tornadoes.

Record walking success, velocity/yaw RMSE, forward progress, survival and
failure reason. Summarize paired differences across seeds, report the number
of independent training runs, and retain failed episodes. Separate physics
failures from controller falls. An episode ending at the time limit is a
truncation; a fall or test-region exit is a termination.

MPC gets current full simulation state only when declared. Report whether
wind is unknown, measured currently, or known as a field. Do not give it the
future random-pulse schedule. Its planner must model the same actuator
dynamics, and its solve latency must be compared with the 20 ms control
period. Synchronous offline planning can score well while missing that
real-time deadline.

## Scope of conclusions

This simulator tests a prescribed wind field and an approximate surface
pressure model. It is not CFD, calibrated tornado reconstruction, a
rough-terrain rescue simulator, or hardware validation. Keep those later
research directions separate from measured course-project results.
