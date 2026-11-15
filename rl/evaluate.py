"""Paired-seed wind/disturbance evaluation for PPO and external controllers.

Examples:
  python -m rl.evaluate --checkpoint rl/runs/baseline --checkpoint rl/runs/robust
  python -m rl.evaluate --stand --scenarios calm tornado --episodes 1 --duration 2
  python -m rl.evaluate --plugin my_dial_adapter:create_controller --episodes 5

Stand is a diagnostic, never labelled as DIAL-MPC or a walking policy.
"""

import argparse
from dataclasses import replace
import csv
import json
from pathlib import Path
import time

import numpy as np

from .contracts import create_output, environment_contract, load_bundle, package_versions
from .policies import ExternalController, PPOController, StandController, checked_action


SCENARIOS = ("calm", "uniform", "gust", "tornado", "shifted_vortex", "reverse_vortex",
             "tight_vortex", "shear", "moving_vortex", "random_push")


def scenario_config(name, duration=None, base_config=None):
    from sim_env.rl_config import RLConfig
    if name not in SCENARIOS:
        raise ValueError(f"Unknown evaluation scenario {name!r}.")
    base = "tornado" if name in ("shifted_vortex", "reverse_vortex", "tight_vortex") else name
    if base == "random_push":
        cfg = RLConfig.preset("train_random")
    else:
        cfg = RLConfig.preset(base)
    changes = {"training": False, "domain_randomization": False, "observation_noise": False,
               "max_action_delay_steps": 0, "command": (.35, 0.0, 0.0), "command_ranges": None,
               "curriculum_strength": 1.0}
    if base_config is not None:
        changes.update(wind=cfg.wind, disturbances=cfg.disturbances, drag_model=cfg.drag_model)
        cfg = base_config
    if duration is not None:
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("duration must be finite and positive")
        changes["episode_duration_s"] = float(duration)
    if name == "shifted_vortex":
        changes["wind"] = {**changes.get("wind", cfg.wind), "center_m": [60.0, 55.0]}
    elif name == "reverse_vortex":
        changes["wind"] = {**changes.get("wind", cfg.wind), "tangential_speed_mps": -12.0}
    elif name == "tight_vortex":
        changes["wind"] = {**changes.get("wind", cfg.wind), "core_radius_m": 6.0}
    return replace(cfg, **changes)


def controller_reset_info(info):
    """Expose current task information, never future events or wind metadata."""
    return {"command": np.asarray(info["command"], dtype=float).copy(),
            "elapsed_s": float(info["elapsed_s"])}


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def summarize_episode(start_position, final_info, velocity_errors, yaw_errors, rewards,
                      action_seconds, terminated, truncated, requested_duration):
    """Report progress separately from survival; standing cannot pass walking."""
    start = np.asarray(start_position, dtype=float)
    end = np.asarray(final_info["base_position_m"], dtype=float)
    command = np.asarray(final_info["command"], dtype=float)
    command_speed = float(np.linalg.norm(command[:2]))
    delta = end[:2] - start[:2]
    progress = float(np.dot(delta, command[:2] / command_speed)) if command_speed > 1e-9 else 0.0
    elapsed = float(final_info["elapsed_s"])
    rmse = float(np.sqrt(np.mean(np.sum(np.asarray(velocity_errors) ** 2, axis=1)))) if velocity_errors else 0.0
    yaw_rmse = float(np.sqrt(np.mean(np.asarray(yaw_errors) ** 2))) if yaw_errors else 0.0
    failure = final_info.get("failure_reason")
    finished = bool(truncated and not terminated and elapsed >= requested_duration - 1e-7)
    success = bool(finished and failure is None and command_speed * requested_duration >= .1
                   and progress >= .6 * command_speed * requested_duration and rmse < .25 and yaw_rmse < .5)
    return {"survival_s": elapsed, "requested_duration_s": requested_duration,
            "completed_horizon": finished, "terminated": bool(terminated), "truncated": bool(truncated),
            "failure_reason": failure, "walking_success": success,
            "fell": bool(terminated and failure and any(token in str(failure).lower()
                                                        for token in ("fall", "pelvis_height", "tilt"))),
            "horizontal_displacement_m": float(np.linalg.norm(delta)), "forward_progress_m": progress,
            "commanded_distance_m": command_speed * requested_duration,
            "velocity_tracking_rmse_mps": rmse, "yaw_tracking_rmse_radps": yaw_rmse,
            "return": float(np.sum(rewards)), "steps": len(rewards),
            "controller_mean_ms": float(1000 * np.mean(action_seconds)) if action_seconds else 0.0,
            "controller_p95_ms": float(1000 * np.percentile(action_seconds, 95)) if action_seconds else 0.0,
            "controller_max_ms": float(1000 * max(action_seconds)) if action_seconds else 0.0}


def evaluate_episode(env, controller, seed, save_rollout=None):
    observation, info = env.reset(seed=int(seed))
    initial_info = info
    start = np.asarray(info["base_position_m"], dtype=float).copy()
    controller.reset(seed=int(seed), info=controller_reset_info(info))
    velocity_errors, yaw_errors, rewards, timings = [], [], [], []
    rows = []
    terminated = truncated = False
    while not (terminated or truncated):
        began = time.perf_counter()
        action = checked_action(controller.act(observation), env)
        timings.append(time.perf_counter() - began)
        if save_rollout is not None:
            state = env.get_state()
            rows.append((state["time"], state["qpos"].copy(), state["qvel"].copy(), action.copy()))
        observation, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        command = np.asarray(info["command"], dtype=float)
        velocity_errors.append(np.asarray(info["body_linear_velocity_mps"], dtype=float)[:2] - command[:2])
        yaw_errors.append(float(info["body_angular_velocity_radps"][2]) - float(command[2]))
    result = summarize_episode(start, info, velocity_errors, yaw_errors, rewards, timings,
                               terminated, truncated, env.config.episode_duration_s)
    result["seed"] = int(seed)
    result["episode_initialization"] = {key: initial_info.get(key) for key in
                                         ("episode_seed", "disturbance_seed", "domain_parameters", "disturbance_events",
                                          "initial_qpos", "initial_qvel", "initialization_randomization")}
    if save_rollout is not None and rows:
        np.savez_compressed(save_rollout, time=np.asarray([r[0] for r in rows]),
                            qpos=np.asarray([r[1] for r in rows]), qvel=np.asarray([r[2] for r in rows]),
                            action=np.asarray([r[3] for r in rows]),
                            final_qpos=env.physics.data.qpos.copy(), final_qvel=env.physics.data.qvel.copy())
    return result


def run(args):
    from sim_env.rl_config import RLConfig
    from sim_env.rl_env import G1RobustEnv
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    specifications = [("ppo", path) for path in args.checkpoint]
    specifications += [("plugin", spec) for spec in args.plugin]
    if args.stand:
        specifications.append(("stand", None))
    if not specifications:
        raise ValueError("Select --checkpoint, --plugin, or --stand.")
    plugin_config = json.loads(Path(args.plugin_config).read_text(encoding="utf-8")) if args.plugin_config else {}
    output = create_output(args.output, "rl/evaluations", "evaluation")
    seeds = list(range(args.seed, args.seed + args.episodes))
    base_config = None
    if args.config:
        base_config = RLConfig.from_json(args.config)
    elif args.checkpoint:
        # Keep physical policy semantics (joint offsets, action scale, timing).
        # Every additional bundle still must match the resulting exact contract.
        bundle = load_bundle(args.checkpoint[0])
        base_config = RLConfig.from_dict(bundle.manifest["environment_config"])
    configs = {name: scenario_config(name, args.duration, base_config) for name in args.scenarios}
    records, controllers, contract = [], [], None
    for index, (kind, source) in enumerate(specifications):
        policy_id = f"{index:02d}_{kind}"
        for scenario, config in configs.items():
            env = G1RobustEnv(config)
            controller = None
            try:
                controller = PPOController(env, source, args.device) if kind == "ppo" else (
                    ExternalController(env, source, plugin_config) if kind == "plugin" else StandController(env))
                contract = environment_contract(env)
                if scenario == next(iter(configs)):
                    controllers.append({"policy_id": policy_id, "name": controller.name, "kind": kind,
                                        "source": source, "information_access": controller.information_access,
                                        "wind_access": controller.wind_access})
                for seed in seeds:
                    rollout = output / f"{policy_id}_{scenario}_{seed}.npz" if args.save_rollouts else None
                    row = evaluate_episode(env, controller, seed, rollout)
                    records.append({"policy_id": policy_id, "controller": controller.name, "scenario": scenario, **row})
                    print(f"{controller.name}: {scenario}, seed={seed}, survived={row['survival_s']:.2f}s, "
                          f"progress={row['forward_progress_m']:.3f}m, walking_success={row['walking_success']}", flush=True)
            finally:
                if controller is not None:
                    controller.close()
                env.close()
    grouped = []
    for controller in controllers:
        for scenario in configs:
            subset = [row for row in records if row["policy_id"] == controller["policy_id"] and row["scenario"] == scenario]
            grouped.append({"policy_id": controller["policy_id"], "controller": controller["name"], "scenario": scenario,
                            "episodes": len(subset), "walking_success_rate": float(np.mean([r["walking_success"] for r in subset])),
                            "fall_rate": float(np.mean([r["fell"] for r in subset])),
                            "termination_rate": float(np.mean([r["terminated"] for r in subset])),
                            "mean_survival_s": float(np.mean([r["survival_s"] for r in subset])),
                            "mean_forward_progress_m": float(np.mean([r["forward_progress_m"] for r in subset])),
                            "mean_velocity_tracking_rmse_mps": float(np.mean([r["velocity_tracking_rmse_mps"] for r in subset]))})
    report = {"protocol": "paired scenario configurations and episode seeds for every controller",
              "seeds": seeds, "controllers": controllers, "configurations": {k: v.to_dict() for k, v in configs.items()},
              "contract": contract, "package_versions": package_versions(), "episodes": records, "summary": grouped,
              "success_definition": "straight-course walking: finish horizon without failure; command travel>=0.1m; world forward displacement>=60% commanded distance; body velocity RMSE<0.25m/s; yaw RMSE<0.5rad/s",
              "timing_definition": "controller inference/planning only, excluding environment integration and rendering",
              "limitations": ["simulation evaluation is not hardware validation", "stand is an integration diagnostic, not a walking or DIAL-MPC baseline"]}
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False, default=json_value), encoding="utf-8")
    for name, rows in (("episodes.csv", records), ("summary.csv", grouped)):
        columns = [key for key in rows[0] if key != "episode_initialization"]
        with (output / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    print(f"Evaluation saved: {output.resolve()}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", default=[], help="repeat to compare PPO bundles with paired seeds")
    parser.add_argument("--plugin", action="append", default=[], help="external controller module:factory")
    parser.add_argument("--plugin-config", help="JSON passed to every external factory")
    parser.add_argument("--config", help="base RLConfig JSON; otherwise inherit physical settings from first PPO bundle")
    parser.add_argument("--stand", action="store_true", help="include a standing integration diagnostic")
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS,
                        default=["calm", "uniform", "gust", "tornado", "shifted_vortex", "reverse_vortex", "tight_vortex"])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=10000, help="held-out starting seed; use same list for every controller")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-rollouts", action="store_true")
    parser.add_argument("--output")
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
