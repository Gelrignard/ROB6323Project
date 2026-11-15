"""Fine-tune a compatible PPO bundle, or explicitly start a new baseline.

Examples:
  python -m rl.train --from-scratch --timesteps 512 --n-steps 128 --batch-size 64
  python -m rl.train --checkpoint rl/runs/baseline --timesteps 1000000

A short smoke run validates plumbing; it does not train a walking policy.
"""

import argparse
from dataclasses import replace
from functools import partial
import json
from pathlib import Path
import random

import numpy as np

from .contracts import create_output, environment_contract, load_bundle, package_versions, save_bundle


def _make_env(config_dict, monitor_path):
    from stable_baselines3.common.monitor import Monitor
    from sim_env.rl_config import RLConfig
    from sim_env.rl_env import G1RobustEnv
    return Monitor(G1RobustEnv(RLConfig.from_dict(config_dict)), filename=str(monitor_path))


def train(args):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
    from sim_env.rl_config import RLConfig
    from sim_env.rl_env import G1RobustEnv

    if min(args.timesteps, args.n_steps, args.batch_size, args.n_envs, args.epochs, args.torch_threads) <= 0:
        raise ValueError("Training counts must be positive.")
    if args.n_steps * args.n_envs < 2 or args.batch_size < 2:
        raise ValueError("PPO advantage normalization requires rollout and batch size >1.")
    if (args.n_steps * args.n_envs) % args.batch_size:
        raise ValueError("batch-size must divide n-steps*n-envs.")
    if args.learning_rate <= 0 or not np.isfinite(args.learning_rate):
        raise ValueError("learning-rate must be finite and positive.")
    torch.set_num_threads(args.torch_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = RLConfig.from_dict(json.loads(Path(args.config).read_text(encoding="utf-8"))) if args.config else RLConfig.preset(args.preset)
    if args.curriculum_strength is not None:
        cfg = replace(cfg, curriculum_strength=args.curriculum_strength)
    if not cfg.training:
        raise ValueError("Training requires a training=true configuration; evaluation presets are held out.")
    probe = G1RobustEnv(cfg)
    try:
        contract = environment_contract(probe)
    finally:
        probe.close()
    source = load_bundle(args.checkpoint, contract) if args.checkpoint else None
    if source and args.no_normalize and source.normalization_path is not None:
        raise ValueError("Cannot discard a pretrained policy's normalization statistics.")
    output = create_output(args.output, "rl/runs", "ppo")
    factories = [partial(_make_env, cfg.to_dict(), output / f"monitor_{rank:03d}.csv")
                 for rank in range(args.n_envs)]
    vector = SubprocVecEnv(factories, start_method="spawn") if args.subproc and args.n_envs > 1 else DummyVecEnv(factories)
    vector.seed(args.seed)
    normalizer = None
    if source and source.normalization_path is not None:
        normalizer = VecNormalize.load(str(source.normalization_path), vector)
        normalizer.training = True
        normalizer.norm_reward = False
    elif source is None and not args.no_normalize:
        normalizer = VecNormalize(vector, training=True, norm_obs=True, norm_reward=False, clip_obs=10.0)
    env = normalizer or vector
    manifest = {"contract": contract, "environment_config": cfg.to_dict(), "seed": args.seed,
                "requested_additional_timesteps": args.timesteps, "package_versions": package_versions(),
                "source_checkpoint": str(source.directory) if source else None,
                "initialization": "fine-tuning compatible PPO weights and normalization" if source else "from scratch",
                "training_args": vars(args), "status": "training",
                "validation_scope": "Runnable training pipeline; model competence must be measured separately."}

    class BundleCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.last_saved = 0

        def _on_training_start(self):
            self.last_saved = self.num_timesteps

        def _on_step(self):
            if args.save_every > 0 and self.num_timesteps - self.last_saved >= args.save_every:
                save_bundle(output / "checkpoints" / f"step_{self.num_timesteps:012d}", self.model,
                            normalizer, {**manifest, "status": "intermediate"})
                self.last_saved = self.num_timesteps
            return True

    try:
        if source:
            # load() creates a new model; the source bundle is never overwritten.
            model = PPO.load(str(source.model_path), env=env, device=args.device,
                             learning_rate=args.learning_rate, seed=args.seed,
                             n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=args.epochs)
        else:
            model = PPO("MlpPolicy", env, learning_rate=args.learning_rate, n_steps=args.n_steps,
                        batch_size=args.batch_size, n_epochs=args.epochs, gamma=.99, gae_lambda=.95,
                        clip_range=.2, ent_coef=.0, vf_coef=1.0, max_grad_norm=1.0,
                        policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
                        device=args.device, seed=args.seed, verbose=1)
        model.learn(total_timesteps=args.timesteps, reset_num_timesteps=source is None,
                    callback=BundleCallback(), progress_bar=False)
        result = save_bundle(output, model, normalizer, {**manifest, "status": "complete"})
        print(json.dumps({"output": str(output.resolve()), "num_timesteps": result["num_timesteps"],
                          "initialization": result["initialization"],
                          "walking_competence": "not established by training completion"}, indent=2))
        return output
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    initialization = parser.add_mutually_exclusive_group(required=True)
    initialization.add_argument("--checkpoint", help="compatible bundle directory or model.zip")
    initialization.add_argument("--from-scratch", action="store_true", help="explicitly initialize a new PPO policy")
    parser.add_argument("--config", help="RLConfig JSON; otherwise use --preset")
    parser.add_argument("--preset", default="train_random")
    parser.add_argument("--output")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--subproc", action="store_true")
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--curriculum-strength", type=float)
    parser.add_argument("--save-every", type=int, default=100_000)
    parser.add_argument("--no-normalize", action="store_true", help="only changes new-policy initialization")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
