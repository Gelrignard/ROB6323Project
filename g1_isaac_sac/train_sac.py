"""Train a wind-resistance residual SAC on a PRETRAINED Isaac Lab 3.0 G1 Flat policy.

Run directly from the sibling Isaac Lab checkout with uv:
  uv run --extra video --with-editable ../g1_wind_sac_v5_5090_standalone \
    python ../g1_wind_sac_v5_5090_standalone/train_sac.py --pretrained /path/to/G1_flat_RSLRL.pt
No Isaac Gym, no PPO updates, no randomly initialized walking policy.
"""
import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', type=Path, help='Official G1 Flat RSL-RL PPO checkpoint (teacher weights only)')
parser.add_argument('--resume', type=Path, help='Resume a residual SAC checkpoint; contains frozen teacher')
parser.add_argument('--num_envs', type=int, default=256)
parser.add_argument('--steps', type=int, default=100000, help='Number of vectorized environment steps')
parser.add_argument('--max_runtime_minutes', type=float, default=28.0, help='Stop gracefully and checkpoint after this many training minutes (0 = unlimited)')
parser.add_argument('--wind_max', type=float, default=42.0, help='Maximum wind speed at end of curriculum (m/s)')
parser.add_argument('--wind_start', type=float, default=4.0, help='Initial maximum wind speed for curriculum (m/s); omit for constant wind')
parser.add_argument('--wind_curriculum_steps', type=int, default=80000, help='Number of vectorized steps over which wind maximum rises linearly')
parser.add_argument('--wind_off', action='store_true')
parser.add_argument('--wind_model', choices=('torso','panels'), default='panels', help='panels: per-body proxy-box pressure quadrature')
parser.add_argument('--wind_field', choices=('uniform','gust','stochastic_gust','vortex'), default='stochastic_gust')
parser.add_argument('--wind_force_cap_n', type=float, default=0.0, help='0 disables cap; default: 60 N torso / uncapped panels')
parser.add_argument('--wind_fixed_heading_deg', type=float, default=-1.0, help='-1=random; 0=wind flows along +x')
parser.add_argument('--wind_min', type=float, default=0.0)
parser.add_argument('--wind_calm_probability', type=float, default=0.1)
parser.add_argument('--wind_pressure_coefficient', type=float, default=1.0)
parser.add_argument('--wind_gust_amplitude', type=float, default=0.0)
parser.add_argument('--wind_gust_period', type=float, default=3.0)
parser.add_argument('--wind_gust_sigma_fraction', type=float, default=0.15, help='OU gust 1-sigma as a fraction of mean wind (synthetic)')
parser.add_argument('--wind_gust_tau_s', type=float, default=0.8, help='OU gust autocorrelation seconds (synthetic)')
parser.add_argument('--wind_resample_s', type=float, default=12.0, help='Seconds between new mean wind targets')
parser.add_argument('--wind_ramp_s', type=float, default=2.0, help='Seconds to interpolate mean wind after resampling')
parser.add_argument('--wind_shear_exponent', type=float, default=0.14, help='Power-law exponent; 0=uniform; storm preset uses 0.14 except tornado gust equivalents')
parser.add_argument('--wind_reference_height_m', type=float, default=10.0, help='Height of nominal wind speed (m); NOAA hurricanes use 10m')
parser.add_argument('--storm_target', choices=('high_wind','severe_gust','tropical_storm','hurricane_cat1','hurricane_cat2','hurricane_cat3','hurricane_cat4','hurricane_cat5','ef0_gust','ef1_gust','ef2_gust','ef3_gust'), default=None, help='Set NOAA-inspired storm speed ceiling and staged curriculum; NOT a full storm simulation')
parser.add_argument('--storm_curriculum_steps', type=int, default=0, help='Length of storm-stage curriculum; 0=80%% of --steps')
parser.add_argument('--storm_min_fraction', type=float, default=0.8, help='Minimum sampled non-calm wind as fraction of current ceiling')
parser.add_argument('--wind_vortex_radius', type=float, default=4.0)
parser.add_argument('--history_len', type=int, default=4, help='SAC receives newest-to-oldest observations; frozen teacher receives 123 newest values')
# Audit v5: opt-in gated curriculum and safety shaping. Defaults preserve old runs.
parser.add_argument('--curriculum_mode', choices=('time','gated'), default='gated',
                    help='gated: only increase wind after completed high-exposure episodes achieve target survival')
parser.add_argument('--gate_start_mps', type=float, default=4.0)
parser.add_argument('--gate_increment_mps', type=float, default=2.0)
parser.add_argument('--gate_success', type=float, default=0.85)
parser.add_argument('--gate_min_episodes', type=int, default=128)
parser.add_argument('--gate_required_windows', type=int, default=2)
parser.add_argument('--gate_exposure_level', type=float, default=0.65,
                    help='Nominal wind >= this fraction of current ceiling counts as an exposed step')
parser.add_argument('--gate_exposure_fraction', type=float, default=0.4,
                    help='Fraction of episode steps that must be exposed to qualify for gate')
parser.add_argument('--wind_survival_bonus', type=float, default=0.0,
                    help='Optional bonus per living step, scaled by squared nominal wind/ceiling; 0 preserves stock reward')
parser.add_argument('--wind_fall_penalty', type=float, default=0.0,
                    help='Optional EXTRA fall penalty; stock G1 already penalizes termination')
parser.add_argument('--flat_orientation_weight', type=float, default=-0.3,
                    help='Optional replacement for stock flat_orientation_l2 reward weight; negative; e.g. -0.3 allows more tilt')
parser.add_argument('--alpha_floor', type=float, default=0.001,
                    help='Optional temperature floor for exploration ablations, 0 keeps original SAC')
parser.add_argument('--kit', action='store_true', help='Launch Isaac Sim Kit; leave unset for default 3.0 kitless Newton backend')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--save_every', type=int, default=2500)
parser.add_argument('--log_every', type=int, default=250)
parser.add_argument('--checkpoint_dir', type=Path, default=Path(__file__).resolve().parent / 'runs' / 'train')
parser.add_argument('--learning_starts', type=int, default=131072)
parser.add_argument('--teacher_only_steps', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--updates_per_step', type=int, default=4)
parser.add_argument('--replay_capacity', type=int, default=1048576)
parser.add_argument('--replay_strategy', choices=('uniform','balanced'), default='balanced',help='balanced: retain equal capacity per nominal wind band for stratified replay')
parser.add_argument('--residual_scale', type=float, default=0.25)
parser.add_argument('--actor_lr', type=float, default=5e-5)
parser.add_argument('--residual_penalty', type=float, default=0.5)
parser.add_argument('--video', action=argparse.BooleanOptionalAction, default=True,
                    help='Record native Newton GL MP4 during training (--no-video to disable)')
parser.add_argument('--video_length', type=int, default=200, help='Frames/steps per clip (200 = 4 simulated seconds)')
parser.add_argument('--video_interval', type=int, default=1000, help='0 = only first clip; otherwise record every N steps')
parser.add_argument('--video_fps', type=int, default=50)
parser.add_argument('--video_headless', action=argparse.BooleanOptionalAction, default=True, help='Record without a visible Newton GL window')
parser.add_argument('--video_static_camera', action='store_true', help='Disable the G1-following camera during recording')
parser.add_argument('--video_dir', type=Path, default=None, help='Default: RUN_DIR/videos')
# Plain argparse defaults: no named profile or external launcher.
# Explicit flags always override the integrated 5090/audit/video defaults.
args = parser.parse_args()
from g1_wind_sac_v5.storms import STORM_TARGETS_MPS, storm_curriculum_ceiling, storm_min_wind
if args.storm_target:
    if args.wind_off:
        parser.error('--storm_target cannot be combined with --wind_off')
    args.wind_max = STORM_TARGETS_MPS[args.storm_target]
    args.wind_start = min(args.gate_start_mps, args.wind_max)
    args.wind_curriculum_steps = args.storm_curriculum_steps or max(1, int(0.8 * args.steps))
    args.wind_model = 'panels'
    args.wind_field = 'stochastic_gust'
    args.wind_force_cap_n = 0.0
    args.wind_resample_s = 12.0
    args.wind_ramp_s = 2.0
    args.wind_shear_exponent = 0.0 if args.storm_target.startswith('ef') else 0.14
    print(f'[STORM] {args.storm_target}: wind ceiling={args.wind_max:.1f} m/s; '
          'synthetic OU gusts; NOT a full meteorological storm simulation', flush=True)
if args.curriculum_mode == 'gated':
    if args.wind_start is None:
        parser.error('gated curriculum requires --storm_target or --wind_start/--wind_max')
    if args.gate_start_mps < 0 or args.gate_start_mps > args.wind_max:
        parser.error('--gate_start_mps must be between 0 and the wind target')
    if args.gate_increment_mps <= 0 or args.gate_min_episodes < 1 or args.gate_required_windows < 1:
        parser.error('invalid gate increment / minimum episodes / required windows')
    if not 0 <= args.gate_success <= 1 or not 0 <= args.gate_exposure_level <= 1 or not 0 <= args.gate_exposure_fraction <= 1:
        parser.error('gate success/exposure fractions must be in [0,1]')
    args.wind_start = args.gate_start_mps
if args.flat_orientation_weight is not None and args.flat_orientation_weight > 0:
    parser.error('--flat_orientation_weight should be non-positive')
if args.alpha_floor < 0 or args.wind_survival_bonus < 0 or args.wind_fall_penalty < 0:
    parser.error('alpha floor and reward shaping values must be nonnegative')
if args.wind_gust_sigma_fraction < 0 or args.wind_gust_tau_s <= 0:
    parser.error('--wind_gust_sigma_fraction must be nonnegative and --wind_gust_tau_s > 0')
if not 0 <= args.wind_shear_exponent <= 0.5 or args.wind_reference_height_m <= 0:
    parser.error('Require 0 <= shear exponent <= 0.5 and wind reference height >0')
if args.wind_resample_s <= 0 or args.wind_ramp_s <= 0:
    parser.error('--wind_resample_s and --wind_ramp_s must be >0')
if (args.pretrained is None) == (args.resume is None):
    parser.error('Supply EXACTLY ONE of --pretrained and --resume. Never start from a random walking policy.')
if args.history_len < 1 or args.history_len > 16:
    parser.error('--history_len must be 1..16')
if args.wind_min < 0 or args.wind_min > (args.wind_start if args.wind_start is not None else args.wind_max):
    parser.error('--wind_min must be in [0, initial wind max]')
if args.wind_calm_probability < 0 or args.wind_calm_probability > 1:
    parser.error('--wind_calm_probability must be between 0 and 1')
if args.wind_force_cap_n is not None and args.wind_force_cap_n < 0:
    parser.error('--wind_force_cap_n must be >= 0')
if args.max_runtime_minutes < 0:
    parser.error('--max_runtime_minutes must be >= 0')
if args.wind_max < 0:
    parser.error('--wind_max must be >= 0')
if args.wind_start is not None:
    if args.wind_curriculum_steps <= 0:
        parser.error('--wind_start requires --wind_curriculum_steps > 0')
    if not 0 <= args.wind_start <= args.wind_max:
        parser.error('Require 0 <= --wind_start <= --wind_max')
elif args.wind_curriculum_steps != 0:
    parser.error('--wind_curriculum_steps requires --wind_start')
for opt in (args.pretrained, args.resume):
    if opt is not None and not opt.is_file():
        parser.error(f'Checkpoint not found: {opt}')

# The stock Isaac Lab 3.0 G1 Flat task uses Newton MJWarp (kitless by default).
# Launching Isaac Sim is optional, not a dependency of Newton training.
simulation_app = None
if args.kit:
    from isaaclab.app import AppLauncher
    simulation_app = AppLauncher(headless=True).app

import torch
from g1_wind_sac_v5.cfg import G1WindSACEnvCfg
from g1_wind_sac_v5.env import G1WindSACEnv
from g1_wind_sac_v5.sac import SACConfig, SACAgent, ReplayBuffer, BinnedReplayBuffer
from g1_wind_sac_v5.recording import configure_video, follow_robot_camera, camera_capture_active
from g1_wind_sac_v5.schedules import linear_wind_limit
from g1_wind_sac_v5.history import ObservationHistory
from g1_wind_sac_v5.audit import exclusive_outcomes, SuccessGatedWindCurriculum


def main():
    torch.manual_seed(args.seed)
    cfg = G1WindSACEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    if args.flat_orientation_weight is not None:
        cfg.rewards.flat_orientation_l2.weight = args.flat_orientation_weight
    cfg.wind_enable = not args.wind_off
    cfg.wind_model = args.wind_model
    cfg.wind_field = args.wind_field
    cfg.wind_max_force_n = args.wind_force_cap_n if args.wind_force_cap_n is not None else (0.0 if args.wind_model == "panels" else 60.0)
    cfg.wind_fixed_heading_deg = args.wind_fixed_heading_deg
    cfg.wind_min_mps = args.wind_min
    cfg.wind_calm_probability = args.wind_calm_probability
    cfg.wind_pressure_coefficient = args.wind_pressure_coefficient
    cfg.wind_gust_amplitude_mps = args.wind_gust_amplitude
    cfg.wind_gust_period_s = args.wind_gust_period
    cfg.wind_gust_sigma_fraction = args.wind_gust_sigma_fraction
    cfg.wind_gust_tau_s = args.wind_gust_tau_s
    cfg.wind_resample_s = args.wind_resample_s
    cfg.wind_ramp_s = args.wind_ramp_s
    cfg.wind_shear_exponent = args.wind_shear_exponent
    cfg.wind_reference_height_m = args.wind_reference_height_m
    cfg.wind_vortex_core_radius_m = args.wind_vortex_radius
    cfg.wind_max_mps = args.wind_start if args.wind_start is not None else args.wind_max
    if args.wind_start is not None and args.storm_min_fraction > 0:
        cfg.wind_min_mps = storm_min_wind(cfg.wind_max_mps, args.storm_min_fraction)
    run_dir = args.checkpoint_dir / datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir.mkdir(parents=True, exist_ok=True)
    # Save the exact resolved launch settings (including manual CLI overrides).
    (run_dir / 'run_config.json').write_text(
        json.dumps({k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(args).items()}, indent=2, sort_keys=True) + '\n'
    )
    if args.video:
        video_dir = args.video_dir or (run_dir / 'videos')
        configure_video(cfg, output_dir=video_dir, video_length=args.video_length,
                        video_interval=args.video_interval, fps=args.video_fps,
                        headless=args.video_headless, filename_prefix='train')
        print(f'[VIDEO] Native Newton GL MP4 recording enabled: {video_dir}')
    env = G1WindSACEnv(cfg)
    try:
        obs_dict, _ = env.reset()
        raw_obs = obs_dict['policy']
        history = ObservationHistory(raw_obs, args.history_len)
        obs = history.flattened()
        obs_dim = obs.shape[-1]
        act_dim = env.action_manager.total_action_dim
        agent_cfg = SACConfig(
            learning_starts=args.learning_starts,
            teacher_only_steps=args.teacher_only_steps,
            batch_size=args.batch_size,
            updates_per_step=args.updates_per_step,
            replay_capacity=args.replay_capacity,
            residual_scale=args.residual_scale,
            actor_lr=args.actor_lr,
            residual_penalty=args.residual_penalty,
            history_len=args.history_len,
            alpha_floor=args.alpha_floor,
        )
        agent = SACAgent(obs_dim, act_dim, env.device, agent_cfg)
        start_step = 0
        if args.pretrained is not None:
            agent.actor.teacher.load_ppo_checkpoint(str(args.pretrained))
            print(f'[OK] Loaded PRETRAINED G1 actor from {args.pretrained}; '
                  f'obs={obs_dim}, act={act_dim}, trainable=RESIDUAL SAC only')
        else:
            saved = torch.load(args.resume, map_location=env.device, weights_only=True)
            if saved.get('format') != 'g1_wind_residual_sac_v1':
                raise ValueError('Not a residual SAC checkpoint.')
            if tuple(saved['dimensions']) != (obs_dim, act_dim):
                raise ValueError('Resume checkpoint obs/action dimensions differ from environment.')
            if int(saved['sac_cfg'].get('history_len', 1)) != args.history_len:
                raise ValueError('Resume --history_len must equal checkpoint value.')
            if abs(saved['sac_cfg']['residual_scale'] - args.residual_scale) > 1e-9:
                raise ValueError('Resume --residual_scale must equal checkpoint value.')
            agent.load_state_dict(saved['agent'])
            start_step = int(saved['step'])
            print(f'[OK] Resumed SAC at vector step {start_step}. Replay buffer starts empty.')
        start_update_count = agent.num_updates
        replay_type = BinnedReplayBuffer if args.replay_strategy == 'balanced' else ReplayBuffer
        replay = replay_type(args.replay_capacity, obs_dim, act_dim, torch.device(env.device))
        gate = (SuccessGatedWindCurriculum(
            start=args.wind_start, target=args.wind_max,
            increment=args.gate_increment_mps, min_episodes=args.gate_min_episodes,
            survival_threshold=args.gate_success, required_windows=args.gate_required_windows)
            if args.curriculum_mode == 'gated' else None)
        ep_steps = torch.zeros(args.num_envs, device=env.device, dtype=torch.int32)
        ep_wind_exposed = torch.zeros_like(ep_steps)
        total_gate_eligible = total_gate_falls = 0
        metrics_file = (run_dir / 'train_metrics.csv').open('w', newline='')
        metrics_csv = csv.DictWriter(metrics_file, fieldnames=[
            'step', 'transitions', 'reward', 'fall_fraction',
            'critic_loss', 'alpha', 'residual_rms', 'wind_max_mps', 'sac_updates',
            'total_transitions', 'episodes_completed', 'episode_falls', 'episode_timeouts', 'episode_fall_rate',
            'wind_instantaneous_mean_mps', 'wind_instantaneous_peak_mps', 'wind_net_force_mean_n',
            'reward_sac_mean', 'wind_survival_bonus_mean', 'actor_entropy', 'actor_log_std_mean',
            'replay_window_seconds', 'sample_reuse_ratio', 'gate_ceiling_mps',
            'gate_qualified_episodes', 'gate_qualified_falls', 'gate_qualified_fall_rate',
            'gate_last_window_survival', 'gate_last_window_episodes', 'gate_advanced', 'replay_bin_sizes', 'command_changed_fraction', 'step_mod_500'])
        metrics_csv.writeheader()
        train_started = time.monotonic()
        episodes_total = falls_total = timeouts_total = 0
        window_episodes = window_falls = window_timeouts = 0
        def save_checkpoint(completed_step):
            checkpoint = {'format':'g1_wind_residual_sac_v1', 'step':completed_step,
                          'training_setup':'5090_audit_v5_integrated',
                          'dimensions':(obs_dim,act_dim), 'sac_cfg':vars(agent_cfg),
                          'agent':agent.state_dict(),
                          'environment': {'wind_model':args.wind_model, 'wind_field':args.wind_field,
                                          'wind_force_cap_n':cfg.wind_max_force_n,
                                          'history_len':args.history_len,
                                          'storm_target':args.storm_target,
                                          'storm_min_fraction':args.storm_min_fraction,
                                          'wind_gust_sigma_fraction':args.wind_gust_sigma_fraction,
                                          'wind_gust_tau_s':args.wind_gust_tau_s,
                                          'wind_shear_exponent':args.wind_shear_exponent,
                                          'wind_reference_height_m':args.wind_reference_height_m,
                                          'wind_pressure_coefficient':args.wind_pressure_coefficient,
                                          'curriculum_mode':args.curriculum_mode,
                                          'wind_survival_bonus':args.wind_survival_bonus,
                                          'wind_fall_penalty':args.wind_fall_penalty,
                                          'aero_face_quadrature':2,
                                          'replay_strategy':args.replay_strategy,
                                          'flat_orientation_weight':args.flat_orientation_weight},
                          'note':'Frozen pretrained G1 actor + trainable SAC residual; replay not serialized.'}
            path = run_dir / f'sac_{completed_step}.pt'
            torch.save(checkpoint, path)
            print(f'Saved {path}', flush=True)

        for step in range(start_step, start_step + args.steps):
            if args.video and not args.video_static_camera and camera_capture_active(
                step - start_step, args.video_length, args.video_interval
            ):
                follow_robot_camera(env)
            # Advance a smooth disturbance curriculum without resetting the replay buffer.
            # The new limit takes effect for each environment on its next wind resample.
            if gate is not None:
                env.cfg.wind_max_mps = gate.ceiling
                if args.storm_min_fraction > 0:
                    env.cfg.wind_min_mps = storm_min_wind(env.cfg.wind_max_mps, args.storm_min_fraction)
            elif args.wind_start is not None:
                env.cfg.wind_max_mps = (
                    storm_curriculum_ceiling(step, args.wind_start, args.wind_max, args.wind_curriculum_steps)
                    if args.storm_target else
                    linear_wind_limit(step, args.wind_start, args.wind_max, args.wind_curriculum_steps)
                )
                if args.storm_min_fraction > 0:
                    env.cfg.wind_min_mps = storm_min_wind(env.cfg.wind_max_mps, args.storm_min_fraction)
            with torch.no_grad():
                action = agent.act(obs, teacher_only=(step < args.teacher_only_steps))
                # Capture wind and command before env.step(), which can reset envs
                # and resample velocity commands at a 10s/500-step interval.
                nominal_before = env._wind_speed.clone()
                command_before = env.command_manager.get_command('base_velocity').clone()
                nxt_dict, rew, terminated, truncated, extras = env.step(action)
                next_obs_raw = nxt_dict['policy']
                command_after = env.command_manager.get_command('base_velocity')
                command_changed_fraction = ((command_after-command_before).abs().amax(dim=-1)>1e-5).float().mean().item()
                falls, timeouts, done = exclusive_outcomes(terminated, truncated)
                if gate is not None:
                    ep_steps += 1
                    ep_wind_exposed += (nominal_before >= args.gate_exposure_level * gate.ceiling).int()
                    eligible = done & (ep_wind_exposed.float() >= args.gate_exposure_fraction * ep_steps.float())
                    gate.add_completed(falls, eligible)
                    total_gate_eligible += int(eligible.sum().item())
                    total_gate_falls += int((falls & eligible).sum().item())
                    ep_steps[done] = 0
                    ep_wind_exposed[done] = 0
                wind_fraction = (nominal_before / max(env.cfg.wind_max_mps, 1.0)).clamp(0,1)
                survival_bonus = args.wind_survival_bonus * wind_fraction.square() * (~done).float()
                learn_reward = rew + survival_bonus - args.wind_fall_penalty * falls.float()
                final_obs = extras.get('final_obs') if bool(done.any()) else None
                if bool(done.any()) and final_obs is None:
                    raise RuntimeError('Missing final_obs on episode termination; compute_final_obs must be True.')
                terminal_raw = final_obs['policy'] if final_obs is not None else None
                terminal_hist, new_policy_hist = history.advance(next_obs_raw, done, terminal_raw)
                transition_next = torch.where(done[:, None], terminal_hist, new_policy_hist)
                if args.replay_strategy == 'balanced':
                    replay.add(obs, action, learn_reward, transition_next, falls, wind_mps=nominal_before)
                else:
                    replay.add(obs, action, learn_reward, transition_next, falls)
                obs = new_policy_hist
                f = int(falls.sum().item())
                t = int(timeouts.sum().item())
                d = int(done.sum().item())
                falls_total += f
                timeouts_total += t
                episodes_total += d
                window_falls += f
                window_timeouts += t
                window_episodes += d
            metrics = None
            if replay.size >= max(args.learning_starts, args.batch_size):
                for _ in range(args.updates_per_step):
                    metrics = agent.update(replay.sample(args.batch_size))
            current = step + 1
            if current % args.log_every == 0:
                gate_decision = gate.check() if gate is not None else None
                if gate_decision is not None:
                    print(f'[GATE] exposed episodes={gate_decision.tested_episodes} '
                          f'survival={gate_decision.survival_fraction:.3f}; '
                          f'wind_ceiling={gate_decision.ceiling_before:g}->{gate_decision.ceiling_after:g}m/s '
                          f'advanced={gate_decision.advanced}',flush=True)
                with torch.no_grad():
                    # Measured effective wind, not just the curriculum's nominal ceiling.
                    local_wind = env._body_wind(env._robot.data.body_com_pos_w.torch)
                    instantaneous_speed = torch.linalg.vector_norm(local_wind, dim=-1)
                    actual_mean = instantaneous_speed.mean().item()
                    actual_peak = instantaneous_speed.amax().item()
                    mean_net_force = torch.linalg.vector_norm(env._forces.sum(dim=1), dim=-1).mean().item()
                info = f'step={current} transitions={replay.size} reward={rew.mean().item():+.4f} fall_fraction={terminated.float().mean().item():.3f}'
                if metrics:
                    info += f' critic_loss={metrics["critic_loss"]:.3f} alpha={metrics["alpha"]:.4f} residual_rms={metrics["residual_rms"]:.3f}'
                info += f' wind_max={env.cfg.wind_max_mps:.2f} actual_wind_mean={actual_mean:.1f} actual_peak={actual_peak:.1f} force_mean={mean_net_force:.0f}N sac_updates={agent.num_updates}'
                print(info, flush=True)
                metrics_csv.writerow({
                    'step': current,
                    'transitions': replay.size,
                    'reward': float(rew.mean().item()),
                    'fall_fraction': float(terminated.float().mean().item()),
                    'critic_loss': metrics['critic_loss'] if metrics else '',
                    'alpha': metrics['alpha'] if metrics else '',
                    'residual_rms': metrics['residual_rms'] if metrics else '',
                    'wind_max_mps': env.cfg.wind_max_mps,
                    'sac_updates': agent.num_updates,
                    'total_transitions': current * args.num_envs,
                    'episodes_completed': window_episodes,
                    'episode_falls': window_falls,
                    'episode_timeouts': window_timeouts,
                    'episode_fall_rate': window_falls / window_episodes if window_episodes else '',
                    'wind_instantaneous_mean_mps': actual_mean,
                    'wind_instantaneous_peak_mps': actual_peak,
                    'wind_net_force_mean_n': mean_net_force,
                    'reward_sac_mean':float(learn_reward.mean().item()),
                    'wind_survival_bonus_mean':float(survival_bonus.mean().item()),
                    'actor_entropy':metrics['actor_entropy'] if metrics else '',
                    'actor_log_std_mean':metrics['actor_log_std_mean'] if metrics else '',
                    'replay_window_seconds':replay.size/args.num_envs/50.0,
                    'sample_reuse_ratio':(agent.num_updates-start_update_count)*args.batch_size / max(1,(current-start_step)*args.num_envs),
                    'gate_ceiling_mps':gate.ceiling if gate is not None else '',
                    'gate_qualified_episodes':total_gate_eligible if gate is not None else '',
                    'gate_qualified_falls':total_gate_falls if gate is not None else '',
                    'gate_qualified_fall_rate':total_gate_falls/total_gate_eligible if gate is not None and total_gate_eligible else '',
                    'gate_last_window_survival':gate_decision.survival_fraction if gate_decision else '',
                    'gate_last_window_episodes':gate_decision.tested_episodes if gate_decision else '',
                    'gate_advanced':int(gate_decision.advanced) if gate_decision else '',
                    'replay_bin_sizes':str(replay.bin_sizes) if args.replay_strategy=='balanced' else '',
                    'command_changed_fraction':command_changed_fraction,
                    'step_mod_500':current%500,
                })
                metrics_file.flush()
                window_episodes = window_falls = window_timeouts = 0
            is_final = current == start_step + args.steps
            elapsed = time.monotonic() - train_started
            time_up = bool(args.max_runtime_minutes and elapsed >= 60 * args.max_runtime_minutes)
            if current % args.save_every == 0 or is_final or time_up:
                save_checkpoint(current)
            if time_up:
                print(f'[TIME LIMIT] Stopped after {elapsed / 60:.1f} training minutes '
                      f'at step {current}; final checkpoint saved.', flush=True)
                break
        print(f'[EPISODES] completed={episodes_total} falls={falls_total} timeouts={timeouts_total} '
              f'fall_rate={falls_total/max(1,episodes_total):.3f}', flush=True)
        print(f'[DONE] Training loop elapsed {(time.monotonic()-train_started)/60:.1f} minutes', flush=True)
    finally:
        if 'metrics_file' in locals():
            metrics_file.close()
        env.close()


try:
    main()
finally:
    if simulation_app is not None:
        simulation_app.close()
