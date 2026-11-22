"""Frozen NVIDIA G1-Flat walking-policy wind evaluation (NO SAC, NO TRAINING).

Creates a Newton-GL MP4, a per-environment/per-step telemetry CSV,
per-step aggregate CSV, completed-episode CSV, and JSON run summary.

Run in Isaac Lab 3.0 with: uv run --extra video --with-editable ../g1_wind_sac_v5_5090_standalone ...
"""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pretrained', type=Path, required=True,
                   help='Official NVIDIA G1 Flat RSL-RL actor checkpoint; no SAC checkpoint necessary')
    p.add_argument('--num_envs', type=int, default=16)
    p.add_argument('--steps', type=int, default=3000, help='Vectorized 50 Hz steps; 3000 = 60 simulated seconds')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--wind_off', action='store_true')
    p.add_argument('--wind_model', choices=('panels', 'torso'), default='panels')
    p.add_argument('--wind_field', choices=('uniform', 'gust', 'stochastic_gust', 'vortex'), default='stochastic_gust')
    p.add_argument('--wind_min', type=float, default=33.0, help='Minimum nominal 10 m wind speed [m/s]')
    p.add_argument('--wind_max', type=float, default=33.0, help='Maximum nominal 10 m wind speed [m/s]')
    p.add_argument('--wind_calm_probability', type=float, default=0.0)
    p.add_argument('--wind_fixed_heading_deg', type=float, default=0.0, help='0 = wind blows +x; -1 = randomized')
    p.add_argument('--wind_force_cap_n', type=float, default=0.0, help='0 disables artificial cap')
    p.add_argument('--wind_pressure_coefficient', type=float, default=1.0)
    p.add_argument('--wind_gust_sigma_fraction', type=float, default=0.15)
    p.add_argument('--wind_gust_tau_s', type=float, default=0.8)
    p.add_argument('--wind_gust_amplitude', type=float, default=0.0)
    p.add_argument('--wind_gust_period', type=float, default=3.0)
    p.add_argument('--wind_shear_exponent', type=float, default=0.14)
    p.add_argument('--wind_reference_height_m', type=float, default=10.0)
    p.add_argument('--wind_resample_s', type=float, default=12.0)
    p.add_argument('--wind_ramp_s', type=float, default=2.0)
    p.add_argument('--wind_vortex_radius', type=float, default=4.0)
    p.add_argument('--video', action='store_true', help='Record an MP4 from env 0')
    p.add_argument('--video_length', type=int, default=3000, help='Video length in control steps')
    p.add_argument('--video_interval', type=int, default=0)
    p.add_argument('--video_fps', type=int, default=50)
    p.add_argument('--video_headless', action='store_true')
    p.add_argument('--video_static_camera', action='store_true')
    p.add_argument('--log_every', type=int, default=100)
    p.add_argument('--output_dir', type=Path, default=None)
    p.add_argument('--kit', action='store_true')
    a = p.parse_args()
    if not a.pretrained.is_file():
        p.error(f'--pretrained is not an existing file: {a.pretrained}')
    if a.num_envs < 1 or a.steps < 1 or a.log_every < 1:
        p.error('--num_envs, --steps, --log_every must be positive')
    if a.wind_min < 0 or a.wind_max < a.wind_min:
        p.error('Require 0 <= wind_min <= wind_max')
    if not 0 <= a.wind_calm_probability <= 1:
        p.error('--wind_calm_probability must be in [0,1]')
    if a.wind_gust_tau_s <= 0 or a.wind_gust_sigma_fraction < 0:
        p.error('Invalid gust parameters')
    if not 0 <= a.wind_shear_exponent <= 0.5 or a.wind_reference_height_m <= 0:
        p.error('Invalid wind shear/reference height')
    if a.video and (a.video_length > a.steps or a.video_length < 1):
        p.error('--video_length must be in [1, --steps]')
    return a


def main():
    args = get_args()
    sim_app = None
    if args.kit:
        from isaaclab.app import AppLauncher
        sim_app = AppLauncher(headless=True).app

    import numpy as np
    import torch
    from g1_wind_sac_v5.cfg import G1WindSACEnvCfg
    from g1_wind_sac_v5.env import G1WindSACEnv
    from g1_wind_sac_v5.teacher import FrozenG1Teacher
    from g1_wind_sac_v5.recording import configure_video, follow_robot_camera, camera_capture_active

    cfg = G1WindSACEnvCfg()
    cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    cfg.wind_enable = not args.wind_off
    cfg.wind_model = args.wind_model
    cfg.wind_field = args.wind_field
    cfg.wind_min_mps = args.wind_min
    cfg.wind_max_mps = args.wind_max
    cfg.wind_calm_probability = args.wind_calm_probability
    cfg.wind_fixed_heading_deg = args.wind_fixed_heading_deg
    cfg.wind_max_force_n = args.wind_force_cap_n
    cfg.wind_pressure_coefficient = args.wind_pressure_coefficient
    cfg.wind_gust_sigma_fraction = args.wind_gust_sigma_fraction
    cfg.wind_gust_tau_s = args.wind_gust_tau_s
    cfg.wind_gust_amplitude_mps = args.wind_gust_amplitude
    cfg.wind_gust_period_s = args.wind_gust_period
    cfg.wind_shear_exponent = args.wind_shear_exponent
    cfg.wind_reference_height_m = args.wind_reference_height_m
    cfg.wind_resample_s = args.wind_resample_s
    cfg.wind_ramp_s = args.wind_ramp_s
    cfg.wind_vortex_core_radius_m = args.wind_vortex_radius

    out_dir = args.output_dir or (Path(__file__).resolve().parent / 'runs' / 'teacher_baseline' /
                                  f'{datetime.now():%Y%m%d_%H%M%S}_pretrained_wind')
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.video:
        configure_video(cfg, output_dir=out_dir / 'videos', video_length=args.video_length,
                        video_interval=args.video_interval, fps=args.video_fps,
                        headless=args.video_headless, filename_prefix='pretrained_baseline')
        print(f'[VIDEO] Recording to: {out_dir / "videos"}', flush=True)
    (out_dir / 'run_config.json').write_text(json.dumps({
        'controller': 'frozen_pretrained_NVIDIA_G1_Flat_RSLRL',
        'sac_enabled': False, 'pretrained_checkpoint': str(args.pretrained.resolve()),
        'seed': args.seed, 'num_envs': args.num_envs, 'steps': args.steps,
        'wind_settings': {k: v for k, v in vars(args).items() if k.startswith('wind_')},
        'video': args.video, 'video_length': args.video_length,
        'physics_step_s': 0.005, 'control_step_s': 0.02,
        'aerodynamics_note': 'Per-link estimated six-panel box pressure, NOT exact mesh integration or CFD',
    }, indent=2))

    env = None
    try:
        env = G1WindSACEnv(cfg)
        env.capture_eval_telemetry = True  # Capture BEFORE reset; wind and motion stay in sync.
        obs, _ = env.reset()
        raw = obs['policy']
        n_obs = raw.shape[-1]
        n_act = env.action_manager.total_action_dim
        assert n_obs == 123 and n_act == 37, (n_obs, n_act)
        policy = FrozenG1Teacher(n_obs, n_act).to(env.device)
        policy.load_ppo_checkpoint(str(args.pretrained))
        policy.eval()
        print(f'[BASELINE] Frozen pretrained G1: obs={n_obs}, act={n_act}, SAC=OFF, '
              f'{args.num_envs} envs, wind={args.wind_min:g}..{args.wind_max:g} m/s', flush=True)
        n = args.num_envs
        ep_id = np.zeros(n, dtype=np.int64)
        ep_len = np.zeros(n, dtype=np.int64)
        ep_return = np.zeros(n, dtype=np.float64)
        ep_vel_sq = np.zeros(n, dtype=np.float64)
        ep_peak_wind = np.zeros(n, dtype=np.float64)
        ep_peak_force = np.zeros(n, dtype=np.float64)
        completed = falls = timeouts = 0
        velocity_sq_sum = 0.0
        reward_sum = 0.0

        step_path = out_dir / 'step_telemetry.csv'
        metrics_path = out_dir / 'evaluation_metrics.csv'
        episodes_path = out_dir / 'episodes.csv'
        with step_path.open('w', newline='') as sf, metrics_path.open('w', newline='') as mf, episodes_path.open('w', newline='') as ef:
            s = csv.writer(sf)
            s.writerow(['step','time_s','env_id','episode_id','reward',
                        'nominal_target_wind_mps','reference_wind_x_mps','reference_wind_y_mps',
                        'torso_wind_x_mps','torso_wind_y_mps','torso_wind_speed_mps',
                        'force_world_x_n','force_world_y_n','force_world_z_n',
                        'net_force_norm_n','sum_link_force_norms_n',
                        'base_local_x_m','base_local_y_m','base_local_z_m',
                        'base_vx_body_mps','base_vy_body_mps','base_vz_body_mps',
                        'base_yaw_rate_rps','base_lean_deg',
                        'cmd_vx_mps','cmd_vy_mps','cmd_yaw_rps',
                        'xy_velocity_error_mps','fall','timeout'])
            m = csv.writer(mf)
            m.writerow(['step','time_s','mean_reward','falls_this_step','timeouts_this_step',
                        'completed_episodes','cumulative_falls','cumulative_timeouts','episode_fall_rate',
                        'torso_wind_mean_mps','torso_wind_peak_mps',
                        'net_force_mean_n','net_force_peak_n','xy_velocity_rmse_mps'])
            e = csv.writer(ef)
            e.writerow(['env_id','episode_id','end_step','length_steps','duration_s',
                        'episode_return','velocity_rmse_mps','peak_torso_wind_mps',
                        'peak_aero_net_force_n','fall','time_limit'])
            for k in range(args.steps):
                if args.video and not args.video_static_camera and camera_capture_active(
                    k, args.video_length, args.video_interval
                ):
                    follow_robot_camera(env)
                with torch.inference_mode():
                    action = policy(raw)
                    cmd = env.command_manager.get_command('base_velocity').clone()
                    obs, reward, terminated, truncated, extras = env.step(action)
                    raw = obs['policy']
                    t = extras['wind_eval_telemetry']
                    # One device-to-host synchronization per vectorized step.
                    packed = torch.cat([
                        reward[:, None],
                        t['wind_target_mps'][:, None],
                        t['wind_reference_world_mps'][:, :2],
                        t['wind_torso_world_mps'][:, :2],
                        torch.linalg.vector_norm(t['wind_torso_world_mps'], dim=-1)[:, None],
                        t['wind_net_force_world_n'],
                        torch.linalg.vector_norm(t['wind_net_force_world_n'],dim=-1)[:,None],
                        t['wind_sum_link_force_magnitudes_n'][:,None],
                        t['root_pos_local_m'],
                        t['root_lin_vel_b_mps'],
                        t['root_ang_vel_b_rps'][:, 2:3],
                        t['lean_deg'][:,None],
                        cmd[:, :3],
                        (t['root_lin_vel_b_mps'][:,:2]-cmd[:,:2]).square().sum(dim=-1).sqrt()[:,None],
                        terminated[:, None].float(),
                        (truncated & ~terminated)[:, None].float(),
                    ], dim=-1).cpu().numpy()
                assert packed.shape == (n, 26), f'Unexpected telemetry shape: {packed.shape}'
                reward_v, wind_v, force_v = packed[:,0], packed[:,6], packed[:,10]
                # Indices: reward=0, nominal=1, ref=(2,3), torso=(4,5), speed=6,
                # net force=(7,8,9), norm=10, sum link norms=11, base pos=(12..14),
                # base velocity=(15..17), yaw=18, lean=19, cmd=(20..22), velerr=23,
                # fall=24, timeout=25.
                # Keep actual calculated indices explicit to avoid silent column drift.
                velerr_v = packed[:,23]
                fall_v = packed[:,24].astype(np.int64)
                timeout_v = packed[:,25].astype(np.int64)
                for j in range(n):
                    s.writerow([k+1, round((k+1)*env.step_dt,6), j, ep_id[j],
                                *[float(x) for x in packed[j]]])
                ep_len += 1
                ep_return += reward_v
                ep_vel_sq += velerr_v**2
                ep_peak_wind = np.maximum(ep_peak_wind, wind_v)
                ep_peak_force = np.maximum(ep_peak_force, force_v)
                velocity_sq_sum += float(np.sum(velerr_v**2))
                reward_sum += float(np.mean(reward_v))
                falls_now, timeouts_now = int(fall_v.sum()), int(timeout_v.sum())
                falls += falls_now
                timeouts += timeouts_now
                done_ids = np.nonzero(fall_v | timeout_v)[0]
                for j in done_ids:
                    completed += 1
                    e.writerow([int(j),int(ep_id[j]),k+1,int(ep_len[j]),float(ep_len[j]*env.step_dt),
                                float(ep_return[j]),float(np.sqrt(ep_vel_sq[j]/ep_len[j])),
                                float(ep_peak_wind[j]),float(ep_peak_force[j]),
                                int(fall_v[j]),int(timeout_v[j])])
                    ep_id[j] += 1
                    ep_len[j] = 0
                    ep_return[j] = ep_vel_sq[j] = ep_peak_wind[j] = ep_peak_force[j] = 0
                m.writerow([k+1,round((k+1)*env.step_dt,6),float(reward_v.mean()),
                            falls_now,timeouts_now,completed,falls,timeouts,
                            float(falls/completed) if completed else '',
                            float(wind_v.mean()),float(wind_v.max()),
                            float(force_v.mean()),float(force_v.max()),
                            float(np.sqrt(velocity_sq_sum/((k+1)*n)))])
                if (k+1) % args.log_every == 0 or k+1 == args.steps:
                    sf.flush(); mf.flush(); ef.flush()
                    print(f'step={k+1} t={(k+1)*env.step_dt:.1f}s '
                          f'reward={reward_v.mean():+.4f} falls={falls} episodes={completed} '
                          f'fall_rate={falls/completed if completed else 0:.3f} '
                          f'torso_wind={wind_v.mean():.1f}m/s net_force={force_v.mean():.1f}N',flush=True)

        summary = {
            'controller':'frozen_pretrained_NVIDIA_G1_Flat_RSLRL',
            'sac_training':False, 'teacher_only':True,
            'steps_per_env': args.steps, 'num_envs': n,
            'sampled_transitions':args.steps*n,
            'simulated_seconds_per_env':args.steps*env.step_dt,
            'completed_episodes':completed, 'episode_falls':falls,
            'episode_timeouts':timeouts,
            'episode_fall_rate':falls/completed if completed else None,
            'unfinished_episodes_censored':int(np.count_nonzero(ep_len)),
            'mean_reward_per_step':reward_sum/args.steps,
            'xy_velocity_rmse_mps':float(np.sqrt(velocity_sq_sum/(args.steps*n))),
            'output_dir':str(out_dir),
            'comparison_note':'Only completed episodes enter episode fall rate; unfinished episodes are censored. Use matching settings for trained SAC evaluation.',
            'wind_measurements_note':'wind target is nominal at 10m; torso wind accounts for shear/gust; forces are last physics substep before episode resets.',
        }
        (out_dir/'evaluation_summary.json').write_text(json.dumps(summary,indent=2))
        print(f'[DONE] Baseline saved: {out_dir}', flush=True)
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        if env is not None:
            env.close()
        if sim_app is not None:
            sim_app.close()


if __name__ == '__main__':
    main()
