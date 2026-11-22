"""Compare frozen pretrained G1 teacher with a trained wind-residual SAC checkpoint."""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True, type=Path, help='Saved sac_*.pt from train_sac.py')
parser.add_argument('--num_envs', type=int, default=16)
parser.add_argument('--steps', type=int, default=1000)
parser.add_argument('--wind_max', type=float, default=2.0)
parser.add_argument('--wind_off', action='store_true')
parser.add_argument('--wind_model', choices=('torso','panels'), default=None, help='panels: six pressure panels per body (estimated box geometry)')
parser.add_argument('--wind_field', choices=('uniform','gust','stochastic_gust','vortex'), default=None)
parser.add_argument('--wind_force_cap_n', type=float, default=None, help='0 disables cap; defaults to checkpoint setting')
parser.add_argument('--wind_fixed_heading_deg', type=float, default=-1.0, help='-1=random; 0=wind flows along +x')
parser.add_argument('--wind_min', type=float, default=0.0)
parser.add_argument('--wind_calm_probability', type=float, default=0.2)
parser.add_argument('--wind_pressure_coefficient', type=float, default=1.0)
parser.add_argument('--wind_gust_amplitude', type=float, default=0.0)
parser.add_argument('--wind_gust_period', type=float, default=3.0)
parser.add_argument('--wind_gust_sigma_fraction', type=float, default=None, help='OU gust standard deviation fraction (checkpoint default)')
parser.add_argument('--wind_gust_tau_s', type=float, default=None, help='OU gust correlation time in seconds (checkpoint default)')
parser.add_argument('--wind_resample_s', type=float, default=12.0)
parser.add_argument('--wind_ramp_s', type=float, default=2.0)
parser.add_argument('--wind_shear_exponent', type=float, default=None, help='Override checkpoint shear exponent')
parser.add_argument('--wind_reference_height_m', type=float, default=None)
parser.add_argument('--wind_vortex_radius', type=float, default=4.0)

parser.add_argument('--kit', action='store_true', help='Launch Isaac Sim Kit (not needed for default Newton task)')
parser.add_argument('--teacher_only', action='store_true', help='Play the frozen pretrained policy without SAC corrections')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--video', action='store_true', help='Record an MP4 using the Newton GL visualizer')
parser.add_argument('--video_length', type=int, default=400, help='Recorded steps (400 = 8 simulated seconds)')
parser.add_argument('--video_interval', type=int, default=0, help='0 = one clip; N = recurring clips')
parser.add_argument('--video_fps', type=int, default=50)
parser.add_argument('--video_headless', action='store_true', help='Record without opening a window')
parser.add_argument('--video_static_camera', action='store_true', help='Disable the G1-following camera during recording')
parser.add_argument('--output_dir', type=Path, default=None, help='Evaluation output folder for video and metrics')
args = parser.parse_args()
sim_app = None
if args.kit:
    from isaaclab.app import AppLauncher
    sim_app = AppLauncher(headless=True).app
import torch
from g1_wind_sac_v5.cfg import G1WindSACEnvCfg
from g1_wind_sac_v5.env import G1WindSACEnv
from g1_wind_sac_v5.sac import SACAgent, SACConfig
from g1_wind_sac_v5.recording import configure_video, follow_robot_camera, camera_capture_active
from g1_wind_sac_v5.history import ObservationHistory
try:
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    assert ckpt.get('format') == 'g1_wind_residual_sac_v1', 'Not a residual SAC checkpoint.'
    cfg = G1WindSACEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.wind_enable = not args.wind_off
    saved_env = ckpt.get('environment', {})
    cfg.wind_model = args.wind_model or saved_env.get('wind_model', 'torso')
    cfg.wind_field = args.wind_field or saved_env.get('wind_field','uniform')
    cfg.wind_max_force_n = (args.wind_force_cap_n if args.wind_force_cap_n is not None
                                 else saved_env.get('wind_force_cap_n', 0.0 if cfg.wind_model=='panels' else 60.0))
    cfg.wind_fixed_heading_deg = args.wind_fixed_heading_deg
    cfg.wind_min_mps = args.wind_min
    cfg.wind_calm_probability = args.wind_calm_probability
    cfg.wind_pressure_coefficient = args.wind_pressure_coefficient
    cfg.wind_gust_amplitude_mps = args.wind_gust_amplitude
    cfg.wind_gust_period_s = args.wind_gust_period
    cfg.wind_gust_sigma_fraction = (args.wind_gust_sigma_fraction if args.wind_gust_sigma_fraction is not None else saved_env.get('wind_gust_sigma_fraction',0.15))
    cfg.wind_gust_tau_s = (args.wind_gust_tau_s if args.wind_gust_tau_s is not None else saved_env.get('wind_gust_tau_s',0.8))
    cfg.wind_resample_s = args.wind_resample_s
    cfg.wind_ramp_s = args.wind_ramp_s
    cfg.wind_shear_exponent = (args.wind_shear_exponent if args.wind_shear_exponent is not None else saved_env.get('wind_shear_exponent',0.0))
    cfg.wind_reference_height_m = (args.wind_reference_height_m if args.wind_reference_height_m is not None else saved_env.get('wind_reference_height_m',10.0))
    cfg.wind_vortex_core_radius_m = args.wind_vortex_radius
    cfg.wind_max_mps = args.wind_max
    cfg.seed = args.seed
    mode = 'teacher' if args.teacher_only else 'sac'
    out_dir = args.output_dir or (Path(__file__).resolve().parent / 'runs' / 'play' / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{mode}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.video:
        configure_video(cfg, output_dir=out_dir / 'videos', video_length=args.video_length,
                        video_interval=args.video_interval, fps=args.video_fps,
                        headless=args.video_headless, filename_prefix=mode)
        print(f'[VIDEO] Native Newton GL MP4 recording enabled: {out_dir / "videos"}')
    env = G1WindSACEnv(cfg)
    try:
        obs, _ = env.reset()
        history_len = int(ckpt['sac_cfg'].get('history_len', 1))
        history = ObservationHistory(obs['policy'], history_len)
        current_obs = history.flattened()
        odim, adim = current_obs.shape[-1], env.action_manager.total_action_dim
        assert tuple(ckpt['dimensions']) == (odim, adim), 'G1 policy observation/action mismatch.'
        ag = SACAgent(odim, adim, env.device, SACConfig(**ckpt['sac_cfg']))
        ag.load_state_dict(ckpt['agent'])
        rewards, terminated_count, truncated_count = 0.0, 0, 0
        episode_return = torch.zeros(args.num_envs, device=env.device)
        episode_length = torch.zeros(args.num_envs, dtype=torch.int64, device=env.device)
        episode_id = torch.zeros(args.num_envs, dtype=torch.int64, device=env.device)
        episode_velocity_sq = torch.zeros(args.num_envs, device=env.device)
        total_vel_sq = total_vel_count = 0.0
        episodes = []
        with (out_dir / 'evaluation_metrics.csv').open('w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['step', 'reward_mean', 'terminated_count', 'truncated_count'])
            for step in range(args.steps):
                if args.video and not args.video_static_camera and camera_capture_active(
                    step, args.video_length, args.video_interval
                ):
                    follow_robot_camera(env)
                with torch.inference_mode():
                    action = ag.act(current_obs, teacher_only=args.teacher_only, deterministic=True)
                    applied_command = env.command_manager.get_command('base_velocity').clone()
                    obs, reward, terminated, truncated, extras = env.step(action)
                    done = terminated | truncated
                    last_obs = extras['final_obs']['policy'] if bool(done.any()) else None
                    _, current_obs = history.advance(obs['policy'], done, last_obs)
                    episode_return += reward
                    episode_length += 1
                    cmd = applied_command
                    measured = env._robot.data.root_lin_vel_b.torch[:, :2]
                    vel_sq = (measured - cmd[:, :2]).square().sum(dim=-1)
                    episode_velocity_sq += vel_sq
                    total_vel_sq += float(vel_sq.sum().item())
                    total_vel_count += args.num_envs
                    if bool(done.any()):
                        for i in done.nonzero(as_tuple=True)[0].tolist():
                            episodes.append({'env_id':i, 'episode_id':int(episode_id[i].item()),
                                             'end_step': step+1, 'length_steps':int(episode_length[i].item()),
                                             'return':float(episode_return[i].item()),
                                             'velocity_rmse_mps':float((episode_velocity_sq[i]/episode_length[i]).sqrt().item()),
                                             'fall':int(terminated[i].item()),
                                             'time_limit':int((truncated[i] & ~terminated[i]).item())})
                        episode_velocity_sq[done] = 0
                        episode_return[done] = 0
                        episode_length[done] = 0
                        episode_id[done] += 1
                    rewards += float(reward.mean().item())
                    terminated_count += int(terminated.sum().item())
                    truncated_count += int((truncated & ~terminated).sum().item())
                writer.writerow([step + 1, float(reward.mean().item()),
                                 int(terminated.sum().item()), int((truncated & ~terminated).sum().item())])
        with (out_dir/'episodes.csv').open('w',newline='') as ef:
            ep_writer = csv.DictWriter(ef, fieldnames=['env_id','episode_id','end_step',
                'length_steps','return','velocity_rmse_mps','fall','time_limit'])
            ep_writer.writeheader()
            ep_writer.writerows(episodes)
        summary = {'velocity_rmse_mps':(total_vel_sq/total_vel_count)**0.5 if total_vel_count else None,
               'episodes_completed':len(episodes),
               'episode_fall_rate':terminated_count/len(episodes) if episodes else None,
               'teacher_only':args.teacher_only, 'wind_max_mps':args.wind_max if not args.wind_off else 0,
               'mean_reward_per_step':rewards/args.steps,
               'terminations':terminated_count, 'truncations':truncated_count,
               'num_envs':args.num_envs, 'output_dir':str(out_dir),
               'wind_model':cfg.wind_model, 'wind_field':cfg.wind_field,
               'wind_force_cap_n':cfg.wind_max_force_n, 'wind_min_mps':cfg.wind_min_mps,
               'seed':args.seed, 'history_len':history_len,
               'wind_gust_sigma_fraction':cfg.wind_gust_sigma_fraction,
               'wind_gust_tau_s':cfg.wind_gust_tau_s,
               'wind_shear_exponent':cfg.wind_shear_exponent,
               'wind_reference_height_m':cfg.wind_reference_height_m}
        (out_dir/'evaluation_summary.json').write_text(json.dumps(summary,indent=2))
        print(summary)
    finally:
        env.close()
finally:
    if sim_app is not None:
        sim_app.close()
