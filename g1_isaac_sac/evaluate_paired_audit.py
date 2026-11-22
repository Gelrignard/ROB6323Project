"""Matched-wind Isaac Lab evaluation of frozen G1 teacher and residual SAC.

Both controllers receive the same prescribed, reset-independent wind sequence
and fixed velocity command. Outputs one CSV per run plus paired_summary.csv.
This is a COMPATIBILITY/EVALUATION experiment, NOT training.

Each run starts a fresh Isaac Lab environment; initial conditions are seeded,
but subsequent robot reset trajectories differ if a controller falls earlier.
Compare distributions of episode outcomes, not frame-wise paired states.
"""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


def get_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--wind_speeds',type=float,nargs='+',default=[0.,10.,20.,30.,42.])
    p.add_argument('--steps',type=int,default=1500)
    p.add_argument('--num_envs',type=int,default=16)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--heading_deg',type=float,default=0.)
    p.add_argument('--gust_fraction',type=float,default=0.,help='Deterministic sinusoidal perturbation magnitude')
    p.add_argument('--gust_period_s',type=float,default=8.)
    p.add_argument('--shear_exponent',type=float,default=0.14)
    p.add_argument('--wind_pressure_coefficient',type=float,default=1.)
    p.add_argument('--command_forward_mps',type=float,default=0.5)
    p.add_argument('--video',action='store_true')
    p.add_argument('--video_at_wind',type=float,default=20.,help='Only record this wind-speed case')
    p.add_argument('--video_length',type=int,default=400)
    p.add_argument('--output_dir',type=Path,default=None)
    p.add_argument('--kit',action='store_true')
    args=p.parse_args()
    if not args.checkpoint.is_file(): p.error('checkpoint does not exist')
    if args.steps<1 or args.num_envs<1 or any(x<0 for x in args.wind_speeds): p.error('invalid steps/envs/wind')
    if args.video_length<1 or args.video_length>args.steps: p.error('video_length must be <= steps')
    return args


def main():
    args=get_args()
    simulation_app=None
    if args.kit:
        from isaaclab.app import AppLauncher
        simulation_app=AppLauncher(headless=True).app
    import torch
    from g1_wind_sac_v5.cfg import G1WindSACEnvCfg
    from g1_wind_sac_v5.env import G1WindSACEnv
    from g1_wind_sac_v5.sac import SACAgent,SACConfig
    from g1_wind_sac_v5.history import ObservationHistory
    from g1_wind_sac_v5.audit import exclusive_outcomes
    from g1_wind_sac_v5.recording import configure_video,follow_robot_camera,camera_capture_active
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if ckpt.get('format')!='g1_wind_residual_sac_v1':
        raise ValueError('Expected a g1_wind_residual_sac_v1 checkpoint')
    out=args.output_dir or Path(__file__).resolve().parent / 'runs' / 'paired_evaluation'/datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    try:
        for wind in args.wind_speeds:
            for mode in ('teacher','sac'):
                torch.manual_seed(args.seed)
                cfg=G1WindSACEnvCfg()
                cfg.seed=args.seed; cfg.scene.num_envs=args.num_envs
                cfg.wind_enable=True; cfg.wind_model='panels'; cfg.wind_field='uniform'
                cfg.wind_max_force_n=0.0
                cfg.wind_max_mps=wind; cfg.wind_min_mps=wind; cfg.wind_calm_probability=0.
                cfg.wind_pressure_coefficient=args.wind_pressure_coefficient
                cfg.wind_shear_exponent=args.shear_exponent
                cfg.commands.base_velocity.ranges.lin_vel_x=(args.command_forward_mps,)*2
                cfg.commands.base_velocity.ranges.lin_vel_y=(0.,0.)
                cfg.commands.base_velocity.ranges.ang_vel_z=(0.,0.)
                cfg.commands.base_velocity.resampling_time_range=(3600.,3600.)
                for field in ('rel_standing_envs','rel_heading_envs'):
                    if hasattr(cfg.commands.base_velocity,field): setattr(cfg.commands.base_velocity,field,0.)
                if hasattr(cfg.commands.base_velocity,'heading_command'):
                    cfg.commands.base_velocity.heading_command=False
                label=f'{mode}_wind_{wind:g}mps'
                case=out/label;case.mkdir(exist_ok=True)
                do_video=args.video and abs(wind-args.video_at_wind)<1e-6
                if do_video:
                    configure_video(cfg,output_dir=case/'videos',video_length=args.video_length,
                                    headless=True,filename_prefix=label)
                env=G1WindSACEnv(cfg)
                try:
                    env.capture_eval_telemetry=True
                    obs,_=env.reset()
                    env.enable_prescribed_wind(wind,args.heading_deg,args.gust_fraction,args.gust_period_s)
                    hist_len=int(ckpt['sac_cfg'].get('history_len',1))
                    hist=ObservationHistory(obs['policy'],hist_len)
                    raw=hist.flattened()
                    odim=raw.shape[-1]; adim=env.action_manager.total_action_dim
                    if tuple(ckpt['dimensions'])!=(odim,adim):
                        raise ValueError(f'checkpoint expects {ckpt["dimensions"]}, got {(odim,adim)}')
                    agent=SACAgent(odim,adim,env.device,SACConfig(**ckpt['sac_cfg']))
                    agent.load_state_dict(ckpt['agent']); agent.actor.eval()
                    n=args.num_envs
                    ep_lengths=torch.zeros(n,device=env.device,dtype=torch.int32)
                    ep_reward=torch.zeros(n,device=env.device)
                    ep_velsq=torch.zeros(n,device=env.device)
                    ep_wind_peak=torch.zeros(n,device=env.device)
                    ep_ids=[0]*n
                    fall_count=timeout_count=0
                    vel_error_sum=reward_sum=torso_wind_sum=force_sum=0.
                    completed=[]
                    with (case/'step_metrics.csv').open('w',newline='') as f:
                        writer=csv.writer(f)
                        writer.writerow(['step','mean_reward','falls_this_step','timeouts_this_step','mean_xy_vel_sq','mean_torso_wind_mps','mean_aero_net_force_n'])
                        for k in range(args.steps):
                            if do_video and camera_capture_active(k,args.video_length,0):follow_robot_camera(env)
                            with torch.inference_mode():
                                act=agent.act(raw,teacher_only=(mode=='teacher'),deterministic=True)
                                cmd=env.command_manager.get_command('base_velocity').clone()
                                nxt,rew,term,trunc,extras=env.step(act)
                                falls,timeouts,done=exclusive_outcomes(term,trunc)
                                final=extras['final_obs']['policy'] if bool(done.any()) else None
                                _,raw=hist.advance(nxt['policy'],done,final)
                                telemetry=extras['wind_eval_telemetry']
                                xyvel=telemetry['root_lin_vel_b_mps'][:,:2]
                                velsq=(xyvel-cmd[:,:2]).square().sum(-1)
                                torso=torch.linalg.vector_norm(telemetry['wind_torso_world_mps'],dim=-1)
                                aero=torch.linalg.vector_norm(telemetry['wind_net_force_world_n'],dim=-1)
                                ep_lengths+=1;ep_reward+=rew;ep_velsq+=velsq;ep_wind_peak=torch.maximum(ep_wind_peak,torso)
                                fall_count+=int(falls.sum());timeout_count+=int(timeouts.sum())
                                reward_sum+=rew.mean().item();vel_error_sum+=velsq.sum().item()
                                torso_wind_sum+=torso.mean().item();force_sum+=aero.mean().item()
                                if bool(done.any()):
                                    for i in done.nonzero(as_tuple=True)[0].tolist():
                                        completed.append({'env_id':i,'episode_id':ep_ids[i],'end_step':k+1,
                                            'duration_s':float(ep_lengths[i].item()*env.step_dt),
                                            'return':float(ep_reward[i].item()),
                                            'velocity_rmse_mps':float((ep_velsq[i]/ep_lengths[i]).sqrt().item()),
                                            'peak_torso_wind_mps':float(ep_wind_peak[i].item()),
                                            'fall':int(falls[i].item()),'timeout':int(timeouts[i].item())})
                                        ep_ids[i]+=1
                                    ep_lengths[done]=0;ep_reward[done]=0;ep_velsq[done]=0;ep_wind_peak[done]=0
                                writer.writerow([k+1,float(rew.mean()),int(falls.sum()),int(timeouts.sum()),
                                    float(velsq.mean()),float(torso.mean()),float(aero.mean())])
                    with (case/'episodes.csv').open('w',newline='') as f:
                        writer=csv.DictWriter(f,fieldnames=['env_id','episode_id','end_step','duration_s','return',
                                                             'velocity_rmse_mps','peak_torso_wind_mps','fall','timeout'])
                        writer.writeheader();writer.writerows(completed)
                    result={'controller':mode,'wind_reference_mps':wind,'episodes_completed':len(completed),
                        'falls':fall_count,'timeouts':timeout_count,'episode_fall_rate':fall_count/len(completed) if completed else None,
                        'censored_incomplete_episodes':int((ep_lengths>0).sum()),
                        'mean_reward_per_step':reward_sum/args.steps,
                        'velocity_tracking_rmse_mps':(vel_error_sum/(args.steps*n))**0.5,
                        'mean_torso_wind_mps':torso_wind_sum/args.steps,
                        'mean_resultant_force_n':force_sum/args.steps}
                    (case/'summary.json').write_text(json.dumps(result,indent=2))
                    rows.append(result)
                    print(f'[PAIRED] {label} falls={fall_count}/{len(completed)} '
                          f'rmse={result["velocity_tracking_rmse_mps"]:.3f}m/s',flush=True)
                finally:
                    env.close()
        with (out/'paired_summary.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
        (out/'experiment.json').write_text(json.dumps({'checkpoint':str(args.checkpoint.resolve()),
            'seed':args.seed,'speeds_mps':args.wind_speeds,'steps_per_case':args.steps,
            'num_envs':args.num_envs,'heading_deg':args.heading_deg,
            'gust_fraction':args.gust_fraction,'gust_period_s':args.gust_period_s,
            'shear_exponent':args.shear_exponent,'wind_pressure_coefficient':args.wind_pressure_coefficient,
            'note':'Deterministic wind independent of resets; initial states seeded; subsequent reset states not paired.'},indent=2))
        print(f'[DONE] Paired comparison saved in {out}',flush=True)
    finally:
        if simulation_app is not None:simulation_app.close()

if __name__=='__main__': main()
