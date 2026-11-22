"""Stationary-robot proxy-box drag check against an EXTERNAL CdA reference.

This does not calibrate distributed moments or prove equivalence to mesh-pressure.
Run first with --wind_speed 10 and compare force and moment across directions.
"""
import argparse,csv,json,math
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--speed',type=float,default=10.)
    p.add_argument('--target_cda',type=float,default=.43,
                   help='External literature reference for other G1 configuration, NOT ground truth for this one')
    p.add_argument('--headings',type=float,nargs='+',default=[0,45,90,135,180,225,270,315])
    p.add_argument('--pressure_coefficient',type=float,default=1.)
    p.add_argument('--out_dir',type=Path,default=Path(__file__).resolve().parent / 'runs' / 'calibration')
    a=p.parse_args()
    if a.speed<=0 or a.target_cda<=0 or a.pressure_coefficient<=0:p.error('positive speed/CdA/Cp required')
    import torch
    from g1_wind_sac_v5.cfg import G1WindSACEnvCfg
    from g1_wind_sac_v5.env import G1WindSACEnv
    from g1_wind_sac_v5.aerodynamics import box_pressure_wrenches,effective_cda
    cfg=G1WindSACEnvCfg(); cfg.scene.num_envs=1;cfg.wind_enable=False
    env=G1WindSACEnv(cfg)
    try:
        env.reset(); d=env._robot.data
        def t(x): return x.torch if hasattr(x,'torch') else x
        p_w=t(d.body_com_pos_w).clone();q=t(d.body_link_quat_w).clone()
        v=torch.zeros_like(p_w);omega=torch.zeros_like(p_w)
        # Static snapshot only; no physics movement/wind-driven posture adaptation.
        base_pos=t(d.root_pos_w).clone()
        rows=[]
        for deg in a.headings:
            th=math.radians(deg)
            wind=torch.tensor([a.speed*math.cos(th),a.speed*math.sin(th),0.],device=env.device)
            winds=wind[None,None,:].expand_as(p_w)
            f,torque=box_pressure_wrenches(winds,p_w,q,v,omega,env._box_dimensions,
                              rho=cfg.wind_rho,cp=a.pressure_coefficient)
            force=f.sum(dim=1)
            moment=(torque+torch.cross(p_w-base_pos[:,None,:],f,dim=-1)).sum(dim=1)
            rows.append({'heading_deg':deg,'wind_mps':a.speed,'cda_m2':float(effective_cda(force,wind[None],cfg.wind_rho)[0]),
                'force_along_wind_n':float((force[0]*wind/a.speed).sum()),
                'resultant_force_n':float(torch.linalg.vector_norm(force)),
                'root_moment_n_m':float(torch.linalg.vector_norm(moment)),
                'moment_x_n_m':float(moment[0,0]),'moment_y_n_m':float(moment[0,1]),'moment_z_n_m':float(moment[0,2])})
        a.out_dir.mkdir(parents=True,exist_ok=True)
        with (a.out_dir/'calibration_sweep.csv').open('w',newline='') as ff:
            wr=csv.DictWriter(ff,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
        cda=rows[0]['cda_m2']; scale=a.target_cda/cda if cda>0 else None
        (a.out_dir/'calibration_summary.json').write_text(json.dumps({
            'reference_cda_m2':a.target_cda, 'current_cp':a.pressure_coefficient,
            'first_heading_effective_cda_m2':cda,
            'candidate_cp_for_first_heading_only':a.pressure_coefficient*scale if scale else None,
            'warning':'Single-stance calibration only. Comparison target is from a different G1 model; per-link force, moments, pose and yaw direction still need mesh-level validation.'},indent=2))
        print(f'[CALIBRATION] Static drag and base moments saved in {a.out_dir}',flush=True)
        print(f'[CALIBRATION] Effective CdA at first heading={cda:.3f}m² vs reference={a.target_cda:.3f}m²',flush=True)
    finally: env.close()

if __name__=='__main__': main()
