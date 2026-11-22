"""CPU tests for panel pressure, world-frame torque, vortex and history resets."""
import math
import torch
from g1_wind_sac_v5.aerodynamics import (box_pressure_wrenches, proxy_box_dimensions,
                                           tornado_field_world)
from g1_wind_sac_v5.history import ObservationHistory
from g1_wind_sac_v5.sac import SACAgent, SACConfig


def tensors(n=1,b=1):
    pos = torch.zeros(n,b,3)
    q = torch.zeros(n,b,4)
    q[...,3] = 1.0  # Isaac Lab 3.0 xyzw
    vel = torch.zeros(n,b,3)
    omega = torch.zeros(n,b,3)
    d = torch.tensor([[.2, .3, .4]]).expand(b,-1).clone()
    return pos,q,vel,omega,d


def test_pressure_newton_law_and_quadratic_speed():
    p,q,v,w,d=tensors()
    wind=torch.tensor([[[10.,0,0]]])
    force, torque = box_pressure_wrenches(wind,p,q,v,w,d,rho=1.,cp=1.)
    assert torch.allclose(force,torch.tensor([[[6.,0,0]]]),atol=1.e-5) # A_yz=.12, p=50
    assert torch.allclose(torque,torch.zeros_like(torque),atol=1.e-6)
    force2,_=box_pressure_wrenches(2*wind,p,q,v,w,d,rho=1.,cp=1.)
    assert torch.allclose(force2,4*force)


def test_body_rotation_xyzw():
    p,q,v,w,d=tensors()
    q[...,2]=math.sqrt(.5)
    q[...,3]=math.sqrt(.5)  # rotate 90 deg about z
    f,_ = box_pressure_wrenches(torch.tensor([[[10.,0,0]]]),p,q,v,w,d,rho=1.,cp=1.)
    # Box new normal world x comes from local -y -> area local x*z = .08
    assert torch.allclose(f,torch.tensor([[[4.,0,0]]]),atol=1.e-5)


def test_pressure_zero_when_air_relative_velocity_zero():
    p,q,v,w,d=tensors()
    v[...,0]=14
    f,t=box_pressure_wrenches(v,p,q,v,w,d)
    assert torch.count_nonzero(f)==0
    assert torch.count_nonzero(t)==0


def test_vortex_zero_at_core_and_vmax_at_radius():
    pts=torch.tensor([[[0.,0.,0.],[4.,0.,0.]]])
    v=tornado_field_world(pts,20.,core_radius=4.,inflow_fraction=0.)
    assert torch.all(v[0,0]==0)
    assert torch.allclose(v[0,1],torch.tensor([0.,20.,0.]))


def test_body_dimensions_positive_and_heterogeneous():
    dims=proxy_box_dimensions(['torso_link','head_link','left_knee_link','left_ankle_roll_link'])
    assert (dims>0).all()
    assert len({tuple(x.tolist()) for x in dims})==4


def test_history_preserves_terminal_and_resets_only_done_envs():
    initial=torch.tensor([[0.,1.],[10.,11.]])
    h=ObservationHistory(initial,3)
    terminal,nxt=h.advance(torch.tensor([[2.,3.],[12.,13.]]),torch.tensor([False,True]),
                           torch.tensor([[998.,999.],[20.,21.]]))
    assert torch.equal(terminal[0,:2],torch.tensor([2.,3.]))
    assert torch.equal(terminal[1,:2],torch.tensor([20.,21.]))
    assert torch.equal(nxt[0],torch.tensor([2.,3.,0.,1.,0.,1.]))
    assert torch.equal(nxt[1],torch.tensor([12.,13.,12.,13.,12.,13.]))


def test_teacher_receives_only_newest_obs_history():
    a=SACAgent(24,4,'cpu',SACConfig(history_len=3,residual_scale=.5))
    obs=torch.randn(5,24)
    ref=a.actor.teacher(obs[:,:8])
    assert torch.allclose(a.act(obs,teacher_only=True),ref)
    assert torch.allclose(a.act(obs,deterministic=True),ref)
    sample=a.act(obs)
    assert torch.all((sample-ref).abs()<=.500001)
