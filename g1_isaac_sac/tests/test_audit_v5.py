import math
import torch
from g1_wind_sac_v5.aerodynamics import box_pressure_wrenches, box_patch_quadrature, effective_cda
from g1_wind_sac_v5.audit import exclusive_outcomes,prescribed_uniform_wind,SuccessGatedWindCurriculum
from g1_wind_sac_v5.sac import BinnedReplayBuffer,SACAgent,SACConfig


def setup():
    body=torch.zeros(1,1,3)
    quat=torch.tensor([[[0.,0.,0.,1.]]])
    vel=body.clone(); omega=body.clone(); size=torch.tensor([[.2,.3,.4]])
    return body,quat,vel,omega,size


def test_quadrature_area_and_uniform_stationary_torque():
    o,n,a=box_patch_quadrature(torch.tensor([[.2,.3,.4]]),subdivisions=2)
    assert o.shape == (1,24,3)
    assert abs(a.sum().item()-2*(.2*.3+.3*.4+.2*.4))<1e-6
    p,q,v,omega,d=setup()
    f,t=box_pressure_wrenches(torch.tensor([[[10.,0,0]]]),p,q,v,omega,d,rho=1.,cp=1.)
    assert torch.allclose(f,torch.tensor([[[6.,0,0]]]),atol=1e-5)
    assert torch.allclose(t,torch.zeros_like(t),atol=1e-6)
    assert abs(effective_cda(f[:,0],torch.tensor([[10.,0,0]]),1.).item()-.12)<1e-5


def test_spinning_body_has_aerodynamic_damping_torque():
    p,q,v,omega,d=setup()
    omega[...,2]=9.0
    f,t=box_pressure_wrenches(torch.zeros(1,1,3),p,q,v,omega,d,rho=1,cp=1)
    assert abs(float(t[0,0,2]))>1e-8, f'Torque must depend on surface point velocity: {t}'
    assert t[0,0,2] < 0, 'Aerodynamic torque should oppose rotation'


def test_outcome_exclusivity():
    fall,to,done=exclusive_outcomes(torch.tensor([1,1,0,0]),torch.tensor([1,0,1,0]))
    assert fall.tolist()==[True,True,False,False]
    assert to.tolist()==[False,False,True,False]
    assert done.tolist()==[True,True,True,False]
    assert int(fall.sum()+to.sum())==int(done.sum())


def test_prescribed_wind_independent_of_resets():
    x=prescribed_uniform_wind(3.,8,25,heading_deg=32,gust_fraction=.15)
    y=prescribed_uniform_wind(3.,8,25,heading_deg=32,gust_fraction=.15)
    assert torch.equal(x,y)
    assert x.shape==(8,3)
    assert torch.all(x[:,2]==0)
    w0=prescribed_uniform_wind(3.,3,25,gust_fraction=0)
    assert torch.allclose(w0[:,0],torch.full((3,),25.))


def test_gate_rejects_failures_and_requires_two_success_windows():
    g=SuccessGatedWindCurriculum(4.,10.,increment=2.,min_episodes=4,
                                 survival_threshold=.75,required_windows=2)
    assert g.check() is None
    g.add_completed(torch.tensor([1,1,1,0],dtype=torch.bool),torch.ones(4,dtype=torch.bool))
    d=g.check(); assert not d.advanced and g.ceiling==4.
    g.add_completed(torch.tensor([0,0,0,0],dtype=torch.bool),torch.ones(4,dtype=torch.bool))
    assert not g.check().advanced
    g.add_completed(torch.tensor([0,0,0,0],dtype=torch.bool),torch.ones(4,dtype=torch.bool))
    assert g.check().advanced and g.ceiling==6.


def test_gate_exclusion():
    g=SuccessGatedWindCurriculum(4,8, min_episodes=2,required_windows=1)
    g.add_completed(torch.tensor([0,1,0],dtype=torch.bool),torch.tensor([1,0,0],dtype=torch.bool))
    assert g.episodes==1 and g.falls==0
    assert g.check() is None


def test_binned_replay_persists_wind_bands():
    b=BinnedReplayBuffer(400,8,2,torch.device('cpu'))
    for wind in (0.0,5.0,19.0,43.0):
        obs=torch.randn(60,8); act=torch.randn(60,2)
        b.add(obs,act,torch.zeros(60),obs.clone(),torch.zeros(60),torch.full((60,),wind))
    assert b.bin_sizes==[60,60,60,60]
    vals=b.sample(32)
    assert len(vals)==5 and vals[0].shape==(32,8)
    # Repeated strong wind cannot delete calm examples from its independent ring.
    for _ in range(4):
        x=torch.randn(30,8)
        b.add(x,torch.randn(30,2),torch.zeros(30),x,torch.zeros(30),torch.full((30,),43.))
    assert b.bin_sizes[0]==60


def test_actor_diagnostics_and_alpha_floor():
    a=SACAgent(16,3,'cpu',SACConfig(alpha_floor=.001,history_len=1,alpha_init=.01))
    b=BinnedReplayBuffer(200,16,3,torch.device('cpu'))
    x=torch.randn(50,16)
    b.add(x,torch.randn(50,3),torch.randn(50),torch.randn(50,16),torch.zeros(50),torch.zeros(50))
    metrics=a.update(b.sample(16))
    assert all(math.isfinite(v) for v in metrics.values())
    assert metrics['alpha']>=.001-1e-7
    assert 'actor_entropy' in metrics and 'actor_log_std_mean' in metrics
