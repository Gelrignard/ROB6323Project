from dataclasses import replace
import pytest
import torch
from g1_wind_sac_v5.teacher import FrozenG1Teacher
from g1_wind_sac_v5.sac import SACAgent, SACConfig, ReplayBuffer
from g1_wind_sac_v5.wind_math import wind_drag


def test_teacher_checkpoint_v5_and_forward(tmp_path):
    obs_dim, act_dim = 16, 4
    original = FrozenG1Teacher(obs_dim, act_dim)
    model = {'actor_state_dict': {'mlp.' + k: v.detach().clone() for k, v in original.mlp.state_dict().items()},
             'critic_state_dict': {'not': torch.tensor(1.)}}
    checkpoint = tmp_path / 'model_100.pt'
    torch.save(model, checkpoint)
    loaded = FrozenG1Teacher(obs_dim, act_dim)
    loaded.load_ppo_checkpoint(str(checkpoint))
    x = torch.randn(13, obs_dim)
    assert torch.equal(loaded(x), original(x))
    assert not any(p.requires_grad for p in loaded.parameters())


def test_teacher_checkpoint_wrong_shape_fails_loudly(tmp_path):
    teacher = FrozenG1Teacher(9, 4)
    source = {'actor_state_dict': {'mlp.' + k: v.detach() for k, v in teacher.mlp.state_dict().items()}}
    path = tmp_path / 'actor.pt'
    torch.save(source, path)
    with pytest.raises(ValueError, match='incompatible'):
        FrozenG1Teacher(11, 4).load_ppo_checkpoint(str(path))


def test_sac_initial_policy_is_teacher_centered_and_bounded():
    torch.manual_seed(123)
    cfg = SACConfig(residual_scale=0.2)
    agent = SACAgent(12, 4, 'cpu', cfg)
    obs = torch.randn(16, 12)
    teacher = agent.act(obs, teacher_only=True)
    a = agent.act(obs, deterministic=True)
    assert torch.allclose(a, teacher, atol=1e-7)
    stochastic = agent.act(obs)
    assert torch.all((stochastic - teacher).abs() <= 0.200001)


def test_sac_update_changes_residual_not_pretrained_teacher():
    torch.manual_seed(2)
    cfg = SACConfig(batch_size=32, replay_capacity=64, residual_penalty=0.1)
    agent = SACAgent(11, 3, 'cpu', cfg)
    before = {k:v.detach().clone() for k,v in agent.actor.teacher.state_dict().items()}
    rb = ReplayBuffer(64, 11, 3, torch.device('cpu'))
    obs = torch.randn(80, 11)
    with torch.no_grad():
        act = agent.act(obs)
    rb.add(obs, act, torch.randn(80), torch.randn(80,11), torch.zeros(80, dtype=torch.bool))
    assert rb.size == 64
    info = agent.update(rb.sample(32))
    assert info['critic_loss'] >= 0
    assert info['alpha'] > 0
    assert all(torch.equal(v, before[k]) for k, v in agent.actor.teacher.state_dict().items())
    assert agent.num_updates == 1


def test_wind_force_cda():
    wind = torch.tensor([[10., 0., 0.]])
    force = wind_drag(wind, torch.zeros_like(wind))
    assert torch.allclose(force, torch.tensor([[26.3375, 0., 0.]]), atol=1e-4)
