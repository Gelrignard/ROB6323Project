"""Twin-Q SAC for bounded residual corrections to a frozen pretrained actor.

The policy acts in the ORIGINAL Isaac Lab G1 action space:
  total_action = teacher(obs) + residual_scale * tanh(z), z ~ Gaussian.
Only the residual's entropy is optimized (translation by deterministic teacher is measure preserving).
"""
import copy
import math
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from .teacher import FrozenG1Teacher


@dataclass
class SACConfig:
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 512
    updates_per_step: int = 1
    learning_starts: int = 16384  # replay TRANSITIONS, not vectorized simulator steps
    teacher_only_steps: int = 100  # vectorized simulator steps for rollout without residual exploration
    replay_capacity: int = 262144
    residual_scale: float = 0.15  # action units; NOT joint radians
    actor_lr: float = 1.0e-4
    critic_lr: float = 3.0e-4
    alpha_lr: float = 1.0e-4
    alpha_init: float = 0.01
    target_entropy_scale: float = 1.0
    residual_penalty: float = 0.25
    grad_clip: float = 5.0
    history_len: int = 1
    alpha_floor: float = 0.0  # optional entropy regularization floor, ablation only


class ResidualActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, scale: float, teacher_obs_dim: int):
        super().__init__()
        self.teacher_obs_dim = teacher_obs_dim
        self.teacher = FrozenG1Teacher(teacher_obs_dim, action_dim)
        self.residual_scale = float(scale)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 2 * action_dim),
        )
        # Mean 0; small but nonzero exploration initially. Teacher deterministic output is preserved.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            self.net[-1].bias[action_dim:] = -3.0
        self.action_dim = action_dim

    def sample(self, obs: torch.Tensor, deterministic: bool = False):
        mean, log_std = self.net(obs).chunk(2, dim=-1)
        log_std = log_std.clamp(-5.0, 1.0)
        std = log_std.exp()
        z = mean if deterministic else mean + std * torch.randn_like(mean)
        unit = torch.tanh(z)
        # Squashed Gaussian log density in normalized residual coordinates [-1,1].
        # Constant scale transformation is omitted consistently from logπ and target entropy.
        gaussian_logp = -0.5 * (((z - mean) / std).square() + 2 * log_std + math.log(2 * math.pi))
        logp = gaussian_logp.sum(-1, keepdim=True)
        logp -= (2 * (math.log(2) - z - F.softplus(-2*z))).sum(-1, keepdim=True)
        with torch.no_grad():
            base = self.teacher(obs[..., :self.teacher_obs_dim])
        return base + self.residual_scale * unit, logp, unit

    def forward(self, obs, deterministic=True):
        return self.sample(obs, deterministic=deterministic)[0]


class TwinQ(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        def q_net():
            return nn.Sequential(
                nn.Linear(obs_dim + act_dim, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 1),
            )
        self.q1, self.q2 = q_net(), q_net()

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        x = torch.cat((obs, act), dim=-1)
        return self.q1(x), self.q2(x)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.obs = torch.empty((capacity, obs_dim), device=device)
        self.next_obs = torch.empty_like(self.obs)
        self.actions = torch.empty((capacity, action_dim), device=device)
        self.rewards = torch.empty((capacity, 1), device=device)
        self.terminated = torch.empty((capacity, 1), device=device)
        self.size = 0
        self.pos = 0

    @torch.no_grad()
    def add(self, obs, action, rewards, next_obs, terminated):
        n = obs.shape[0]
        if n > self.capacity:
            obs, action, rewards, next_obs, terminated = (
                x[-self.capacity:] for x in (obs, action, rewards, next_obs, terminated)
            )
            n = self.capacity
        idx = (self.pos + torch.arange(n, device=self.device)) % self.capacity
        self.obs[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = rewards.reshape(-1, 1)
        self.next_obs[idx] = next_obs
        self.terminated[idx] = terminated.reshape(-1, 1).float()
        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.capacity, self.size + n)

    def sample(self, batch_size):
        indices = torch.randint(self.size, (batch_size,), device=self.device)
        return (self.obs[indices], self.actions[indices], self.rewards[indices],
                self.next_obs[indices], self.terminated[indices])


class BinnedReplayBuffer:
    """Replay reservoir partitioned by *nominal wind* to retain old wind regimes.

    Fixed equal-capacity rings for calm (<=1), weak (1..10), strong (10..25),
    extreme (>25) m/s. Sample evenly across nonempty bins; labels represent
    nominal 10 m speed and are not measured body-level gust exposure.
    This is a new replay distribution and should be tested as an ablation.
    """
    def __init__(self, capacity, obs_dim, action_dim, device):
        if capacity<4: raise ValueError('Need >=4 replay slots for four bins')
        caps=[capacity//4]*4
        for i in range(capacity%4): caps[i]+=1
        self.rings=[ReplayBuffer(c,obs_dim,action_dim,device) for c in caps]
        self.capacity=capacity; self.device=device

    @property
    def size(self): return sum(r.size for r in self.rings)

    @property
    def bin_sizes(self): return [r.size for r in self.rings]

    @torch.no_grad()
    def add(self,obs,action,rewards,next_obs,terminated,wind_mps=None):
        if wind_mps is None: raise ValueError('Wind labels required for binned replay')
        bins=torch.bucketize(wind_mps.reshape(-1),
                             torch.tensor([1.,10.,25.],device=self.device))
        for i,ring in enumerate(self.rings):
            mask=bins==i
            if bool(mask.any()):
                ring.add(obs[mask],action[mask],rewards[mask],next_obs[mask],terminated[mask])

    def sample(self,batch_size):
        alive=[r for r in self.rings if r.size>0]
        if not alive: raise ValueError('Replay buffer empty')
        base,remainder=divmod(batch_size,len(alive))
        allocations=[base+(i<remainder) for i in range(len(alive))]
        sets=[ring.sample(n) for ring,n in zip(alive,allocations) if n]
        out=[torch.cat([part[j] for part in sets],0) for j in range(5)]
        index=torch.randperm(batch_size,device=self.device)
        return tuple(x[index] for x in out)


class SACAgent:
    def __init__(self, obs_dim: int, act_dim: int, device: str, cfg: SACConfig):
        self.cfg, self.device = cfg, device
        if cfg.history_len < 1 or obs_dim % cfg.history_len:
            raise ValueError('Observation dimension must be divisible by history_len')
        self.teacher_obs_dim = obs_dim // cfg.history_len
        self.actor = ResidualActor(obs_dim, act_dim, cfg.residual_scale, self.teacher_obs_dim).to(device)
        self.critic = TwinQ(obs_dim, act_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.net.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.log_alpha = torch.tensor(math.log(cfg.alpha_init), device=device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
        self.target_entropy = -cfg.target_entropy_scale * act_dim
        self.num_updates = 0

    @torch.no_grad()
    def act(self, obs, teacher_only=False, deterministic=False):
        return self.actor.teacher(obs[..., :self.teacher_obs_dim]) if teacher_only else self.actor(obs, deterministic=deterministic)

    def update(self, batch):
        obs, action, reward, next_obs, terminated = batch
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            next_act, next_logp, _ = self.actor.sample(next_obs)
            tq1, tq2 = self.critic_target(next_obs, next_act)
            target = reward + self.cfg.gamma * (1 - terminated) * (torch.minimum(tq1, tq2) - alpha * next_logp)
        q1, q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.grad_clip)
        self.critic_opt.step()

        # Critic provides gradients wrt actions, but isn't itself optimized during actor update.
        for p in self.critic.parameters():
            p.requires_grad_(False)
        new_action, logp, residual = self.actor.sample(obs)
        aq1, aq2 = self.critic(obs, new_action)
        residual_cost = residual.square().mean()
        actor_loss = (alpha * logp - torch.minimum(aq1, aq2)).mean() + self.cfg.residual_penalty * residual_cost
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.net.parameters(), self.cfg.grad_clip)
        self.actor_opt.step()
        for p in self.critic.parameters():
            p.requires_grad_(True)

        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()
        if self.cfg.alpha_floor > 0:
            with torch.no_grad():
                self.log_alpha.clamp_(min=math.log(self.cfg.alpha_floor))

        with torch.no_grad():
            for target_p, p in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_p.lerp_(p, self.cfg.tau)
        self.num_updates += 1
        return {'critic_loss': critic_loss.item(), 'actor_loss': actor_loss.item(),
                'alpha': self.log_alpha.exp().item(), 'residual_rms': residual.square().mean().sqrt().item(),
                'actor_entropy':(-logp.detach()).mean().item(),
                'actor_log_std_mean':self.actor.net(obs).detach().chunk(2,dim=-1)[1].clamp(-5.,1.).mean().item()}

    def state_dict(self):
        return {'actor': self.actor.state_dict(), 'critic': self.critic.state_dict(),
                'critic_target': self.critic_target.state_dict(),
                'actor_opt': self.actor_opt.state_dict(), 'critic_opt': self.critic_opt.state_dict(),
                'alpha_opt': self.alpha_opt.state_dict(), 'log_alpha': self.log_alpha.detach(),
                'num_updates': self.num_updates}

    def load_state_dict(self, saved):
        for key in ('actor','critic','critic_target','actor_opt','critic_opt','alpha_opt'):
            getattr(self, key).load_state_dict(saved[key])
        with torch.no_grad():
            self.log_alpha.copy_(saved['log_alpha'])
        self.num_updates = int(saved.get('num_updates', 0))
        self.actor.teacher.requires_grad_(False)
        self.actor.teacher.eval()
