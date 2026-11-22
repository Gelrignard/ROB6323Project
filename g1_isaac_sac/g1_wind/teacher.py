"""Strict import of Isaac Lab 3.0 G1-Flat RSL-RL PPO deterministic MLP actor.

RSL-RL v5 uses checkpoint['actor_state_dict'] with 'mlp.0.weight' etc.
The G1 Flat config defines [256,128,128] with ELU, no obs normalization.
We deliberately do not reuse the PPO Gaussian noise parameters.
"""
import re
import torch
from torch import nn


class FrozenG1Teacher(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        dims = [obs_dim, 256, 128, 128, action_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ELU())
        self.mlp = nn.Sequential(*layers)
        self.requires_grad_(False)

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs)

    def load_ppo_checkpoint(self, checkpoint: str) -> None:
        try:
            data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise RuntimeError(
                'Cannot safely read checkpoint. Use a trusted Isaac Lab 3.0 G1-Flat RSL-RL PPO .pt file.'
            ) from exc
        if not isinstance(data, dict):
            raise ValueError('Not an RSL-RL checkpoint dictionary.')
        if 'actor_state_dict' in data:
            source = data['actor_state_dict']
            prefix = 'mlp.'
        elif 'model_state_dict' in data:  # legacy RSL-RL v4 combined state
            source = data['model_state_dict']
            prefix = 'actor.mlp.' if 'actor.mlp.0.weight' in source else 'actor.'
        else:
            raise ValueError('Missing actor_state_dict (RSL-RL 5) or model_state_dict (legacy).')
        if any('obs_normalizer.' in k for k in source):
            raise ValueError('Teacher has observation normalization; expected stock G1-Flat with none.')
        expected = self.mlp.state_dict()
        remapped = {}
        for key in expected:
            orig_key = prefix + key
            if orig_key not in source:
                raise ValueError(f'Pretrained G1 teacher missing {orig_key}; checkpoint may not be G1 Flat.')
            if source[orig_key].shape != expected[key].shape:
                raise ValueError(
                    f'{orig_key} incompatible: checkpoint {tuple(source[orig_key].shape)}, '
                    f'expected {tuple(expected[key].shape)}. Verify env observations/actions and G1 Flat config.'
                )
            remapped[key] = source[orig_key]
        self.mlp.load_state_dict(remapped, strict=True)
        self.requires_grad_(False)
        self.eval()
