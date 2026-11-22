"""Most-recent-first per-environment observation history for off-policy SAC.

The frozen teacher reads only the newest unmodified observation; the SAC
residual and Q networks receive the flattened entire history.
"""
import torch


class ObservationHistory:
    def __init__(self, first_obs: torch.Tensor, length: int):
        if length < 1:
            raise ValueError('History length must be positive')
        self.length = length
        self.data = first_obs.detach().unsqueeze(1).repeat(1, length, 1)

    def flattened(self):
        return self.data.reshape(self.data.shape[0], -1).clone()

    @torch.no_grad()
    def advance(self, next_obs: torch.Tensor, done: torch.Tensor,
                terminal_obs: torch.Tensor | None=None):
        """Return transition next state BEFORE resetting history for done envs.

        If the environment auto-resets, terminal_obs is the *pre-reset* observation.
        Done envs next policy observation history is reset to repeated new episode obs.
        """
        if bool(done.any()) and terminal_obs is None:
            raise ValueError('Pre-reset terminal_obs required for done histories')
        outgoing = next_obs.clone()
        if terminal_obs is not None:
            outgoing[done] = terminal_obs[done]
        expanded = torch.cat((outgoing[:,None,:], self.data[:,:-1,:]), dim=1)
        terminal_flat = expanded.reshape(next_obs.shape[0], -1).clone()
        updated = torch.cat((next_obs[:,None,:], self.data[:,:-1,:]), dim=1)
        if bool(done.any()):
            updated[done] = next_obs[done,None,:].expand(-1,self.length,-1)
        self.data = updated
        return terminal_flat, self.flattened()
