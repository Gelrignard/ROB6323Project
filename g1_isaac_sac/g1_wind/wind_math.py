"""Pure-torch aerodynamic approximation. World-frame quantities.

F=0.5*rho*CdA*|wind - local_body_velocity|*(wind - local_body_velocity).
This is NOT MuJoCo's per-mesh ellipsoid model; CdA is a rough calibration target.
"""
import torch


def wind_drag(wind_world: torch.Tensor, body_velocity_world: torch.Tensor,
              rho: float = 1.225, cd_area: float = 0.43) -> torch.Tensor:
    relative = wind_world - body_velocity_world
    return 0.5 * rho * cd_area * torch.linalg.vector_norm(relative, dim=-1, keepdim=True) * relative
