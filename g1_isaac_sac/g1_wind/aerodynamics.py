"""GPU-vectorized windward panel pressure on *proxy* box surfaces, per G1 link.

Based on the windward pressure law used by ROB6323Project/sim_env/surface_drag.py:
 p_i = (rho Cp / 2) max(0, -u_i dot n_i)^2; F_i = -p_i A_i n_i.

IMPORTANT: This is a SIX-PANEL BOX PROXY per body, NOT that repository's
triangle-mesh quadrature. Each box is centered on the COM with estimated dimensions.
Calibrate dimensions and Cp against mesh geometry before claiming the same model.
The pressure model neglects pressure suction, shielding and flow-structure coupling.

All arrays are torch tensors in SI units. Isaac Lab 3.0 quaternion tensors use scalar-last xyzw order.
"""
import torch


def quat_rotate(q: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by unit quaternions xyzw; uses broadcast dimensions."""
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-9)
    v = q[..., :3]
    t = 2.0 * torch.cross(v, x, dim=-1)
    return x + q[..., 3:] * t + torch.cross(v, t, dim=-1)


def proxy_box_dimensions(body_names, device='cpu', dtype=torch.float32):
    """Plausible link-local dimensions (metres). NOT actual measured G1 geometry."""
    sizes = []
    for name in body_names:
        s = name.lower()
        if 'torso' in s or 'waist' in s:
            dims = (.30, .22, .35)
        elif 'pelvis' in s or 'hip_yaw' in s:
            dims = (.22, .19, .16)
        elif 'head' in s:
            dims = (.19, .17, .21)
        elif 'knee' in s or 'hip_pitch' in s or 'hip_roll' in s:
            dims = (.12, .11, .34)
        elif 'ankle_roll' in s or 'foot' in s:
            dims = (.22, .11, .08)
        elif 'ankle_pitch' in s:
            dims = (.10, .10, .31)
        elif 'shoulder' in s:
            dims = (.10, .11, .24)
        elif 'elbow' in s:
            dims = (.08, .085, .22)
        elif 'palm' in s or 'hand' in s:
            dims = (.085, .055, .13)
        elif 'finger' in s or any(k in s for k in ('_one_link', '_two_link','_three_link','_four_link','_five_link','_six_link')):
            dims = (.035, .035, .07)
        else:
            dims = (.085, .075, .12)
        sizes.append(dims)
    return torch.as_tensor(sizes, device=device, dtype=dtype)


def box_panel_geometry(dimensions: torch.Tensor):
    """Return local face offsets, normals and areas: [B,6,3], [B,6,3], [B,6]."""
    if dimensions.ndim != 2 or dimensions.shape[-1] != 3 or not bool((dimensions > 0).all()):
        raise ValueError('Box dimensions must be positive [B,3]')
    basis = torch.eye(3, dtype=dimensions.dtype, device=dimensions.device)
    normals = torch.cat((basis, -basis), dim=0)[None, :, :].expand(len(dimensions), -1, -1)
    offsets = normals * (dimensions[:, None, :] * 0.5)
    x, y, z = dimensions.unbind(dim=-1)
    areas = torch.stack((y*z, x*z, x*y, y*z, x*z, x*y), dim=-1)
    return offsets, normals, areas


def box_patch_quadrature(dimensions: torch.Tensor, subdivisions: int = 2):
    """Face-midpoint tensor-product quadrature for six faces of each proxy box.

    Returns offsets [B,6*n*n,3], outward normals, and *subpatch* areas.
    A single face-center panel cannot resolve aerodynamic rotational damping;
    a finite quadrature resolves spatial variations of surface velocity.
    For a symmetric stationary box in uniform wind, torque is correctly zero.
    """
    if subdivisions < 1:
        raise ValueError('subdivisions must be positive')
    if dimensions.ndim != 2 or dimensions.shape[-1] != 3 or not bool((dimensions > 0).all()):
        raise ValueError('dimensions must be [B,3] and positive')
    nodes = (torch.arange(subdivisions,device=dimensions.device,dtype=dimensions.dtype)+0.5)/subdivisions-0.5
    uu,vv = torch.meshgrid(nodes,nodes,indexing='ij')
    u,v = uu.reshape(1,-1),vv.reshape(1,-1)
    offs=[]; norms=[]; ar=[]
    for axis in range(3):
        other=[j for j in range(3) if j!=axis]
        for sign in (+1,-1):
            r=torch.zeros((dimensions.shape[0],subdivisions**2,3),device=dimensions.device,dtype=dimensions.dtype)
            r[...,axis]=sign*dimensions[:,axis,None]/2
            r[...,other[0]]=u*dimensions[:,other[0],None]
            r[...,other[1]]=v*dimensions[:,other[1],None]
            n=torch.zeros_like(r); n[...,axis]=sign
            area=(dimensions[:,other[0]]*dimensions[:,other[1]]/(subdivisions**2))[:,None].expand(-1,subdivisions**2)
            offs.append(r); norms.append(n); ar.append(area)
    return torch.cat(offs,dim=1),torch.cat(norms,dim=1),torch.cat(ar,dim=1)


def box_pressure_wrenches(wind_world: torch.Tensor, body_com_pos_w: torch.Tensor,
                          body_com_quat_w: torch.Tensor, body_com_lin_vel_w: torch.Tensor,
                          body_com_ang_vel_w: torch.Tensor, dimensions: torch.Tensor,
                          rho: float=1.225, cp: float=1.0, subdivisions: int = 2):
    """Integrate windward pressure on quadrature points of proxy boxes.

    Uses link-center wind, so spatial variation is from angular surface velocity;
    vertical shear across a single body is still approximated at its COM.
    NOT calibrated mesh-surface integration. Outputs world-frame COM wrenches.
    """
    if rho <= 0 or cp < 0:
        raise ValueError('Density must be positive; Cp nonnegative')
    offsets,normals,areas=box_patch_quadrature(dimensions,subdivisions)
    e,b=body_com_pos_w.shape[:2]
    if wind_world.shape!=(e,b,3) or body_com_quat_w.shape!=(e,b,4):
        raise ValueError('Wind/body quaternion tensor dimensions do not match E,B')
    offsets_w=quat_rotate(body_com_quat_w[:,:,None,:],offsets[None].expand(e,-1,-1,-1))
    normals_w=quat_rotate(body_com_quat_w[:,:,None,:],normals[None].expand(e,-1,-1,-1))
    surface_velocity=(body_com_lin_vel_w[:,:,None,:]+
                      torch.cross(body_com_ang_vel_w[:,:,None,:].expand_as(offsets_w),offsets_w,dim=-1))
    relative=wind_world[:,:,None,:]-surface_velocity
    incoming=(-torch.sum(relative*normals_w,dim=-1)).clamp_min(0)
    pressure=0.5*rho*cp*incoming.square()
    patch_forces=-pressure[...,None]*areas[None,:,:,None]*normals_w
    return patch_forces.sum(dim=2), torch.cross(offsets_w,patch_forces,dim=-1).sum(dim=2)


def effective_cda(force_world: torch.Tensor, wind_world: torch.Tensor, rho: float=1.225):
    """Effective CdA = 2 F_parallel /(rho |wind|²), stationary-robot only."""
    speed=torch.linalg.vector_norm(wind_world,dim=-1).clamp_min(1e-9)
    along=(force_world*wind_world).sum(-1)/speed
    return 2*along/(rho*speed.square())


def tornado_field_world(points_local_w: torch.Tensor, max_tangential_speed: float,
                        core_radius: float=4.0, inflow_fraction: float=0.1) -> torch.Tensor:
    """Repo-style prescribed horizontal vortex, centered at each env origin.

    Note: actual tangential component peaks at radius=core_radius; NOT atmospheric CFD.
    """
    if core_radius <= 0 or max_tangential_speed < 0:
        raise ValueError('Invalid tornado field magnitude/radius')
    r = points_local_w[..., :2]
    r2 = r.square().sum(-1, keepdim=True)
    tangent = torch.stack((-r[...,1], r[...,0]), dim=-1)
    scale = 2 * core_radius / (core_radius*core_radius + r2)
    xy = scale * (max_tangential_speed * tangent - inflow_fraction * max_tangential_speed * r)
    return torch.cat((xy, torch.zeros_like(points_local_w[...,2:3])), -1)


def power_law_wind_profile(wind_at_ref: torch.Tensor, height_m: torch.Tensor,
                           ref_height_m: float = 10.0, exponent: float = 0.14,
                           min_height_m: float = 0.25) -> torch.Tensor:
    """Engineering wind-shear approximation U(z)=U(ref)*(z/ref)^alpha.

    wind_at_ref [...,3], height_m [...]; both broadcast. Horizontal wind only;
    NOT a calibrated hurricane boundary-layer or tornado near-ground model.
    """
    if ref_height_m <= 0 or min_height_m <= 0 or exponent < 0:
        raise ValueError('Reference height, minimum height >0 and alpha>=0 required')
    factor = (height_m.clamp(min=min_height_m) / ref_height_m).pow(exponent)
    return wind_at_ref * factor[..., None]
