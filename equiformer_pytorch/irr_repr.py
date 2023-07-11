from pathlib import Path
from functools import partial

import torch
import torch.nn.functional as F
from torch import sin, cos, atan2, acos

from einops import rearrange, pack, unpack

from equiformer_pytorch.utils import (
    exists,
    default,
    cast_torch_tensor,
    to_order,
    identity,
    l2norm
)

DATA_PATH = Path(__file__).parents[0] / 'data'
path = DATA_PATH / 'J_dense.pt'
Jd = torch.load(str(path))

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def wigner_d_matrix(degree, alpha, beta, gamma, dtype = None, device = None):
    """Create wigner D matrices for batch of ZYZ Euler angles for degree l."""
    batch = alpha.shape[0]
    J = Jd[degree].type(dtype).to(device)
    order = to_order(degree)
    x_a = z_rot_mat(alpha, degree)
    x_b = z_rot_mat(beta, degree)
    x_c = z_rot_mat(gamma, degree)
    res = x_a @ J @ x_b @ J @ x_c
    return res.view(batch, order, order)

def z_rot_mat(angle, l):
    device, dtype = angle.device, angle.dtype

    batch = angle.shape[0]
    arange = partial(torch.arange, device = device)

    order = to_order(l)

    m = angle.new_zeros((batch, order, order))

    batch_range = arange(batch, dtype = torch.long)[..., None]
    inds = arange(order, dtype = torch.long)[None, ...]
    reversed_inds = arange(2 * l, -1, -1, dtype = torch.long)[None, ...]
    frequencies = arange(l, -l - 1, -1, dtype = dtype)[None]

    m[batch_range, inds, reversed_inds] = sin(frequencies * angle[..., None])
    m[batch_range, inds, inds] = cos(frequencies * angle[..., None])
    return m

def irr_repr(order, angles):
    """
    irreducible representation of SO3 - accepts multiple angles in tensor
    """
    dtype, device = angles.dtype, angles.device
    angles, ps = pack_one(angles, '* c')

    alpha, beta, gamma = angles.unbind(dim = -1)
    rep = wigner_d_matrix(order, alpha, beta, gamma, dtype = dtype, device = device)

    return unpack_one(rep, ps, '* o1 o2')

@cast_torch_tensor
def rot_z(gamma):
    '''
    Rotation around Z axis
    '''
    c = cos(gamma)
    s = sin(gamma)
    z = torch.zeros_like(gamma)
    o = torch.ones_like(gamma)

    out = torch.stack((
        c, -s, z,
        s, c, z,
        z, z, o
    ), dim = -1)

    return rearrange(out, '... (r1 r2) -> ... r1 r2', r1 = 3)

@cast_torch_tensor
def rot_y(beta):
    '''
    Rotation around Y axis
    '''
    c = cos(beta)
    s = sin(beta)
    z = torch.zeros_like(beta)
    o = torch.ones_like(beta)

    out = torch.stack((
        c, z, s,
        z, o, z,
        -s, z, c
    ), dim = -1)

    return rearrange(out, '... (r1 r2) -> ... r1 r2', r1 = 3)

def rot(alpha, beta, gamma):
    '''
    ZYZ Euler angles rotation
    '''
    return rot_z(alpha) @ rot_y(beta) @ rot_z(gamma)

def rot_to_euler_angles(R):
    '''
    Rotation matrix to ZYZ Euler angles
    '''
    device, dtype = R.device, R.dtype
    xyz = R @ torch.tensor([0.0, 1.0, 0.0], device = device, dtype = dtype)
    xyz = l2norm(xyz).clamp(-1., 1.)

    b = acos(xyz[..., 1])
    a = atan2(xyz[..., 0], xyz[..., 2])

    R = rot(a, b, torch.zeros_like(a)).transpose(-1, -2) @ R
    c = atan2(R[..., 0, 2], R[..., 0, 0])
    return torch.stack((a, b, c), dim = -1)
