import os
from math import pi
from pathlib import Path
from functools import wraps, partial

import numpy as np

import torch
import torch.nn.functional as F
from torch import sin, cos, atan2, acos

from einops import rearrange, pack, unpack

from equiformer_pytorch.utils import exists, default, cast_torch_tensor, to_order
from equiformer_pytorch.spherical_harmonics import get_spherical_harmonics, clear_spherical_harmonics_cache

DATA_PATH = path = Path(os.path.dirname(__file__)) / 'data'

try:
    path = DATA_PATH / 'J_dense.pt'
    Jd = torch.load(str(path))
except:
    path = DATA_PATH / 'J_dense.npy'
    Jd_np = np.load(str(path), allow_pickle = True)
    Jd = list(map(torch.from_numpy, Jd_np))

def exists(val):
    return val is not None

def identity(t):
    return t

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

def irr_repr(order, alpha, beta, gamma, dtype = None, device = None):
    """
    irreducible representation of SO3
    - compatible with compose and spherical_harmonics
    """
    cast_ = cast_torch_tensor(identity)
    dtype = default(dtype, torch.get_default_dtype())
    alpha, beta, gamma = map(cast_, (alpha, beta, gamma))
    alpha, beta, gamma = map(lambda t: t[None], (alpha, beta, gamma))

    rep = wigner_d_matrix(order, alpha, beta, gamma, dtype = dtype, device = device)
    return rearrange(rep, '1 ... -> ...')

def irr_repr_tensor(order, angles):
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

@cast_torch_tensor
def x_to_alpha_beta(x):
    '''
    Convert point (x, y, z) on the sphere into (alpha, beta)
    '''
    x = F.normalize(x, dim = -1)
    a0, a1, b1 = x.unbind(dim = -1)
    beta = acos(b1)
    alpha = atan2(a1, a0)
    return (alpha, beta)

def rot(alpha, beta, gamma):
    '''
    ZYZ Euler angles rotation
    '''
    return rot_z(alpha) @ rot_y(beta) @ rot_z(gamma)

def rot_tensor(angles):
    angles, ps = pack_one(angles, '* a')
    alpha, beta, gamma = angles.unbind(dim = -1)
    rotations = rot_z(alpha) @ rot_y(beta) @ rot_z(gamma)
    return unpack_one(rotations, ps, '* r1 r2')

def rot_to_euler_angles(R):
    '''
    Rotation matrix to ZYZ Euler angles
    '''
    device, dtype = R.device, R.dtype
    xyz = R @ torch.tensor([0, 0, 1.], device = device, dtype = dtype)
    a, b = x_to_alpha_beta(xyz)
    rotz = rot(0, -b, -a) @ R
    c = atan2(rotz[..., 1, 0], rotz[..., 0, 0])
    return torch.stack((a, b, c), dim = -1)

def rot_x_to_y_direction(x, y):
    '''
    Rotates a vector x to the same direction as vector y
    Taken from https://math.stackexchange.com/a/2672702
    Turns out there are multiple rotation matrices that can bring one vector to another. Re-examine this taken solution, could be why nothing is working
    '''
    n, dtype, device = x.shape[-1], x.dtype, x.device

    I = torch.eye(n, device = device, dtype = dtype)

    if torch.allclose(x, y, atol = 1e-6):
        return I

    x, y = x.double(), y.double()

    x = F.normalize(x, dim = -1)
    y = F.normalize(y, dim = -1)

    xy = rearrange(x + y, '... n -> ... n 1')
    xy_t = rearrange(xy, '... n 1 -> ... 1 n')

    R = 2 * (xy @ xy_t) / (xy_t @ xy) - I
    return R.type(dtype)

def compose(a1, b1, c1, a2, b2, c2):
    """
    (a, b, c) = (a1, b1, c1) composed with (a2, b2, c2)
    """
    comp = rot(a1, b1, c1) @ rot(a2, b2, c2)
    return rot_to_euler_angles(comp).unbind(dim = -1)

def compose_tensor(angles1, angles2):
    angles1, ps1 = pack_one(angles1, '* c')
    angles2, ps2 = pack_one(angles2, '* c')

    if angles1.shape[0] == 1:
        angles1 = angles1.expand_as(angles2)
        ps_out = ps2
    elif angles2.shape[0] == 1:
        angles2 = angles2.expand_as(angles1)
        ps_out = ps1
    elif angles1.shape[0] == angles2.shape[0]:
        ps_out = ps1
    else:
        raise ValueError('angles are not broadcastable')

    comp = rot_tensor(angles1) @ rot_tensor(angles2)
    angles_out = rot_to_euler_angles(comp)
    return unpack_one(angles_out, ps_out, '* c')

def spherical_harmonics(order, alpha, beta, dtype = None):
    return get_spherical_harmonics(order, theta = (pi - beta), phi = alpha)
