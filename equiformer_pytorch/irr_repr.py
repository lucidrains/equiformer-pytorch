import os
from math import pi
from pathlib import Path
from functools import wraps

import numpy as np

import torch
import torch.nn.functional as F
from torch import sin, cos, atan2, acos

from einops import rearrange

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

def wigner_d_matrix(degree, alpha, beta, gamma, dtype = None, device = None):
    """Create wigner D matrices for batch of ZYZ Euler angles for degree l."""
    J = Jd[degree].type(dtype).to(device)
    order = to_order(degree)
    x_a = z_rot_mat(alpha, degree)
    x_b = z_rot_mat(beta, degree)
    x_c = z_rot_mat(gamma, degree)
    res = x_a @ J @ x_b @ J @ x_c
    return res.view(order, order)

def z_rot_mat(angle, l):
    device, dtype = angle.device, angle.dtype
    order = to_order(l)
    m = angle.new_zeros((order, order))
    inds = torch.arange(0, order, 1, dtype=torch.long, device=device)
    reversed_inds = torch.arange(2 * l, -1, -1, dtype=torch.long, device=device)
    frequencies = torch.arange(l, -l - 1, -1, dtype=dtype, device=device)[None]

    m[inds, reversed_inds] = sin(frequencies * angle[None])
    m[inds, inds] = cos(frequencies * angle[None])
    return m

def irr_repr(order, alpha, beta, gamma, dtype = None):
    """
    irreducible representation of SO3
    - compatible with compose and spherical_harmonics
    """
    cast_ = cast_torch_tensor(lambda t: t)
    dtype = default(dtype, torch.get_default_dtype())
    alpha, beta, gamma = map(cast_, (alpha, beta, gamma))
    return wigner_d_matrix(order, alpha, beta, gamma, dtype = dtype)

@cast_torch_tensor
def rot_z(gamma):
    '''
    Rotation around Z axis
    '''
    return torch.tensor([
        [cos(gamma), -sin(gamma), 0],
        [sin(gamma), cos(gamma), 0],
        [0, 0, 1]
    ], dtype=gamma.dtype, device=gamma.device)

@cast_torch_tensor
def rot_y(beta):
    '''
    Rotation around Y axis
    '''
    return torch.tensor([
        [cos(beta), 0, sin(beta)],
        [0, 1, 0],
        [-sin(beta), 0, cos(beta)]
    ], dtype=beta.dtype, device=beta.device)

@cast_torch_tensor
def x_to_alpha_beta(x):
    '''
    Convert point (x, y, z) on the sphere into (alpha, beta)
    '''
    x = x / torch.norm(x)
    a0, a1, b1 = x.unbind(dim=0)
    beta = acos(b1)
    alpha = atan2(a1, a0)
    return (alpha, beta)

def rot(alpha, beta, gamma):
    '''
    ZYZ Euler angles rotation
    '''
    return rot_z(alpha) @ rot_y(beta) @ rot_z(gamma)

def rot_to_euler_angles(R):
    '''
    Rotation matrix to ZYZ Euler angles
    '''
    alpha = atan2(R[..., 1, 2], R[..., 0, 2])
    sp, cp = sin(alpha), cos(alpha)
    beta = atan2(cp * R[..., 0, 2] + sp * R[..., 1, 2], R[..., 2, 2])
    gamma = atan2(-sp * R[..., 0, 0] + cp * R[..., 1, 0], -sp * R[..., 0, 1] * cp * R[..., 1, 1])
    return torch.stack((alpha, beta, gamma), dim = -1)

def rot_x_to_y_direction(x, y):
    '''
    Rotates a vector x to the same direction as vector y
    Taken from https://math.stackexchange.com/a/2672702
    '''
    n, dtype, device = x.shape[-1], x.dtype, x.device

    identity = torch.eye(n, device = device, dtype = dtype)

    if torch.allclose(x, y, atol = 1e-6):
        return identity

    x, y = x.double(), y.double()

    x = F.normalize(x, dim = -1)
    y = F.normalize(y, dim = -1)

    xy = rearrange(x + y, '... n -> ... n 1')
    xy_t = rearrange(xy, '... n 1 -> ... 1 n')

    R =  2 * (xy @ xy_t) / (xy_t @ xy) - identity
    return R.type(dtype)

def compose(a1, b1, c1, a2, b2, c2):
    """
    (a, b, c) = (a1, b1, c1) composed with (a2, b2, c2)
    """
    comp = rot(a1, b1, c1) @ rot(a2, b2, c2)
    xyz = comp @ torch.tensor([0, 0, 1.], device=comp.device, dtype=comp.dtype)
    a, b = x_to_alpha_beta(xyz)
    rotz = rot(0, -b, -a) @ comp
    c = atan2(rotz[1, 0], rotz[0, 0])
    return a, b, c

def spherical_harmonics(order, alpha, beta, dtype = None):
    return get_spherical_harmonics(order, theta = (pi - beta), phi = alpha)
