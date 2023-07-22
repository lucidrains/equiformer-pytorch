import os
from itertools import product
from collections import namedtuple

import torch
from einops import rearrange, repeat, reduce, einsum

from equiformer_pytorch.irr_repr import (
    irr_repr,
    rot_to_euler_angles
)

from equiformer_pytorch.utils import (
    torch_default_dtype,
    cache_dir,
    exists,
    default,
    to_order,
    identity,
    l2norm,
    slice_for_centering_y_to_x
)

# constants

CACHE_PATH = default(os.getenv('CACHE_PATH'), os.path.expanduser('~/.cache.equivariant_attention'))
CACHE_PATH = CACHE_PATH if not exists(os.environ.get('CLEAR_CACHE')) else None

# todo (figure out why this was hard coded in official repo)

RANDOM_ANGLES = torch.tensor([
    [4.41301023, 5.56684102, 4.59384642],
    [4.93325116, 6.12697327, 4.14574096],
    [0.53878964, 4.09050444, 5.36539036],
    [2.16017393, 3.48835314, 5.55174441],
    [2.52385107, 0.2908958, 3.90040975]
], dtype = torch.float64)

# functions

def get_matrix_kernel(A, eps = 1e-10):
    '''
    Compute an orthonormal basis of the kernel (x_1, x_2, ...)
    A x_i = 0
    scalar_product(x_i, x_j) = delta_ij

    :param A: matrix
    :return: matrix where each row is a basis vector of the kernel of A
    '''
    A = rearrange(A, '... d -> (...) d')
    _u, s, v = torch.svd(A)
    kernel = v.t()[s < eps]
    return kernel

def sylvester_submatrix(order_out, order_in, J, a, b, c):
    ''' generate Kronecker product matrix for solving the Sylvester equation in subspace J '''
    angles = torch.stack((a, b, c), dim = -1)

    R_tensor = get_R_tensor(order_out, order_in, a, b, c)  # [m_out * m_in, m_out * m_in]

    R_irrep_J = irr_repr(J, angles)  # [m, m]
    R_irrep_J_T = rearrange(R_irrep_J, '... m n -> ... n m')

    R_tensor_identity = torch.eye(R_tensor.shape[-1])
    R_irrep_J_identity = torch.eye(R_irrep_J.shape[-1])

    return kron(R_tensor, R_irrep_J_identity) - kron(R_tensor_identity, R_irrep_J_T)  # [(m_out * m_in) * m, (m_out * m_in) * m]

def kron(a, b):
    """
    A part of the pylabyk library: numpytorch.py at https://github.com/yulkang/pylabyk
    Kronecker product of matrices a and b with leading batch dimensions.
    Batch dimensions are broadcast. The number of them mush
    :type a: torch.Tensor
    :type b: torch.Tensor
    :rtype: torch.Tensor
    """
    res = einsum(a, b, '... i j, ... k l -> ... i k j l')
    return rearrange(res, '... i j k l -> ... (i j) (k l)')

def get_R_tensor(order_out, order_in, a, b, c):
    angles = torch.stack((a, b, c), dim = -1)
    return kron(irr_repr(order_out, angles), irr_repr(order_in, angles))

@cache_dir(CACHE_PATH)
@torch_default_dtype(torch.float64)
@torch.no_grad()
def basis_transformation_Q_J(J, order_in, order_out, random_angles = RANDOM_ANGLES):
    """
    :param J: order of the spherical harmonics
    :param order_in: order of the input representation
    :param order_out: order of the output representation
    :return: one part of the Q^-1 matrix of the article
    """
    sylvester_submatrices = sylvester_submatrix(order_out, order_in, J, *random_angles.unbind(dim = -1))
    null_space = get_matrix_kernel(sylvester_submatrices)

    assert null_space.size(0) == 1, null_space.size()  # unique subspace solution
    Q_J = null_space[0] # [(m_out * m_in) * m]

    Q_J = rearrange(
        Q_J,
        '(oi m) -> oi m',
        m = to_order(J)
    )

    return Q_J.float()  # [m_out * m_in, m]

@cache_dir(CACHE_PATH)
@torch_default_dtype(torch.float64)
@torch.no_grad()
def get_basis(max_degree):
    """
    Return equivariant weight basis (basis)
    assuming edges are aligned to z-axis
    """
    basis = dict()

    # Equivariant basis (dict['<d_in><d_out>'])

    for d_in, d_out in product(range(max_degree+1), range(max_degree+1)):
        K_Js = []

        d_min = min(d_in, d_out)

        m_in, m_out, m_min = map(to_order, (d_in, d_out, d_min))
        slice_in, slice_out = map(lambda t: slice_for_centering_y_to_x(t, m_min), (m_in, m_out))

        if d_min == 0:
            continue

        for J in range(abs(d_in - d_out), d_in + d_out + 1):

            # Get spherical harmonic projection matrices

            Q_J = basis_transformation_Q_J(J, d_in, d_out)

            # aligning edges (r_ij) with z-axis leads to sparse spherical harmonics (ex. degree 1 [0., 1., 0.]) - thus plucking out only the mo index
            # https://arxiv.org/abs/2206.14331
            # equiformer v2 then normalizes the Y, to remove it altogether

            mo_index = J
            K_J = Q_J[..., mo_index]

            K_J = rearrange(K_J, '... (o i) -> ... o i', o = m_out)
            K_J = K_J[..., slice_out, slice_in]

            K_J = reduce(K_J, 'o i -> i', 'sum') # the matrix is a sparse diagonal, but flipped depending on whether J is even or odd

            K_Js.append(K_J)

        K_Js = torch.stack(K_Js, dim = -1)

        basis[f'({d_in},{d_out})'] = K_Js # (mi, mf)

    return basis

# functions for rotating r_ij to z-axis

def rot_x_to_y_direction(x, y, eps = 1e-6):
    '''
    Rotates a vector x to the same direction as vector y
    Taken from https://math.stackexchange.com/a/2672702
    This formulation, although not the shortest path, has the benefit of rotation matrix being symmetric; rotating back to x upon two rotations
    '''
    n, dtype, device = x.shape[-1], x.dtype, x.device

    I = torch.eye(n, device = device, dtype = dtype)

    if torch.allclose(x, y, atol = 1e-6):
        return I

    x, y = x.double(), y.double()

    x, y = map(l2norm, (x, y))

    xy = rearrange(x + y, '... n -> ... n 1')
    xy_t = rearrange(xy, '... n 1 -> ... 1 n')

    R = 2 * (xy @ xy_t) / (xy_t @ xy).clamp(min = eps) - I
    return R.type(dtype)

@torch.no_grad()
def get_D_to_from_z_axis(r_ij, max_degree):
    device, dtype = r_ij.device, r_ij.dtype

    D = dict()

    # precompute D
    # 1. compute rotation to [0., 1., 0.]
    # 2. calculate the ZYZ euler angles from that rotation
    # 3. calculate the D irreducible representation from 0 ... max_degree (technically 0 not needed)

    z_axis = r_ij.new_tensor([0., 1., 0.])

    R = rot_x_to_y_direction(r_ij, z_axis)

    angles = rot_to_euler_angles(R)

    for d in range(max_degree + 1):
        if d == 0:
            continue

        D[d] = irr_repr(d, angles)

    return D
