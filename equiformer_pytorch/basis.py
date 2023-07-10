import os
from math import pi
import torch
from einops import rearrange, repeat, einsum
from itertools import product
from contextlib import contextmanager, nullcontext

from equiformer_pytorch.irr_repr import (
    irr_repr,
    rot_x_to_y_direction,
    rot_to_euler_angles,
    irr_repr_tensor
)

from equiformer_pytorch.utils import torch_default_dtype, cache_dir, exists, default, to_order

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

def identity(t):
    return t

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
    return kron(irr_repr_tensor(order_out, angles), irr_repr_tensor(order_in, angles))

def sylvester_submatrix(order_out, order_in, J, a, b, c):
    ''' generate Kronecker product matrix for solving the Sylvester equation in subspace J '''
    angles = torch.stack((a, b, c), dim = -1)

    R_tensor = get_R_tensor(order_out, order_in, a, b, c)  # [m_out * m_in, m_out * m_in]

    R_irrep_J = irr_repr_tensor(J, angles)  # [m, m]
    R_irrep_J_T = rearrange(R_irrep_J, '... m n -> ... n m')

    R_tensor_identity = torch.eye(R_tensor.shape[-1])
    R_irrep_J_identity = torch.eye(R_irrep_J.shape[-1])

    return kron(R_tensor, R_irrep_J_identity) - kron(R_tensor_identity, R_irrep_J_T)  # [(m_out * m_in) * m, (m_out * m_in) * m]

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

@torch.no_grad()
def get_basis(r_ij, max_degree):
    """Return equivariant weight basis (basis)

    Call this function *once* at the start of each forward pass of the model.
    It computes the equivariant weight basis, W_J^lk(x), and internodal 
    distances, needed to compute varphi_J^lk(x), of eqn 8 of
    https://arxiv.org/pdf/2006.10503.pdf. The return values of this function 
    can be shared as input across all SE(3)-Transformer layers in a model.

    Args:
        r_ij: relative positional vectors
        max_degree: non-negative int for degree of highest feature-type
    Returns:
        dict of equivariant bases, keys are in form '<d_in><d_out>'
    """
    device, dtype = r_ij.device, r_ij.dtype

    # Package will include
    # 1. basis
    # 2. irreducible representation D to rotate all r_ij to [0., 1., 0.]

    basis = dict()
    D = dict()

    # precompute D
    # 1. compute rotation to [0., 1., 0.]
    # 2. calculate the ZYZ euler angles from that rotation
    # 3. calculate the D irreducible representation from 0 ... max_degree (technically 0 not needed)

    z_axis = torch.tensor([0., 1., 0.], device = device, dtype = dtype)

    R = rot_x_to_y_direction(r_ij, z_axis)

    angles = rot_to_euler_angles(R)

    # Equivariant basis (dict['<d_in><d_out>'])

    for d_in, d_out in product(range(max_degree+1), range(max_degree+1)):
        K_Js = []

        if d_in not in D:
            D[d_in] = irr_repr_tensor(d_in, angles)

        for J in range(abs(d_in - d_out), d_in + d_out + 1):

            # Get spherical harmonic projection matrices

            if J not in D:
                D[J] = irr_repr_tensor(J, angles)

            Q_J = basis_transformation_Q_J(J, d_in, d_out).to(r_ij)
            Q_J = einsum(Q_J, D[J], 'oi f, ... f g -> ... oi g')

            # Create kernel from spherical harmonics

            mo_index = Q_J.shape[-1] // 2

            # aligning edges (r_ij) with z-axis leads to sparse Y_J - thus plucking out mo index
            # https://arxiv.org/abs/2206.14331
            # equiformer v2 then normalizes the Y, to remove it altogether

            K_J = Q_J[..., mo_index]
            K_Js.append(K_J)

        K_Js = rearrange(
            K_Js,
            'm ... (o i) -> ... 1 o 1 i m',
            o = to_order(d_out),
            i = to_order(d_in),
            m = to_order(min(d_in, d_out))
        )

        basis[f'{d_in},{d_out}'] = K_Js

    return basis
