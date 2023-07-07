import os
from math import pi
import torch
from torch import einsum
from einops import rearrange, repeat
from itertools import product
from contextlib import contextmanager, nullcontext

from equiformer_pytorch.irr_repr import (
    irr_repr,
    spherical_harmonics,
    rot_x_to_y_direction,
    rot_to_euler_angles,
    irr_repr_tensor
)

from equiformer_pytorch.utils import torch_default_dtype, cache_dir, exists, default, to_order
from equiformer_pytorch.spherical_harmonics import clear_spherical_harmonics_cache

# constants

CACHE_PATH = default(os.getenv('CACHE_PATH'), os.path.expanduser('~/.cache.equivariant_attention'))
CACHE_PATH = CACHE_PATH if not exists(os.environ.get('CLEAR_CACHE')) else None

# todo (figure out why this was hard coded in official repo)

RANDOM_ANGLES = [ 
    [4.41301023, 5.56684102, 4.59384642],
    [4.93325116, 6.12697327, 4.14574096],
    [0.53878964, 4.09050444, 5.36539036],
    [2.16017393, 3.48835314, 5.55174441],
    [2.52385107, 0.2908958, 3.90040975]
]


# functions

def get_matrix_kernel(A, eps = 1e-10):
    '''
    Compute an orthonormal basis of the kernel (x_1, x_2, ...)
    A x_i = 0
    scalar_product(x_i, x_j) = delta_ij

    :param A: matrix
    :return: matrix where each row is a basis vector of the kernel of A
    '''
    _u, s, v = torch.svd(A)
    kernel = v.t()[s < eps]
    return kernel


def get_matrices_kernel(As, eps = 1e-10):
    '''
    Computes the common kernel of all the As matrices
    '''
    matrix = torch.cat(As, dim=0)
    return get_matrix_kernel(matrix, eps)

def get_spherical_from_cartesian(cartesian, divide_radius_by = 1.0):
    """
    # ON ANGLE CONVENTION
    #
    # sh has following convention for angles:
    # :param theta: the colatitude / polar angle, ranging from 0(North Pole, (X, Y, Z) = (0, 0, 1)) to pi(South Pole, (X, Y, Z) = (0, 0, -1)).
    # :param phi: the longitude / azimuthal angle, ranging from 0 to 2 pi.
    #
    # the 3D steerable CNN code therefore (probably) has the following convention for alpha and beta:
    # beta = pi - theta; ranging from 0(South Pole, (X, Y, Z) = (0, 0, -1)) to pi(North Pole, (X, Y, Z) = (0, 0, 1)).
    # alpha = phi
    #
    """
    # initialise return array
    spherical = torch.zeros_like(cartesian)

    # indices for return array
    ind_radius, ind_alpha, ind_beta = 0, 1, 2

    cartesian_y, cartesian_z, cartesian_x  = cartesian.unbind(dim = -1)

    # get projected radius in xy plane
    r_xy = cartesian_x ** 2 + cartesian_y ** 2

    # get second angle
    # version 'elevation angle defined from Z-axis down'
    spherical[..., ind_beta] = torch.atan2(torch.sqrt(r_xy), cartesian_z)

    # get angle in x-y plane
    spherical[...,ind_alpha] = torch.atan2(cartesian_y, cartesian_x)

    # get overall radius
    radius = torch.sqrt(r_xy + cartesian_z ** 2)

    if divide_radius_by != 1.0:
        radius /= divide_radius_by

    spherical[..., ind_radius] = radius
    return spherical

def kron(a, b):
    """
    A part of the pylabyk library: numpytorch.py at https://github.com/yulkang/pylabyk

    Kronecker product of matrices a and b with leading batch dimensions.
    Batch dimensions are broadcast. The number of them mush
    :type a: torch.Tensor
    :type b: torch.Tensor
    :rtype: torch.Tensor
    """
    res = einsum('... i j, ... k l -> ... i k j l', a, b)
    return rearrange(res, '... i j k l -> ... (i j) (k l)')

def get_R_tensor(order_out, order_in, a, b, c):
    return kron(irr_repr(order_out, a, b, c), irr_repr(order_in, a, b, c))

def sylvester_submatrix(order_out, order_in, J, a, b, c):
    ''' generate Kronecker product matrix for solving the Sylvester equation in subspace J '''
    R_tensor = get_R_tensor(order_out, order_in, a, b, c)  # [m_out * m_in, m_out * m_in]
    R_irrep_J = irr_repr(J, a, b, c)  # [m, m]

    R_tensor_identity = torch.eye(R_tensor.shape[0])
    R_irrep_J_identity = torch.eye(R_irrep_J.shape[0])

    return kron(R_tensor, R_irrep_J_identity) - kron(R_tensor_identity, R_irrep_J.t())  # [(m_out * m_in) * m, (m_out * m_in) * m]

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
    sylvester_submatrices = [sylvester_submatrix(order_out, order_in, J, a, b, c) for a, b, c in random_angles]
    null_space = get_matrices_kernel(sylvester_submatrices)
    assert null_space.size(0) == 1, null_space.size()  # unique subspace solution
    Q_J = null_space[0]  # [(m_out * m_in) * m]

    Q_J = rearrange(
        Q_J,
        '(oi m) -> oi m',
        m = to_order(J)
    )

    return Q_J.float()  # [m_out * m_in, m]

def precompute_sh(r_ij, max_J):
    """
    pre-comput spherical harmonics up to order max_J

    :param r_ij: relative positions
    :param max_J: maximum order used in entire network
    :return: dict where each entry has shape [B,N,K,2J+1]
    """
    _, alpha, beta = r_ij.unbind(dim = -1)
    Y_Js = {J: spherical_harmonics(J, alpha, beta) for J in range(max_J + 1)}
    clear_spherical_harmonics_cache()
    return Y_Js

@torch.no_grad()
def get_basis_pkg(r_ij, max_degree):
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

    pkg = dict()
    basis = dict()
    D = dict()

    # precompute D
    # 1. compute rotation to [0, 1, 0]
    # 2. calculate the ZYZ euler angles from that rotation
    # 3. calculate the D irreducible representation from 0 ... max_degree (technically 0 not needed)

    z_axis = torch.tensor([0., 1., 0.], device = device, dtype = dtype)

    R = rot_x_to_y_direction(
        r_ij,
        z_axis
    )

    angles = rot_to_euler_angles(R)

    for d in range(max_degree + 1):
        D[d] = irr_repr_tensor(d, angles)

    # calculate spherical harmonics for [0, 1, 0] only

    z_axis_spherical = torch.tensor([1., 0., 0.], device = device, dtype = dtype)
    Y = precompute_sh(z_axis_spherical, 2 * max_degree)

    # Equivariant basis (dict['<d_in><d_out>']) for [0, 1, 0] only

    for d_in, d_out in product(range(max_degree+1), range(max_degree+1)):
        K_Js = []

        for J in range(abs(d_in - d_out), d_in + d_out + 1):
            # spherical harmonic projection matrices

            Q_J = basis_transformation_Q_J(J, d_in, d_out)
            Q_J = Q_J.to(r_ij)

            # given Y is now sparse (with r_ij rotated to [0, 1, 0]), simply select out the value at m0

            Y_J = Y[J]
            m0_index = Y_J.shape[-1] // 2

            K_J = Y_J[m0_index] * Q_J.T[m0_index]
            K_Js.append(K_J)

        K_Js = rearrange(
            K_Js,
            'm (o i) -> o i m',
            o = to_order(d_out),
            i = to_order(d_in),
            m = to_order(min(d_in, d_out))
        )

        basis[f'{d_in},{d_out}'] = K_Js

    return dict(basis = basis, D = D)
