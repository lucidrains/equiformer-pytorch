import torch
from equiformer_pytorch.basis import get_basis, get_R_tensor, basis_transformation_Q_J
from equiformer_pytorch.irr_repr import irr_repr

def test_basis():
    max_degree = 3
    x = torch.randn(2, 1024, 3)
    basis = get_basis(max_degree, device = x.device, dtype = x.dtype)
    assert len(basis.keys()) == max_degree ** 2, 'correct number of basis kernels'

def test_basis_transformation_Q_J():
    rand_angles = torch.rand(4, 3)
    J, order_out, order_in = 1, 1, 1
    Q_J = basis_transformation_Q_J(J, order_in, order_out).float()
    R_tensor = get_R_tensor(order_out, order_in, *rand_angles.unbind(dim = -1))
    assert torch.allclose(R_tensor @ Q_J, Q_J @ irr_repr(J, rand_angles))

def test_basis_transformation_Q_J_sparse():
    rand_angles = torch.rand(4, 3)
    J, order_out, order_in = 1, 1, 1
    Q_J = basis_transformation_Q_J(J, order_in, order_out).float()
    R_tensor = get_R_tensor(order_out, order_in, *rand_angles.unbind(dim = -1))
    assert torch.allclose(R_tensor @ Q_J[..., 0], (Q_J @ irr_repr(J, rand_angles))[..., 0])
