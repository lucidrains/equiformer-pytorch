import torch

from equiformer_pytorch.irr_repr import (
    spherical_harmonics,
    rot_tensor,
    irr_repr,
    irr_repr_tensor,
    rot_x_to_y_direction,
    rot_to_euler_angles,
    irr_repr_tensor
)

from equiformer_pytorch.basis import (
    get_spherical_from_cartesian,
    get_R_tensor,
    precompute_sh,
    basis_transformation_Q_J
)

from equiformer_pytorch.utils import to_order

def test_faster_equivariance():
    # inputs

    coors = torch.randn(3)
    R = rot_tensor(torch.rand(3))

    # function

    def fn(coors):
        coors_sph = get_spherical_from_cartesian(coors)

        Y = precompute_sh(coors_sph, 2)[1]
        Q_J = basis_transformation_Q_J(1, 0, 1).to(coors)

        K = Y @ Q_J.T
        return K

    def fn_faster(coors):
        z_axis = torch.tensor([0., 1., 0.]).to(coors)

        # the rotation to z-axis

        R      = rot_x_to_y_direction(coors, z_axis)
        angles = rot_to_euler_angles(R)
        D      = irr_repr_tensor(1, angles)

        # rest of the calculation

        coors_sph = get_spherical_from_cartesian(z_axis)
        Y = precompute_sh(coors_sph, 2)[1]

        Q_J = basis_transformation_Q_J(1, 0, 1).to(coors) @ D

        # faster because Y is now sparse

        m0 = Y.shape[-1] // 2
        K = Y[m0] * Q_J.T[m0]
        return K

    out = fn(coors)
    rotated_out = fn(coors @ R)

    out_faster = fn_faster(coors)
    rotated_out_faster = fn_faster(coors @ R)

    assert torch.allclose(out @ R, rotated_out, atol = 1e-6)
    assert torch.allclose(out, out_faster, atol = 1e-6)
    assert torch.allclose(rotated_out_faster, rotated_out, atol = 1e-6)
