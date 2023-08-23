import pytest

import torch
from equiformer_pytorch.equiformer_pytorch import Equiformer
from equiformer_pytorch.irr_repr import rot
from equiformer_pytorch.utils import torch_default_dtype

# test output shape

@pytest.mark.parametrize('dim', [32])
def test_transformer(dim):
    model = Equiformer(
        dim = dim,
        depth = 2,
        num_degrees = 3,
        init_out_zero = False
    )

    feats = torch.randn(1, 32, dim)
    coors = torch.randn(1, 32, 3)
    mask  = torch.ones(1, 32).bool()

    type0, _ = model(feats, coors, mask)
    assert type0.shape == (1, 32, dim), 'output must be of the right shape'

# test equivariance

@pytest.mark.parametrize('dim', [32, (4, 8, 16)])
@pytest.mark.parametrize('dim_in', [32, (32, 32)])
@pytest.mark.parametrize('l2_dist_attention', [True, False])
@pytest.mark.parametrize('reversible', [True, False])
def test_equivariance(
    dim,
    dim_in,
    l2_dist_attention,
    reversible
):

    model = Equiformer(
        dim = dim,
        dim_in=dim_in,
        input_degrees = len(dim_in) if isinstance(dim_in, tuple) else 1,
        depth = 2,
        l2_dist_attention = l2_dist_attention,
        reversible = reversible,
        num_degrees = 3,
        reduce_dim_out = True,
        init_out_zero = False
    )

    if isinstance(dim_in, tuple):
        feats = {deg: torch.randn(1, 32, dim, 2*deg + 1) for deg, dim in enumerate(dim_in)}
    else:
        feats = torch.randn(1, 32, dim_in)

    coors = torch.randn(1, 32, 3)
    mask  = torch.ones(1, 32).bool()

    R   = rot(*torch.randn(3))
    _, out1 = model({0: feats[0], 1: feats[1] @ R} if isinstance(feats, dict) else feats, coors @ R, mask)
    out2 = model(feats, coors, mask)[1] @ R

    assert torch.allclose(out1, out2, atol = 1e-4), 'is not equivariant'
