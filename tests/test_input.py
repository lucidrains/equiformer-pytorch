import pytest

import torch
from equiformer_pytorch.equiformer_pytorch import Equiformer


@pytest.mark.parametrize('dim', [32])
def test_type1_input_feats(dim):
    model = Equiformer(
        dim_in=(dim, dim),
        input_degrees=2,
        dim = dim,
        depth = 2,
        num_degrees = 3,
        init_out_zero = False,
        reduce_dim_out=False
    )

    feats = {
        0: torch.randn(2, 10, dim, 1),
        1: torch.randn(2, 10, dim, 3)
    }
    coors = torch.randn(2, 10, 3)
    mask  = torch.ones(2, 10).bool()

    out = model(feats, coors, mask)
    assert out.type0.shape == (2, 10, 32), 'type-0 output must be of the right shape'
    assert out.type1.shape == (2, 10, 32, 3), 'type-1 output must be of the right shape'
