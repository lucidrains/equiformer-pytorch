import pytest
import torch
from equiformer_pytorch.equiformer_pytorch import Equiformer
from equiformer_pytorch.irr_repr import rot
from equiformer_pytorch.utils import torch_default_dtype

DIM = 32

def test_transformer():
    model = Equiformer(
        dim = DIM,
        depth = 1
    )

    feats = torch.randn(1, 32, DIM)
    coors = torch.randn(1, 32, 3)
    mask  = torch.ones(1, 32).bool()

    type0, _ = model(feats, coors, mask)
    assert type0.shape == (1, 32, DIM), 'output must be of the right shape'

@pytest.mark.parametrize('l2_dist_attention', [True, False])
def test_equivariance(
    l2_dist_attention
):

    model = Equiformer(
        dim = DIM,
        depth = 1,
        l2_dist_attention = l2_dist_attention,
        reduce_dim_out = True
    )

    feats = torch.randn(1, 32, DIM)
    coors = torch.randn(1, 32, 3)
    mask  = torch.ones(1, 32).bool()

    R   = rot(*torch.randn(3))
    _, out1 = model(feats, coors @ R, mask)
    out2 = model(feats, coors, mask)[1] @ R

    diff = (out1 - out2).abs().max()
    assert diff < 1e-4, 'is not equivariant'
