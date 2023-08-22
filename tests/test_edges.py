import pytest

import torch
from equiformer_pytorch.equiformer_pytorch import Equiformer
from equiformer_pytorch.irr_repr import rot
from equiformer_pytorch.utils import torch_default_dtype

# test equivariance with edges

@pytest.mark.parametrize('l2_dist_attention', [True, False])
@pytest.mark.parametrize('reversible', [True, False])
def test_edges_equivariance(
    l2_dist_attention,
    reversible
):
    model = Equiformer(
        num_tokens = 28,
        dim = 64,
        num_edge_tokens = 4,
        edge_dim = 16,
        depth = 2,
        input_degrees = 1,
        num_degrees = 3,
        l2_dist_attention = l2_dist_attention,
        reversible = reversible,
        init_out_zero = False,
        reduce_dim_out = True
    )

    atoms = torch.randint(0, 28, (2, 32))
    bonds = torch.randint(0, 4, (2, 32, 32))
    coors = torch.randn(2, 32, 3)
    mask  = torch.ones(2, 32).bool()

    R   = rot(*torch.randn(3))
    _, out1 = model(atoms, coors @ R, mask, edges = bonds)
    out2 = model(atoms, coors, mask, edges = bonds)[1] @ R

    assert torch.allclose(out1, out2, atol = 1e-4), 'is not equivariant'

# test equivariance with adjacency matrix

@pytest.mark.parametrize('l2_dist_attention', [True, False])
@pytest.mark.parametrize('reversible', [True, False])
def test_adj_mat_equivariance(
    l2_dist_attention,
    reversible
):
    model = Equiformer(
        dim = 32,
        heads = 8,
        depth = 1,
        dim_head = 64,
        num_degrees = 2,
        valid_radius = 10,
        l2_dist_attention = l2_dist_attention,
        reversible = reversible,
        attend_sparse_neighbors = True,
        num_neighbors = 0,
        num_adj_degrees_embed = 2,
        max_sparse_neighbors = 8,
        init_out_zero = False,
        reduce_dim_out = True
    )

    feats = torch.randn(1, 128, 32)
    coors = torch.randn(1, 128, 3)
    mask  = torch.ones(1, 128).bool()

    i = torch.arange(128)
    adj_mat = (i[:, None] <= (i[None, :] + 1)) & (i[:, None] >= (i[None, :] - 1))

    R   = rot(*torch.randn(3))
    _, out1 = model(feats, coors @ R, mask, adj_mat = adj_mat)
    out2 = model(feats, coors, mask, adj_mat = adj_mat)[1] @ R

    assert torch.allclose(out1, out2, atol = 1e-4), 'is not equivariant'
