from math import sqrt
from functools import partial
from itertools import product
from collections import namedtuple

from typing import Optional, Union, Tuple
from beartype import beartype

import torch
import torch.nn.functional as F
from torch import nn, einsum

from equiformer_pytorch.basis import get_basis
from equiformer_pytorch.utils import exists, default, batched_index_select, masked_mean, to_order, cast_tuple, safe_cat, fast_split

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

# constants

Return = namedtuple('Return', ['type0', 'type1'])

EdgeInfo = namedtuple('EdgeInfo', ['neighbor_indices', 'neighbor_mask', 'edges'])

# fiber functions

@beartype
def fiber_product(
    fiber_in: Tuple[int, ...],
    fiber_out: Tuple[int, ...]
):
    fiber_in, fiber_out = tuple(map(lambda t: [(degree, dim) for degree, dim in enumerate(t)], (fiber_in, fiber_out)))
    return product(fiber_in, fiber_out)

@beartype
def fiber_and(
    fiber_in: Tuple[int, ...],
    fiber_out: Tuple[int, ...]
):
    fiber_in = [(degree, dim) for degree, dim in enumerate(fiber_in)]
    fiber_out_degrees = set(range(len(fiber_out)))

    out = []
    for degree, dim in fiber_in:
        if degree not in fiber_out_degrees:
            continue

        dim_out = fiber_out[degree]
        out.append((degree, dim, dim_out))

    return out

# helper functions

def split_num_into_groups(num, groups):
    num_per_group = (num + groups - 1) // groups
    remainder = num % groups

    if remainder == 0:
        return (num_per_group,) * groups

    return (*((num_per_group,) * remainder), *((((num_per_group - 1),) * (groups - remainder))))

def get_tensor_device_and_dtype(features):
    _, first_tensor = next(iter(features.items()))
    return first_tensor.device, first_tensor.dtype

def residual_fn(x, residual):
    out = {}
    for degree, tensor in x.items():
        out[degree] = tensor

        if degree not in residual:
            continue

        out[degree] = out[degree] + residual[degree]
    return out

def tuple_set_at_index(tup, index, value):
    l = list(tup)
    l[index] = value
    return tuple(l)

def feature_shapes(feature):
    return tuple(v.shape for v in feature.values())

def feature_fiber(feature):
    return tuple(v.shape[-2] for v in feature.values())

# classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

@beartype
class Linear(nn.Module):
    def __init__(
        self,
        fiber_in: Tuple[int, ...],
        fiber_out: Tuple[int, ...]
    ):
        super().__init__()
        self.weights = nn.ParameterList([])
        self.degrees = []

        for (degree, dim_in, dim_out) in fiber_and(fiber_in, fiber_out):
            self.weights.append(nn.Parameter(torch.randn(dim_in, dim_out) / sqrt(dim_in)))
            self.degrees.append(degree)

    def forward(self, x):
        out = {}

        for degree, weight in zip(self.degrees, self.weights):
            out[degree] = einsum('... d m, d e -> ... e m', x[degree], weight)

        return out

@beartype
class Norm(nn.Module):
    def __init__(
        self,
        fiber: Tuple[int, ...],
        eps = 1e-12,
    ):
        """
        deviates from the paper slightly, will use rmsnorm throughout (no mean centering or bias, even for type0 fatures)
        this has been proven at scale for a number of models, including T5 and alphacode
        """

        super().__init__()
        self.eps = eps
        self.transforms = nn.ParameterList([])

        for degree, dim in enumerate(fiber):
            self.transforms.append(nn.Parameter(torch.ones(dim, 1)))

    def forward(self, features):
        output = {}

        for scale, (degree, t) in zip(self.transforms, features.items()):
            dim = t.shape[-2]

            l2normed = t.norm(dim = -1, keepdim = True)
            rms = l2normed.norm(dim = -2, keepdim = True) * (dim ** -0.5)

            output[degree] = t / rms.clamp(min = self.eps) * scale

        return output

@beartype
class Gate(nn.Module):
    def __init__(
        self,
        fiber: Tuple[int, ...]
    ):
        super().__init__()

        type0_dim = fiber[0]
        dim_gate = sum(fiber[1:])

        assert type0_dim > dim_gate, 'sum of channels from rest of the degrees must be less than the channels in type 0, as they would be used up for gating and subtracted out'

        self.fiber = fiber
        self.num_degrees = len(fiber)
        self.type0_dim_split = [*fiber[1:], type0_dim - dim_gate]

    def forward(self, x):
        output = {}

        type0_tensor = x[0]
        *gates, type0_tensor = type0_tensor.split(self.type0_dim_split, dim = -2)

        # silu for type 0

        output = {0: F.silu(type0_tensor)}

        # sigmoid gate the higher types

        for degree, gate in zip(range(1, self.num_degrees), gates):
            output[degree] = x[degree] * gate.sigmoid()

        return output

@beartype
class DTP(nn.Module):
    """ 'Tensor Product' - in the equivariant sense """

    def __init__(
        self,
        fiber_in: Tuple[int, ...],
        fiber_out: Tuple[int, ...],
        self_interaction = True,
        project_xi_xj = True,   # whether to project xi and xj and then sum, as in paper
        project_out = True,     # whether to do a project out after the "tensor product"
        pool = True,
        edge_dim = 0,
        radial_hidden_dim = 16,
        splits = 4
    ):
        super().__init__()
        self.fiber_in = fiber_in
        self.fiber_out = fiber_out
        self.edge_dim = edge_dim
        self.self_interaction = self_interaction
        self.pool = pool
        self.splits = splits # for splitting the computation of kernel and basis, to reduce peak memory usage

        self.project_xi_xj = project_xi_xj
        if project_xi_xj:
            self.to_xi = Linear(fiber_in, fiber_in)
            self.to_xj = Linear(fiber_in, fiber_in)

        self.kernel_unary = nn.ModuleDict()

        # in the depthwise tensor product, each channel of the output only gets contribution from one degree of the input (please email me if i misconstrued this)

        for degree_out, dim_out in enumerate(self.fiber_out):
            num_degrees_in = len(self.fiber_in)
            split_dim_out = split_num_into_groups(dim_out, num_degrees_in)  # returns a tuple of ints representing how many channels come from each input degree

            for degree_in, (dim_in, dim_out_from_degree_in) in enumerate(zip(self.fiber_in, split_dim_out)):
                self.kernel_unary[f'({degree_in},{degree_out})'] = PairwiseTP(degree_in, dim_in, degree_out, dim_out_from_degree_in, radial_hidden_dim = radial_hidden_dim, edge_dim = edge_dim)

        # whether a single token is self-interacting

        if self_interaction:
            self.self_interact = Linear(fiber_in, fiber_out)

        self.project_out = project_out
        if project_out:
            self.to_out = Linear(fiber_out, fiber_out)

    def forward(
        self,
        inp,
        edge_info: EdgeInfo,
        rel_dist = None,
        basis = None
    ):
        splits = self.splits
        neighbor_indices, neighbor_masks, edges = edge_info

        kernels = {}
        outputs = {}

        # split basis

        basis_keys = basis.keys()
        split_basis_values = list(zip(*list(map(lambda t: fast_split(t, splits, dim = 1), basis.values()))))
        split_basis = list(map(lambda v: dict(zip(basis_keys, v)), split_basis_values))

        # neighbors

        if self.project_xi_xj:
            source, target = self.to_xi(inp), self.to_xj(inp)
        else:
            source, target = inp, inp

        # go through every permutation of input degree type to output degree type

        for degree_out, _ in enumerate(self.fiber_out):
            output = None

            for degree_in, m_in in enumerate(self.fiber_in):
                etype = f'({degree_in},{degree_out})'

                xi, xj = source[degree_in], target[degree_in]

                x = batched_index_select(xj, neighbor_indices, dim = 1)

                if self.project_xi_xj:
                    xi = rearrange(xi, 'b i d m -> b i 1 d m')
                    x = x + xi

                x = x.view(*x.shape[:3], to_order(degree_in) * m_in, 1)

                kernel_fn = self.kernel_unary[etype]
                edge_features = torch.cat((rel_dist, edges), dim = -1) if exists(edges) else rel_dist

                output_chunk = None
                split_x = fast_split(x, splits, dim = 1)
                split_edge_features = fast_split(edge_features, splits, dim = 1)

                # process input, edges, and basis in chunks along the sequence dimension

                for x_chunk, edge_features, basis in zip(split_x, split_edge_features, split_basis):
                    kernel = kernel_fn(edge_features, basis = basis)
                    chunk = einsum('... o i, ... i c -> ... o c', kernel, x_chunk)
                    output_chunk = safe_cat(output_chunk, chunk, dim = 1)

                output = safe_cat(output, output_chunk, dim = -2)

            if self.pool:
                output = masked_mean(output, neighbor_masks, dim = 2)

            leading_shape = x.shape[:2] if self.pool else x.shape[:3]
            output = output.view(*leading_shape, -1, to_order(degree_out))

            outputs[degree_out] = output

        if not self.self_interaction and not self.project_out:
            return outputs

        if self.project_out:
            outputs = self.to_out(outputs)

        self_interact_out = self.self_interact(inp)

        if self.pool:
            return residual_fn(outputs, self_interact_out)

        self_interact_out = {k: rearrange(v, '... d m -> ... 1 d m') for k, v in self_interact_out.items()}
        outputs = {degree: torch.cat(tensors, dim = -3) for degree, tensors in enumerate(zip(self_interact_out.values(), outputs.values()))}
        return outputs

class PairwiseTP(nn.Module):
    def __init__(
        self,
        degree_in,
        nc_in,
        degree_out,
        nc_out,
        edge_dim = 0,
        radial_hidden_dim = 64
    ):
        super().__init__()
        self.degree_in = degree_in
        self.degree_out = degree_out
        self.nc_in = nc_in
        self.nc_out = nc_out

        self.num_freq = to_order(min(degree_in, degree_out))
        self.d_out = to_order(degree_out)
        self.edge_dim = edge_dim

        mid_dim = radial_hidden_dim
        edge_dim = default(edge_dim, 0)

        self.rp = nn.Sequential(
            nn.Linear(edge_dim + 1, mid_dim),
            nn.SiLU(),
            LayerNorm(mid_dim),
            nn.Linear(mid_dim, mid_dim),
            nn.SiLU(),
            LayerNorm(mid_dim),
            nn.Linear(mid_dim, self.num_freq * nc_in * nc_out),
            Rearrange('... (o i f) -> ... o 1 i 1 f', i = nc_in, o = nc_out)
        )

    def forward(self, feat, basis):
        R = self.rp(feat)

        B = basis[f'{self.degree_in},{self.degree_out}']

        out_shape = (*R.shape[:3], self.d_out * self.nc_out, -1)

        # torch.sum(R * B, dim = -1) is too memory intensive
        # needs to be chunked to reduce peak memory usage

        out = 0
        for i in range(R.shape[-1]):
            out += R[..., i] * B[..., i]

        out = rearrange(out, 'b n h s ... -> (b n h s) ...')

        # reshape and out
        return out.view(*out_shape)

# feed forwards

@beartype
class FeedForward(nn.Module):
    def __init__(
        self,
        fiber: Tuple[int, ...],
        fiber_out: Optional[Tuple[int, ...]] = None,
        mult = 4,
        include_htype_norms = True
    ):
        super().__init__()
        self.fiber = fiber

        fiber_hidden = tuple(dim * mult for dim in fiber)

        project_in_fiber = fiber
        project_in_fiber_hidden = tuple_set_at_index(fiber_hidden, 0, sum(fiber_hidden))

        self.include_htype_norms = include_htype_norms
        if include_htype_norms:
            project_in_fiber = tuple_set_at_index(project_in_fiber, 0, sum(fiber))

        fiber_out = default(fiber_out, fiber)

        self.prenorm     = Norm(fiber)
        self.project_in  = Linear(project_in_fiber, project_in_fiber_hidden)
        self.gate        = Gate(project_in_fiber_hidden)
        self.project_out = Linear(fiber_hidden, fiber_out)

    def forward(self, features):
        outputs = self.prenorm(features)

        if self.include_htype_norms:
            type0, *htypes = [*outputs.values()]
            htypes = map(lambda t: t.norm(dim = -1, keepdim = True), htypes)
            type0 = torch.cat((type0, *htypes), dim = -2)
            outputs[0] = type0

        outputs = self.project_in(outputs)
        outputs = self.gate(outputs)
        outputs = self.project_out(outputs)
        return outputs

# attention

@beartype
class DotProductAttention(nn.Module):
    def __init__(
        self,
        fiber: Tuple[int, ...],
        dim_head: Union[int, Tuple[int, ...]] = 64,
        heads: Union[int, Tuple[int, ...]] = 8,
        attend_self = False,
        edge_dim = None,
        single_headed_kv = False,
        radial_hidden_dim = 64,
        use_cdist_sim = True,
        splits = 4
    ):
        super().__init__()
        num_degrees = len(fiber)

        dim_head = cast_tuple(dim_head, num_degrees)
        assert len(dim_head) == num_degrees

        heads = cast_tuple(heads, num_degrees)
        assert len(heads) == num_degrees

        hidden_fiber = tuple(dim * head for dim, head in zip(dim_head, heads))

        self.single_headed_kv = single_headed_kv
        self.attend_self = attend_self
        self.use_cdist_sim = use_cdist_sim

        kv_hidden_fiber = hidden_fiber if not single_headed_kv else dim_head
        kv_hidden_fiber = tuple(dim * 2 for dim in kv_hidden_fiber)

        self.scale = tuple(dim ** -0.5 for dim in dim_head)
        self.heads = heads

        self.prenorm = Norm(fiber)

        self.to_q = Linear(fiber, hidden_fiber)
        self.to_kv = DTP(fiber, kv_hidden_fiber, radial_hidden_dim = radial_hidden_dim, edge_dim = edge_dim, pool = False, self_interaction = attend_self, splits = splits)

        self.to_out = Linear(hidden_fiber, fiber)

    def forward(
        self,
        features,
        edge_info: EdgeInfo,
        rel_dist,
        basis,
        mask = None
    ):
        one_head_kv = self.single_headed_kv

        device, dtype = get_tensor_device_and_dtype(features)
        neighbor_indices, neighbor_mask, edges = edge_info

        if exists(neighbor_mask):
            neighbor_mask = rearrange(neighbor_mask, 'b i j -> b 1 i j')

        features = self.prenorm(features)

        queries = self.to_q(features)

        keyvalues   = self.to_kv(
            features,
            edge_info = edge_info,
            rel_dist = rel_dist,
            basis = basis
        )

        kv_einsum_eq = 'b h i j d m' if not one_head_kv else 'b i j d m'

        outputs = {}

        for degree, h, scale in zip(features.keys(), self.heads, self.scale):
            q, kv = map(lambda t: t[degree], (queries, keyvalues))

            q = rearrange(q, 'b i (h d) m -> b h i d m', h = h)

            if not one_head_kv:
                kv = rearrange(kv, f'b i j (h d) m -> b h i j d m', h = h)

            k, v = kv.chunk(2, dim = -2)

            if degree == 0 or not self.use_cdist_sim:
                sim = einsum(f'b h i d m, {kv_einsum_eq} -> b h i j', q, k) * scale
            else:
                if one_head_kv:
                    k = repeat(k, 'b i j d m -> b h i j d m', h = h)

                q = rearrange(q, 'b h i d m -> b h i 1 d m')
                sim = -((q - k) ** 2).sum(dim = -1).clamp(min = 1e-5).sqrt().sum(dim = -1) * scale

            if exists(neighbor_mask):
                left_pad_needed = int(self.attend_self)
                mask = F.pad(neighbor_mask, (left_pad_needed, 0), value = True)
                sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

            attn = sim.softmax(dim = -1)
            out = einsum(f'b h i j, {kv_einsum_eq} -> b h i d m', attn, v)
            outputs[degree] = rearrange(out, 'b h n d m -> b n (h d) m')

        return self.to_out(outputs)

@beartype
class MLPAttention(nn.Module):
    def __init__(
        self,
        fiber: Tuple[int, ...],
        dim_head: Union[int, Tuple[int, ...]] = 64,
        heads: Union[int, Tuple[int, ...]] = 8,
        attend_self = False,
        edge_dim = None,
        splits = 4,
        single_headed_kv = False,
        attn_leakyrelu_slope = 0.1,
        attn_hidden_dim_mult = 4,
        radial_hidden_dim = 16,
        **kwargs
    ):
        super().__init__()
        num_degrees = len(fiber)

        dim_head = cast_tuple(dim_head, num_degrees)
        assert len(dim_head) == num_degrees

        heads = cast_tuple(heads, num_degrees)
        assert len(heads) == num_degrees

        hidden_fiber = tuple(dim * head for dim, head in zip(dim_head, heads))

        self.single_headed_kv = single_headed_kv
        value_hidden_fiber = hidden_fiber if not single_headed_kv else dim_head

        self.scale = tuple(dim ** -0.5 for dim in dim_head)
        self.heads = heads

        self.prenorm = Norm(fiber)

        # type 0 needs greater dimension, for
        # (1) gating the htypes on the values branch
        # (2) attention logits, with dimension equal to heads amount for starters

        type0_dim = value_hidden_fiber[0]
        htype_dims = sum(value_hidden_fiber[1:])

        value_gate_fiber = tuple_set_at_index(value_hidden_fiber, 0, type0_dim + htype_dims)

        attn_hidden_dims = tuple(head * attn_hidden_dim_mult for head in heads)

        intermediate_fiber = tuple_set_at_index(value_hidden_fiber, 0, sum(attn_hidden_dims) + type0_dim + htype_dims)
        self.intermediate_type0_split = [*attn_hidden_dims, type0_dim + htype_dims]

        # main branch tensor product

        self.to_attn_and_v = DTP(fiber, intermediate_fiber, radial_hidden_dim = radial_hidden_dim, edge_dim = edge_dim, pool = False, self_interaction = attend_self, splits = splits)

        # non-linear projection of attention branch into the attention logits

        self.to_attn_logits = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(attn_leakyrelu_slope),
                nn.Linear(attn_hidden_dim, h, bias = False)
            ) for attn_hidden_dim, h in zip(attn_hidden_dims, self.heads)
        ])

        # non-linear transform of the value branch
        # todo - needs a DTP here?

        self.to_values = nn.Sequential(
            Gate(value_gate_fiber),
            Linear(value_hidden_fiber, value_hidden_fiber)
        )

        # combining heads and projection out

        self.to_out = Linear(hidden_fiber, fiber)

    def forward(
        self,
        features,
        edge_info: EdgeInfo,
        rel_dist,
        basis,
        mask = None
    ):
        one_headed_kv = self.single_headed_kv

        features = self.prenorm(features)

        intermediate = self.to_attn_and_v(
            features,
            edge_info = edge_info,
            rel_dist = rel_dist,
            basis = basis
        )

        *attn_branch_type0, value_branch_type0 = intermediate[0].split(self.intermediate_type0_split, dim = -2)

        intermediate[0] = value_branch_type0

        # process the attention branch

        attentions = []

        for fn, attn_intermediate, scale in zip(self.to_attn_logits, attn_branch_type0, self.scale):
            attn_intermediate = rearrange(attn_intermediate, '... 1 -> ...')
            attn_logits = fn(attn_intermediate)
            attn_logits = attn_logits * scale
            attn = attn_logits.softmax(dim = -2) # (batch, source, target, heads)
            attentions.append(attn)

        # process values branch

        values = self.to_values(intermediate)

        # aggregate values with attention matrix

        outputs = {}

        value_einsum_eq = 'b i j h d m' if not one_headed_kv else 'b i j d m'

        for degree, (attn, value, h) in enumerate(zip(attentions, values.values(), self.heads)):
            if not one_headed_kv:
                value = rearrange(value, 'b i j (h d) m -> b i j h d m', h = h)

            out = einsum(f'b i j h, {value_einsum_eq} -> b i h d m', attn, value)
            out = rearrange(out, 'b i h d m -> b i (h d) m')
            outputs[degree] = out

        # combine heads out

        return self.to_out(outputs)

# main class

@beartype
class Equiformer(nn.Module):
    def __init__(
        self,
        *,
        dim: Union[int, Tuple[int, ...]],
        dim_in: Optional[Union[int, Tuple[int, ...]]] = None,
        num_degrees = 2,
        input_degrees = 1,
        heads: Union[int, Tuple[int, ...]] = 8,
        dim_head: Union[int, Tuple[int, ...]] = 24,
        depth = 2,
        valid_radius = 1e5,
        num_neighbors = float('inf'),
        reduce_dim_out = False,
        radial_hidden_dim = 64,
        num_tokens = None,
        num_positions = None,
        num_edge_tokens = None,
        edge_dim = None,
        attend_self = True,
        differentiable_coors = False,
        splits = 4,
        linear_out = True,
        embedding_grad_frac = 0.5,
        single_headed_kv = False,          # whether to do single headed key/values for dot product attention, to save on memory and compute
        ff_include_htype_norms = False,    # whether for type0 projection to also involve norms of all higher types, in feedforward first projection. this allows for all higher types to be gated by other type norms
        dot_product_attention = True,
        dot_product_attention_use_cdist_sim = True,
        **kwargs
    ):
        super().__init__()

        self.embedding_grad_frac = embedding_grad_frac # trick for more stable training

        # decide hidden dimensions for all types

        self.dim = cast_tuple(dim, num_degrees)
        assert len(self.dim) == num_degrees

        self.num_degrees = len(self.dim)

        # decide input dimensions for all types

        dim_in = default(dim_in, (self.dim[0],))
        self.dim_in = cast_tuple(dim_in, input_degrees)
        assert len(self.dim_in) == input_degrees

        self.input_degrees = len(self.dim_in)

        # token embedding

        type0_feat_dim = self.dim_in[0]
        self.type0_feat_dim = type0_feat_dim

        self.token_emb = nn.Embedding(num_tokens, type0_feat_dim) if exists(num_tokens) else None

        # positional embedding

        self.num_positions = num_positions
        self.pos_emb = nn.Embedding(num_positions, type0_feat_dim) if exists(num_positions) else None

        # init embeddings

        if exists(self.token_emb):
            nn.init.normal_(self.token_emb.weight, std = 1e-2)

        if exists(self.pos_emb):
            nn.init.normal_(self.pos_emb.weight, std = 1e-2)

        # edges

        assert not (exists(num_edge_tokens) and not exists(edge_dim)), 'edge dimension (edge_dim) must be supplied if equiformer is to have edge tokens'

        self.edge_emb = nn.Embedding(num_edge_tokens, edge_dim) if exists(num_edge_tokens) else None
        self.has_edges = exists(edge_dim) and edge_dim > 0

        # whether to differentiate through basis, needed gradients for iterative refinement

        self.differentiable_coors = differentiable_coors

        # neighbors hyperparameters

        self.valid_radius = valid_radius
        self.num_neighbors = num_neighbors

        # main network

        self.tp_in  = DTP(
            self.dim_in,
            self.dim,
            edge_dim = edge_dim,
            radial_hidden_dim = radial_hidden_dim,
            splits = splits
        )

        # trunk

        self.layers = nn.ModuleList([])

        attention_klass = partial(DotProductAttention, use_cdist_sim = dot_product_attention_use_cdist_sim) if dot_product_attention else MLPAttention

        for ind in range(depth):
            self.layers.append(nn.ModuleList([
                attention_klass(
                    self.dim,
                    heads = heads,
                    dim_head = dim_head,
                    attend_self = attend_self,
                    edge_dim = edge_dim,
                    splits = splits,
                    single_headed_kv = single_headed_kv,
                    radial_hidden_dim = radial_hidden_dim,
                    **kwargs
                ),
                FeedForward(self.dim, include_htype_norms = ff_include_htype_norms)
            ]))

        # out

        self.norm = Norm(self.dim)

        proj_out_klass = Linear if linear_out else FeedForward

        self.ff_out = proj_out_klass(self.dim, (1,) * self.num_degrees) if reduce_dim_out else None

    def forward(
        self,
        feats,
        coors,
        mask = None,
        edges = None,
        return_pooled = False,
        neighbor_mask = None,
    ):
        _mask = mask

        if exists(self.token_emb):
            feats = self.token_emb(feats)

        if exists(self.pos_emb):
            assert feats.shape[1] <= self.num_positions, 'feature sequence length must be less than the number of positions given at init'
            feats = feats + self.pos_emb(torch.arange(feats.shape[1], device = feats.device))

        feats = self.embedding_grad_frac * feats + (1 - self.embedding_grad_frac) * feats.detach()

        assert not (self.has_edges and not exists(edges)), 'edge embedding (num_edge_tokens & edge_dim) must be supplied if one were to train on edge types'

        if torch.is_tensor(feats):
            feats = rearrange(feats, '... -> ... 1')
            feats = {0: feats}

        b, n, d, *_, device = *feats[0].shape, feats[0].device

        assert d == self.type0_feat_dim, f'feature dimension {d} must be equal to dimension given at init {self.type0_feat_dim}'
        assert set(map(int, feats.keys())) == set(range(self.input_degrees)), f'input must have {self.input_degrees} degree'

        num_degrees, neighbors, valid_radius = self.num_degrees, self.num_neighbors, self.valid_radius

        # cannot have a node attend to itself

        exclude_self_mask = rearrange(~torch.eye(n, dtype = torch.bool, device = device), 'i j -> 1 i j')
        remove_self = lambda t: t.masked_select(exclude_self_mask).reshape(b, n, n - 1)
        get_max_value = lambda t: torch.finfo(t.dtype).max

        # exclude edge of token to itself

        indices = repeat(torch.arange(n, device = device), 'j -> b i j', b = b, i = n)
        rel_pos  = rearrange(coors, 'b n d -> b n 1 d') - rearrange(coors, 'b n d -> b 1 n d')

        indices = indices.masked_select(exclude_self_mask).reshape(b, n, n - 1)
        rel_pos = rel_pos.masked_select(exclude_self_mask[..., None]).reshape(b, n, n - 1, 3)

        if exists(mask):
            mask = rearrange(mask, 'b i -> b i 1') * rearrange(mask, 'b j -> b 1 j')
            mask = mask.masked_select(exclude_self_mask).reshape(b, n, n - 1)

        if exists(edges):
            if exists(self.edge_emb):
                edges = self.edge_emb(edges)

            edges = edges.masked_select(exclude_self_mask[..., None]).reshape(b, n, n - 1, -1)

        rel_dist = rel_pos.norm(dim = -1)

        # rel_dist gets modified using adjacency or neighbor mask

        modified_rel_dist = rel_dist.clone()
        max_value = get_max_value(modified_rel_dist) # for masking out nodes from being considered as neighbors

        # neighbors

        if exists(neighbor_mask):
            neighbor_mask = remove_self(neighbor_mask)

            max_neighbors = neighbor_mask.sum(dim = -1).max().item()
            if max_neighbors > neighbors:
                print(f'neighbor_mask shows maximum number of neighbors as {max_neighbors} but specified number of neighbors is {neighbors}')

            modified_rel_dist = modified_rel_dist.masked_fill(~neighbor_mask, max_value)

        # if number of local neighbors by distance is set to 0, then only fetch the sparse neighbors defined by adjacency matrix

        if neighbors == 0:
            valid_radius = 0

        # get neighbors and neighbor mask, excluding self

        neighbors = int(min(neighbors, n - 1))
        total_neighbors = neighbors

        assert total_neighbors > 0, 'you must be fetching at least 1 neighbor'

        total_neighbors = int(min(total_neighbors, n - 1)) # make sure total neighbors does not exceed the length of the sequence itself

        dist_values, nearest_indices = modified_rel_dist.topk(total_neighbors, dim = -1, largest = False)
        neighbor_mask = dist_values <= valid_radius

        neighbor_rel_dist = batched_index_select(rel_dist, nearest_indices, dim = 2)
        neighbor_rel_pos = batched_index_select(rel_pos, nearest_indices, dim = 2)
        neighbor_indices = batched_index_select(indices, nearest_indices, dim = 2)

        if exists(mask):
            neighbor_mask = neighbor_mask & batched_index_select(mask, nearest_indices, dim = 2)

        if exists(edges):
            edges = batched_index_select(edges, nearest_indices, dim = 2)

        # embed relative distances

        neighbor_rel_dist = rearrange(neighbor_rel_dist, '... -> ... 1')

        # calculate basis

        basis = get_basis(neighbor_rel_pos, num_degrees - 1, differentiable = self.differentiable_coors)

        # main logic

        edge_info = EdgeInfo(neighbor_indices, neighbor_mask, edges)

        x = feats

        # project in

        x = self.tp_in(x, edge_info, rel_dist = neighbor_rel_dist, basis = basis)

        # transformer layers

        attn_kwargs = dict(
            edge_info = edge_info,
            rel_dist = neighbor_rel_dist,
            basis = basis,
            mask = _mask
        )

        for attn, ff in self.layers:
            x = residual_fn(attn(x, **attn_kwargs), x)
            x = residual_fn(ff(x), x)

        # norm

        x = self.norm(x)

        # reduce dim if specified

        if exists(self.ff_out):
            x = self.ff_out(x)
            x = {k: rearrange(v, '... 1 c -> ... c') for k, v in x.items()}

        if return_pooled:
            mask_fn = (lambda t: masked_mean(t, _mask, dim = 1))
            x = {k: mask_fn(v) for k, v in x.items()}

        # just return type 0 and type 1 features, reduced or not

        type0, type1 = x[0], x.get(1, None)

        type0 = rearrange(type0, '... 1 -> ...') # for type 0, just squeeze out the last dimension

        return Return(type0, type1)
