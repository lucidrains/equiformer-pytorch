from math import sqrt
from functools import partial
from itertools import product
from collections import namedtuple

from beartype.typing import Optional, Union, Tuple, Dict
from beartype import beartype

import torch
from torch import nn, is_tensor, Tensor
import torch.nn.functional as F

from taylor_series_linear_attention import TaylorSeriesLinearAttn

from opt_einsum import contract as opt_einsum

from equiformer_pytorch.basis import (
    get_basis,
    get_D_to_from_z_axis
)

from equiformer_pytorch.reversible import (
    SequentialSequence,
    ReversibleSequence
)

from equiformer_pytorch.utils import (
    exists,
    default,
    masked_mean,
    to_order,
    cast_tuple,
    safe_cat,
    fast_split,
    slice_for_centering_y_to_x,
    pad_for_centering_y_to_x
)

from einx import get_at

from einops import rearrange, repeat, reduce, einsum, pack, unpack
from einops.layers.torch import Rearrange

# constants

Return = namedtuple('Return', ['type0', 'type1'])

EdgeInfo = namedtuple('EdgeInfo', ['neighbor_indices', 'neighbor_mask', 'edges'])

# helpers

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

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

        if not any(t.requires_grad for t in (out[degree], residual[degree])):
            out[degree] += residual[degree]
        else:
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

def cdist(a, b, dim = -1, eps = 1e-5):
    a = a.expand_as(b)
    a, _ = pack_one(a, '* c')
    b, ps = pack_one(b, '* c')

    dist = F.pairwise_distance(a, b, p = 2)
    dist = unpack_one(dist, ps, '*')
    return dist

# classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        y = self.fn(x, **kwargs)
        if not y.requires_grad and not x.requires_grad:
            return x.add_(y)
        return x + y

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

class Linear(nn.Module):
    @beartype
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

    def init_zero_(self):
        for weight in self.weights:
            weight.data.zero_()

    def forward(self, x):
        out = {}

        for degree, weight in zip(self.degrees, self.weights):
            out[degree] = einsum(x[degree], weight, '... d m, d e -> ... e m')

        return out

class Norm(nn.Module):
    @beartype
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

class Gate(nn.Module):
    @beartype
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

class DTP(nn.Module):
    """ 'Tensor Product' - in the equivariant sense """

    @beartype
    def __init__(
        self,
        fiber_in: Tuple[int, ...],
        fiber_out: Tuple[int, ...],
        self_interaction = True,
        project_xi_xj = True,   # whether to project xi and xj and then sum, as in paper
        project_out = True,     # whether to do a project out after the "tensor product"
        pool = True,
        edge_dim = 0,
        radial_hidden_dim = 16
    ):
        super().__init__()
        self.fiber_in = fiber_in
        self.fiber_out = fiber_out
        self.edge_dim = edge_dim
        self.self_interaction = self_interaction
        self.pool = pool

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
                degree_min = min(degree_out, degree_in)

                self.kernel_unary[f'({degree_in},{degree_out})'] = Radial(degree_in, dim_in, degree_out, dim_out_from_degree_in, radial_hidden_dim = radial_hidden_dim, edge_dim = edge_dim)

        # whether a single token is self-interacting

        if self_interaction:
            self.self_interact = Linear(fiber_in, fiber_out)

        self.project_out = project_out
        if project_out:
            self.to_out = Linear(fiber_out, fiber_out)

    @beartype
    def forward(
        self,
        inp,
        basis,
        D,
        edge_info: EdgeInfo,
        rel_dist = None,
    ):
        neighbor_indices, neighbor_masks, edges = edge_info

        kernels = {}
        outputs = {}

        # neighbors

        if self.project_xi_xj:
            source, target = self.to_xi(inp), self.to_xj(inp)
        else:
            source, target = inp, inp

        # go through every permutation of input degree type to output degree type

        for degree_out, _ in enumerate(self.fiber_out):
            output = None
            m_out = to_order(degree_out)

            for degree_in, _ in enumerate(self.fiber_in):
                etype = f'({degree_in},{degree_out})'

                m_in = to_order(degree_in)
                m_min = min(m_in, m_out)

                degree_min = min(degree_in, degree_out)

                # get source and target (neighbor) representations

                xi, xj = source[degree_in], target[degree_in]

                x = get_at('b [i] d m, b j k -> b j k d m', xj, neighbor_indices)

                if self.project_xi_xj:
                    xi = rearrange(xi, 'b i d m -> b i 1 d m')
                    x = x + xi

                # multiply by D(R) - rotate to z-axis

                if degree_in > 0:
                    Di = D[degree_in]
                    x = einsum(Di, x, '... mi1 mi2, ... li mi1 -> ... li mi2')

                # remove some 0s if degree_in != degree_out

                maybe_input_slice = slice_for_centering_y_to_x(m_in, m_min)
                maybe_output_pad = pad_for_centering_y_to_x(m_out, m_min)

                x = x[..., maybe_input_slice]

                # process input, edges, and basis in chunks along the sequence dimension

                kernel_fn = self.kernel_unary[etype]
                edge_features = safe_cat(edges, rel_dist, dim = -1)

                B = basis.get(etype, None)
                R = kernel_fn(edge_features)

                # mo depends only on mi (or other way around), removing yet another dimension

                if not exists(B): # degree_in or degree_out is 0
                    output_chunk = einsum(R, x, '... lo li, ... li mi -> ... lo mi')
                else:
                    y = x.clone()

                    x = repeat(x, '... mi -> ... mi mf r', mf = (B.shape[-1] + 1) // 2, r = 2) # mf + 1, so that mf can be divided in 2
                    x, x_to_flip = x.unbind(dim = -1)

                    x_flipped = torch.flip(x_to_flip, dims = (-2,)) # flip on the mi axis, as the basis alternates between diagonal and flipped diagonal across mf
                    x = torch.stack((x, x_flipped), dim = -1)
                    x = rearrange(x, '... mf r -> ... (mf r)', r = 2)
                    x = x[..., :-1]

                    output_chunk = opt_einsum('... o i, m f, ... i m f -> ... o m', R, B, x)

                # in the case that degree_out < degree_in

                output_chunk = F.pad(output_chunk, (maybe_output_pad, maybe_output_pad), value = 0.)

                output = safe_cat(output, output_chunk, dim = -2)

            # multiply by D(R^-1) - rotate back from z-axis

            if degree_out > 0:
                Do = D[degree_out]
                output = einsum(output, Do, '... lo mo1, ... mo2 mo1 -> ... lo mo2')

            # pool or not along j (neighbors) dimension

            if self.pool:
                output = masked_mean(output, neighbor_masks, dim = 2)

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

class Radial(nn.Module):
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
            nn.Linear(mid_dim, nc_in * nc_out),
            Rearrange('... (lo li) -> ... lo li', li = nc_in, lo = nc_out)
        )

    def forward(self, feat):
        return self.rp(feat)

# feed forwards

class FeedForward(nn.Module):
    @beartype
    def __init__(
        self,
        fiber: Tuple[int, ...],
        fiber_out: Optional[Tuple[int, ...]] = None,
        mult = 4,
        include_htype_norms = True,
        init_out_zero = True
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

        if init_out_zero:
            self.project_out.init_zero_()

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

# global linear attention

class LinearAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.heads = heads
        dim_inner = dim_head * heads
        self.to_qkv = nn.Linear(dim, dim_inner * 3)

    def forward(self, x, mask = None):
        has_degree_m_dim = x.ndim == 4

        if has_degree_m_dim:
            x = rearrange(x, '... 1 -> ...')

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        if exists(mask):
            mask = rearrange(mask, 'b n -> b 1 n 1')
            k = k.masked_fill(~mask, -torch.finfo(q.dtype).max)
            v = v.masked_fill(~mask, 0.)

        k = k.softmax(dim = -2)
        q = q.softmax(dim = -1)

        kv = einsum(k, v, 'b h n d, b h n e -> b h d e')
        out = einsum(kv, q, 'b h d e, b h n d -> b h n e')
        out = rearrange(out, 'b h n d -> b n (h d)')

        if has_degree_m_dim:
            out = rearrange(out, '... -> ... 1')

        return out

# attention

class L2DistAttention(nn.Module):
    @beartype
    def __init__(
        self,
        fiber: Tuple[int, ...],
        dim_head: Union[int, Tuple[int, ...]] = 64,
        heads: Union[int, Tuple[int, ...]] = 8,
        attend_self = False,
        edge_dim = None,
        single_headed_kv = False,
        radial_hidden_dim = 64,
        splits = 4,
        linear_attn_dim_head = 8,
        num_linear_attn_heads = 0,
        init_out_zero = True,
        gate_attn_head_outputs = True
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

        kv_hidden_fiber = hidden_fiber if not single_headed_kv else dim_head
        kv_hidden_fiber = tuple(dim * 2 for dim in kv_hidden_fiber)

        self.scale = tuple(dim ** -0.5 for dim in dim_head)
        self.heads = heads

        self.prenorm = Norm(fiber)

        self.to_q = Linear(fiber, hidden_fiber)
        self.to_kv = DTP(fiber, kv_hidden_fiber, radial_hidden_dim = radial_hidden_dim, edge_dim = edge_dim, pool = False, self_interaction = attend_self)

        # linear attention heads

        self.has_linear_attn = num_linear_attn_heads > 0

        if self.has_linear_attn:
            degree_zero_dim = fiber[0]
            self.linear_attn = TaylorSeriesLinearAttn(degree_zero_dim, dim_head = linear_attn_dim_head, heads = num_linear_attn_heads, combine_heads = False, gate_value_heads = True)
            hidden_fiber = tuple_set_at_index(hidden_fiber, 0, hidden_fiber[0] + linear_attn_dim_head * num_linear_attn_heads)

        # gating heads across all degree outputs
        # to allow for attending to nothing

        self.attn_head_gates = None

        if gate_attn_head_outputs:
            self.attn_head_gates = nn.Sequential(
                Rearrange('... d 1 -> ... d'),
                nn.Linear(fiber[0], sum(heads)),
                nn.Sigmoid(),
                Rearrange('... n h -> ... h n 1 1')
            )

        # combine heads

        self.to_out = Linear(hidden_fiber, fiber)

        if init_out_zero:
            self.to_out.init_zero_()

    @beartype
    def forward(
        self,
        features,
        edge_info: EdgeInfo,
        rel_dist,
        basis,
        D,
        mask = None
    ):
        one_head_kv = self.single_headed_kv

        device, dtype = get_tensor_device_and_dtype(features)
        neighbor_indices, neighbor_mask, edges = edge_info

        if exists(neighbor_mask):
            neighbor_mask = rearrange(neighbor_mask, 'b i j -> b 1 i j')

            if self.attend_self:
                neighbor_mask = F.pad(neighbor_mask, (1, 0), value = True)

        features = self.prenorm(features)

        # generate queries, keys, values

        queries = self.to_q(features)

        keyvalues   = self.to_kv(
            features,
            edge_info = edge_info,
            rel_dist = rel_dist,
            basis = basis,
            D = D
        )

        # create gates

        gates = (None,) * len(self.heads)

        if exists(self.attn_head_gates):
            gates = self.attn_head_gates(features[0]).split(self.heads, dim = -4)

        # single headed vs not

        kv_einsum_eq = 'b h i j d m' if not one_head_kv else 'b i j d m'

        outputs = {}

        for degree, gate, h, scale in zip(features.keys(), gates, self.heads, self.scale):
            is_degree_zero = degree == 0

            q, kv = map(lambda t: t[degree], (queries, keyvalues))

            q = rearrange(q, 'b i (h d) m -> b h i d m', h = h)

            if not one_head_kv:
                kv = rearrange(kv, f'b i j (h d) m -> b h i j d m', h = h)

            k, v = kv.chunk(2, dim = -2)

            if one_head_kv:
                k = repeat(k, 'b i j d m -> b h i j d m', h = h)

            q = repeat(q, 'b h i d m -> b h i j d m', j = k.shape[-3])

            if is_degree_zero:
                q, k = map(lambda t: rearrange(t, '... 1 -> ...'), (q, k))

            sim = -cdist(q, k) * scale

            if not is_degree_zero:
                sim = sim.sum(dim = -1)
                sim = sim.masked_fill(~neighbor_mask, -torch.finfo(sim.dtype).max)

            attn = sim.softmax(dim = -1)
            out = einsum(attn, v, f'b h i j, {kv_einsum_eq} -> b h i d m')

            if exists(gate):
                out = out * gate

            outputs[degree] = rearrange(out, 'b h n d m -> b n (h d) m')

        if self.has_linear_attn:
            linear_attn_input = rearrange(features[0], '... 1 -> ...')
            lin_attn_out = self.linear_attn(linear_attn_input, mask = mask)
            lin_attn_out = rearrange(lin_attn_out, '... -> ... 1')
            outputs[0] = torch.cat((outputs[0], lin_attn_out), dim = -2)

        return self.to_out(outputs)

class MLPAttention(nn.Module):
    @beartype
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
        linear_attn_dim_head = 8,
        num_linear_attn_heads = 0,
        init_out_zero = True,
        gate_attn_head_outputs = True,
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

        self.attend_self = attend_self

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

        self.to_attn_and_v = DTP(fiber, intermediate_fiber, radial_hidden_dim = radial_hidden_dim, edge_dim = edge_dim, pool = False, self_interaction = attend_self)

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

        # linear attention heads

        self.has_linear_attn = num_linear_attn_heads > 0

        if self.has_linear_attn:
            degree_zero_dim = fiber[0]
            self.linear_attn = TaylorSeriesLinearAttn(degree_zero_dim, dim_head = linear_attn_dim_head, heads = num_linear_attn_heads, combine_heads = False)
            hidden_fiber = tuple_set_at_index(hidden_fiber, 0, hidden_fiber[0] + linear_attn_dim_head * num_linear_attn_heads)

        # gating heads across all degree outputs
        # to allow for attending to nothing

        self.attn_head_gates = None

        if gate_attn_head_outputs:
            self.attn_head_gates = nn.Sequential(
                Rearrange('... d 1 -> ... d'),
                nn.Linear(fiber[0], sum(heads)),
                nn.Sigmoid(),
                Rearrange('... h -> ... h 1 1')
            )

        # combining heads and projection out

        self.to_out = Linear(hidden_fiber, fiber)

        if init_out_zero:
            self.to_out.init_zero_()

    @beartype
    def forward(
        self,
        features,
        edge_info: EdgeInfo,
        rel_dist,
        basis,
        D,
        mask = None
    ):
        one_headed_kv = self.single_headed_kv

        _, neighbor_mask, _ = edge_info

        if exists(neighbor_mask):
            if self.attend_self:
                neighbor_mask = F.pad(neighbor_mask, (1, 0), value = True)

            neighbor_mask = rearrange(neighbor_mask, '... -> ... 1')

        features = self.prenorm(features)

        intermediate = self.to_attn_and_v(
            features,
            edge_info = edge_info,
            rel_dist = rel_dist,
            basis = basis,
            D = D
        )

        *attn_branch_type0, value_branch_type0 = intermediate[0].split(self.intermediate_type0_split, dim = -2)

        intermediate[0] = value_branch_type0

        # create gates

        gates = (None,) * len(self.heads)

        if exists(self.attn_head_gates):
            gates = self.attn_head_gates(features[0]).split(self.heads, dim = -3)

        # process the attention branch

        attentions = []

        for fn, attn_intermediate, scale in zip(self.to_attn_logits, attn_branch_type0, self.scale):
            attn_intermediate = rearrange(attn_intermediate, '... 1 -> ...')
            attn_logits = fn(attn_intermediate)
            attn_logits = attn_logits * scale

            if exists(neighbor_mask):
                attn_logits = attn_logits.masked_fill(~neighbor_mask, -torch.finfo(attn_logits.dtype).max)

            attn = attn_logits.softmax(dim = -2) # (batch, source, target, heads)
            attentions.append(attn)

        # process values branch

        values = self.to_values(intermediate)

        # aggregate values with attention matrix

        outputs = {}

        value_einsum_eq = 'b i j h d m' if not one_headed_kv else 'b i j d m'

        for degree, (attn, value, gate, h) in enumerate(zip(attentions, values.values(), gates, self.heads)):
            if not one_headed_kv:
                value = rearrange(value, 'b i j (h d) m -> b i j h d m', h = h)

            out = einsum(attn, value, f'b i j h, {value_einsum_eq} -> b i h d m')

            if exists(gate):
                out = out * gate

            out = rearrange(out, 'b i h d m -> b i (h d) m')
            outputs[degree] = out

        # linear attention

        if self.has_linear_attn:
            linear_attn_input = rearrange(features[0], '... 1 -> ...')
            lin_attn_out = self.linear_attn(linear_attn_input, mask = mask)
            lin_attn_out = rearrange(lin_attn_out, '... -> ... 1')

            outputs[0] = torch.cat((outputs[0], lin_attn_out), dim = -2)

        # combine heads out

        return self.to_out(outputs)

# main class

class Equiformer(nn.Module):
    @beartype
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
        splits = 4,
        linear_out = True,
        embedding_grad_frac = 0.5,
        single_headed_kv = False,           # whether to do single headed key/values for dot product attention, to save on memory and compute
        ff_include_htype_norms = False,     # whether for type0 projection to also involve norms of all higher types, in feedforward first projection. this allows for all higher types to be gated by other type norms
        l2_dist_attention = True,           # turn to False to use MLP attention as proposed in paper, but dot product attention with -cdist similarity is still far better, and i haven't even rotated distances (rotary embeddings) into the type 0 features yet
        reversible = False,                 # turns on reversible networks, to scale depth without incurring depth times memory cost
        attend_sparse_neighbors = False,    # ability to accept an adjacency matrix
        gate_attn_head_outputs = True,      # gate each attention head output, to allow for attending to nothing
        num_adj_degrees_embed = None,
        adj_dim = 0,
        max_sparse_neighbors = float('inf'),
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

        # sparse neighbors, derived from adjacency matrix or edges being passed in

        self.attend_sparse_neighbors = attend_sparse_neighbors
        self.max_sparse_neighbors = max_sparse_neighbors

        # adjacent neighbor derivation and embed

        assert not exists(num_adj_degrees_embed) or num_adj_degrees_embed >= 1, 'number of adjacent degrees to embed must be 1 or greater'

        self.num_adj_degrees_embed = num_adj_degrees_embed
        self.adj_emb = nn.Embedding(num_adj_degrees_embed + 1, adj_dim) if exists(num_adj_degrees_embed) and adj_dim > 0 else None

        edge_dim = (edge_dim if self.has_edges else 0) + (adj_dim if exists(self.adj_emb) else 0)

        # neighbors hyperparameters

        self.valid_radius = valid_radius
        self.num_neighbors = num_neighbors

        # main network

        self.tp_in  = DTP(
            self.dim_in,
            self.dim,
            edge_dim = edge_dim,
            radial_hidden_dim = radial_hidden_dim
        )

        # trunk

        self.layers = []

        attention_klass = L2DistAttention if l2_dist_attention else MLPAttention

        for ind in range(depth):
            self.layers.append((
                attention_klass(
                    self.dim,
                    heads = heads,
                    dim_head = dim_head,
                    attend_self = attend_self,
                    edge_dim = edge_dim,
                    single_headed_kv = single_headed_kv,
                    radial_hidden_dim = radial_hidden_dim,
                    gate_attn_head_outputs = gate_attn_head_outputs,
                    **kwargs
                ),
                FeedForward(self.dim, include_htype_norms = ff_include_htype_norms)
            ))

        SequenceKlass = ReversibleSequence if reversible else SequentialSequence

        self.layers = SequenceKlass(self.layers)

        # out

        self.norm = Norm(self.dim)

        proj_out_klass = Linear if linear_out else FeedForward

        self.ff_out = proj_out_klass(self.dim, (1,) * self.num_degrees) if reduce_dim_out else None

        # basis is now constant
        # pytorch does not have BufferDict yet, just improvise a solution with python property

        self.basis = get_basis(self.num_degrees - 1)

    @property
    def basis(self):
        out = dict()
        for k in self.basis_keys:
            out[k] = getattr(self, f'basis:{k}')
        return out

    @basis.setter
    def basis(self, basis):
        self.basis_keys = basis.keys()

        for k, v in basis.items():
            self.register_buffer(f'basis:{k}', v)

    @property
    def device(self):
        return next(self.parameters()).device

    @beartype
    def forward(
        self,
        inputs: Union[Tensor, Dict[int, Tensor]],
        coors: Tensor,
        mask = None,
        adj_mat = None,
        edges = None,
        return_pooled = False
    ):
        _mask, device = mask, self.device

        # apply token embedding and positional embedding to type-0 features
        # (if type-0 feats are passed as a tensor they are expected to be of a flattened shape (batch, seq, n_feats)
        # but if they are passed in a dict (fiber) they are expected to be of a unified shape (batch, seq, n_feats, 1=2*0+1))

        if is_tensor(inputs):
            inputs = {0: inputs}

        feats = inputs[0]

        if feats.ndim == 4:
            feats = rearrange(feats, '... 1 -> ...')

        if exists(self.token_emb):
            assert feats.ndim == 2
            feats = self.token_emb(feats)

        if exists(self.pos_emb):
            seq_len = feats.shape[1]
            assert seq_len <= self.num_positions, 'feature sequence length must be less than the number of positions given at init'

            feats = feats + self.pos_emb(torch.arange(seq_len, device = device))

        feats = self.embedding_grad_frac * feats + (1 - self.embedding_grad_frac) * feats.detach()

        assert not (self.has_edges and not exists(edges)), 'edge embedding (num_edge_tokens & edge_dim) must be supplied if one were to train on edge types'

        b, n, d = feats.shape

        feats = rearrange(feats, 'b n d -> b n d 1')

        inputs[0] = feats

        assert d == self.type0_feat_dim, f'feature dimension {d} must be equal to dimension given at init {self.type0_feat_dim}'
        assert set(map(int, inputs.keys())) == set(range(self.input_degrees)), f'input must have {self.input_degrees} degree'

        num_degrees, neighbors, max_sparse_neighbors, valid_radius = self.num_degrees, self.num_neighbors, self.max_sparse_neighbors, self.valid_radius

        assert self.attend_sparse_neighbors or neighbors > 0, 'you must either attend to sparsely bonded neighbors, or set number of locally attended neighbors to be greater than 0'

        # cannot have a node attend to itself

        exclude_self_mask = rearrange(~torch.eye(n, dtype = torch.bool, device = device), 'i j -> 1 i j')
        remove_self = lambda t: t.masked_select(exclude_self_mask).reshape(b, n, n - 1)
        get_max_value = lambda t: torch.finfo(t.dtype).max

        # create N-degrees adjacent matrix from 1st degree connections

        if exists(adj_mat) and adj_mat.ndim == 2:
            adj_mat = repeat(adj_mat, 'i j -> b i j', b = b)

        if exists(self.num_adj_degrees_embed):
            adj_indices = adj_mat.long()

            for ind in range(self.num_adj_degrees_embed - 1):
                degree = ind + 2

                next_degree_adj_mat = (adj_mat.float() @ adj_mat.float()) > 0
                next_degree_mask = next_degree_adj_mat & ~adj_mat
                adj_indices = adj_indices.masked_fill(next_degree_mask, degree)
                adj_mat = next_degree_adj_mat.clone()

            adj_indices = adj_indices.masked_select(exclude_self_mask)
            adj_indices = rearrange(adj_indices, '(b i j) -> b i j', b = b, i = n, j = n - 1)

        # calculate sparsely connected neighbors

        sparse_neighbor_mask = None
        num_sparse_neighbors = 0

        if self.attend_sparse_neighbors:
            assert exists(adj_mat), 'adjacency matrix must be passed in (keyword argument adj_mat)'
            adj_mat = remove_self(adj_mat)

            adj_mat_values = adj_mat.float()
            adj_mat_max_neighbors = reduce(adj_mat_values, '... i j -> ... i', 'sum').amax().item()

            if max_sparse_neighbors < adj_mat_max_neighbors:
                eps = 1e-2
                noise = torch.empty_like(adj_mat_values).uniform_(-eps, eps)
                adj_mat_values += noise

            num_sparse_neighbors = int(min(max_sparse_neighbors, adj_mat_max_neighbors))
            values, indices = adj_mat_values.topk(num_sparse_neighbors, dim = -1)
            sparse_neighbor_mask = torch.zeros_like(adj_mat_values).scatter_(-1, indices, values)
            sparse_neighbor_mask = sparse_neighbor_mask > 0.5

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

        # make sure padding tokens are not considered when ordering by relative distance

        if exists(mask):
            modified_rel_dist = modified_rel_dist.masked_fill(~mask, max_value)

        # use sparse neighbor mask to assign priority of bonded

        if exists(sparse_neighbor_mask):
            modified_rel_dist = modified_rel_dist.masked_fill(sparse_neighbor_mask, 0.)

        # if number of local neighbors by distance is set to 0, then only fetch the sparse neighbors defined by adjacency matrix

        if neighbors == 0:
            valid_radius = 0

        # get neighbors and neighbor mask, excluding self

        neighbors = int(min(neighbors, n - 1))

        total_neighbors = int(neighbors + num_sparse_neighbors)
        assert total_neighbors > 0, 'you must be fetching at least 1 neighbor'

        total_neighbors = int(min(total_neighbors, n - 1)) # make sure total neighbors does not exceed the length of the sequence itself

        dist_values, nearest_indices = modified_rel_dist.topk(total_neighbors, dim = -1, largest = False)
        neighbor_mask = dist_values <= valid_radius

        neighbor_rel_dist = get_at('b i [j], b i k -> b i k', rel_dist, nearest_indices)
        neighbor_rel_pos = get_at('b i [j] c, b i k -> b i k c', rel_pos, nearest_indices)
        neighbor_indices = get_at('b i [j], b i k -> b i k', indices, nearest_indices)

        if exists(mask):
            nearest_mask = get_at('b i [j], b i k -> b i k', mask, nearest_indices)
            neighbor_mask = neighbor_mask & nearest_mask

        if exists(edges):
            edges = get_at('b i [j] d, b i k -> b i k d', edges, nearest_indices)

        # embed relative distances

        neighbor_rel_dist = rearrange(neighbor_rel_dist, '... -> ... 1')

        # calculate basis

        D = get_D_to_from_z_axis(neighbor_rel_pos, num_degrees - 1)

        # main logic

        edge_info = EdgeInfo(neighbor_indices, neighbor_mask, edges)

        x = inputs

        # project in

        x = self.tp_in(
            x,
            edge_info = edge_info,
            rel_dist = neighbor_rel_dist, 
            basis = self.basis,
            D = D
        )

        # transformer layers

        attn_kwargs = dict(
            edge_info = edge_info,
            rel_dist = neighbor_rel_dist,
            basis = self.basis,
            D = D,
            mask = _mask
        )

        x = self.layers(x, **attn_kwargs)

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
