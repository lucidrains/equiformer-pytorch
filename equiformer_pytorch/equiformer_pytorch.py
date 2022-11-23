from collections import namedtuple

from typing import Optional
from beartype import beartype

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# fibers

FiberEl = namedtuple('FiberEl', ['degrees', 'dim'])

@beartype
class Fiber(nn.Module):
    def __init__(
        self,
        structure: dict
    ):
        super().__init__()
        if isinstance(structure, dict):
            structure = [FiberEl(degree, dim) for degree, dim in structure.items()]
        self.structure = structure

    @property
    def dims(self):
        return uniq(map(lambda t: t[1], self.structure))

    @property
    def degrees(self):
        return map(lambda t: t[0], self.structure)

    @staticmethod
    def create(num_degrees, dim):
        dim_tuple = dim if isinstance(dim, tuple) else ((dim,) * num_degrees)
        return Fiber([FiberEl(degree, dim) for degree, dim in zip(range(num_degrees), dim_tuple)])

    def __getitem__(self, degree):
        return dict(self.structure)[degree]

    def __iter__(self):
        return iter(self.structure)

    def __mul__(self, fiber):
        return product(self.structure, fiber.structure)

    def __and__(self, fiber):
        out = []
        degrees_out = fiber.degrees
        for degree, dim in self:
            if degree in fiber.degrees:
                dim_out = fiber[degree]
                out.append((degree, dim, dim_out))
        return out

# main class

@beartype
class Equiformer(nn.Module):
    def __init__(
        self,
        *,
        dim
    ):
        super().__init__()

        self.dim = dim

    def forward(
        self,
        feats,
        coors,
        mask = None,
        return_type = None,
    ):

        if return_type == 0:
            return feats
        elif return_type == 1:
            return coors

        return (feats, coors)
