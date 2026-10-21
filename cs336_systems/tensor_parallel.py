"""Boilerplate for tensor (intra-layer) parallelism.

This module intentionally contains interfaces rather than an implementation.
The planned exercises are column-parallel and row-parallel linear layers.
"""

from __future__ import annotations

import torch
from torch import nn


class ColumnParallelLinear(nn.Module):
    """Linear layer whose output features are partitioned across ranks."""

    def __init__(self, in_features: int, out_features: int, *, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        raise NotImplementedError("Implement column-wise weight sharding and output assembly")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class RowParallelLinear(nn.Module):
    """Linear layer whose input features are partitioned across ranks."""

    def __init__(self, in_features: int, out_features: int, *, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        raise NotImplementedError("Implement row-wise weight sharding and output reduction")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def get_tensor_parallel_group():
    """Return the process group used for tensor-parallel collectives."""
    raise NotImplementedError
