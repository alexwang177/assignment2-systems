"""Boilerplate for tensor (intra-layer) parallelism.

This module intentionally contains interfaces rather than an implementation.
The planned exercises are column-parallel and row-parallel linear layers.
"""

from __future__ import annotations

import torch
from torch import nn
from torch import distributed as dist
from torch.distributed.nn.functional import all_gather
from cs336_basics.model import Linear


class _CopyToTensorParallelRegion(torch.autograd.Function):
    """Identity in forward; all-reduce the gradient in backward.

    Use this for a replicated input to a column-parallel layer. The forward
    value is unchanged, but every rank must receive the sum of all local input
    gradient contributions during backward.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, group=None):
        ctx.group = group
        # TODO: return x unchanged.
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # TODO: all-reduce grad_output over ctx.group and return:
        # (reduced_grad_output, None)
        reduced_grad = grad_output.clone()
        dist.all_reduce(reduced_grad, group=ctx.group)
        return (reduced_grad, None)


class _ReduceFromTensorParallelRegion(torch.autograd.Function):
    """All-reduce in forward; identity in backward.

    Use this for the partial output of a row-parallel layer. The forward sums
    contributions from all ranks. The backward should pass the output gradient
    through unchanged to the local partial-output computation.
    """

    @staticmethod
    def forward(ctx, local_output: torch.Tensor, group=None):
        ctx.group = group
        # TODO: clone local_output if needed, all-reduce the clone over ctx.group,
        # and return the reduced result.
        reduced_out = local_output.clone()
        dist.all_reduce(reduced_out, group=ctx.group)
        return reduced_out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # TODO: return (grad_output, None). Do not all-reduce it here.
        return (grad_output, None)


def copy_to_tensor_parallel_region(x: torch.Tensor, group=None) -> torch.Tensor:
    """Apply the column-parallel input-copy autograd operation."""
    return _CopyToTensorParallelRegion.apply(x, group)


def reduce_from_tensor_parallel_region(local_output: torch.Tensor, group=None) -> torch.Tensor:
    """Apply the row-parallel output-reduction autograd operation."""
    return _ReduceFromTensorParallelRegion.apply(local_output, group)


class ColumnParallelLinear(nn.Module):
    """Linear layer whose output features are partitioned across ranks."""

    def __init__(self, in_features: int, out_features: int, *, gather_output: bool = True, group=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.group = group
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)

        if out_features % self.world_size != 0:
            raise Exception("Output dimension must be divisible by TP world size")

        self.linear = Linear(in_features, out_features // self.world_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        copied_x = copy_to_tensor_parallel_region(x, self.group)
        out = self.linear(copied_x)

        if self.gather_output:
            shards = all_gather(out, group=self.group)
            return torch.cat(shards, dim=-1)
        else:
            return out


class RowParallelLinear(nn.Module):
    """Linear layer whose input features are partitioned across ranks."""

    def __init__(self, in_features: int, out_features: int, *, input_is_parallel: bool = True, group=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        self.group = group
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)

        if in_features % self.world_size != 0:
            raise Exception("Input dimension must be divisible by TP world size")
        
        self.linear = Linear(in_features // self.world_size, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.input_is_parallel:
            # shard the input
            input = torch.chunk(x, self.world_size, dim=-1)[self.rank]
        else:
            input = x

        local_out = self.linear(input)

        # All reduce the final output
        return reduce_from_tensor_parallel_region(local_out, self.group)


def get_tensor_parallel_group():
    """Return the process group used for tensor-parallel collectives."""
    raise NotImplementedError
