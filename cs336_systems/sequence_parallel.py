"""Boilerplate for sequence parallelism.

Sequence parallelism partitions sequence/token positions across the same
(or a compatible) tensor-parallel process group. For an activation shaped
``[batch, sequence, hidden]``, each rank typically owns
``[batch, sequence / group_size, hidden]``. Token-local operations can run on
that local slice; sequence-wide operations require an explicit gather or a
more specialized communication algorithm.

The scatter/gather helpers are intended to be autograd-aware. In a simple
SP+TP layout, sequence activations are gathered before a TP operation that
expects replicated sequence positions and scattered again afterward.
"""

from __future__ import annotations

import torch
from torch import nn
from torch import distributed as dist

class ScatterSequence(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, group=None, dim=-1):
        ctx.rank = dist.get_rank(group)
        ctx.world_size = dist.get_world_size(group)
        ctx.group = group
        ctx.dim = dim

        if x.shape[dim] % ctx.world_size != 0:
            raise ValueError("Sequence dimension length must be divisible by world size.")
        
        return torch.chunk(x, chunks=ctx.world_size, dim=dim)[ctx.rank]

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        local_grads = [torch.zeros_like(grad_output) for _ in range(ctx.world_size)]
        dist.all_gather(local_grads, grad_output, ctx.group)

        full_grad = torch.cat(local_grads, dim=ctx.dim)

        return full_grad, None, None
    

class GatherSequence(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, group=None, dim=-1):
        ctx.rank = dist.get_rank(group)
        ctx.world_size = dist.get_world_size(group)
        ctx.group = group
        ctx.dim = dim

        tensor_list = [torch.zeros_like(x) for _ in range(ctx.world_size)]
        dist.all_gather(tensor_list, x, group=ctx.group)
        return torch.cat(tensor_list, dim=ctx.dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):

        chunks = torch.chunk(grad_output, chunks=ctx.world_size, dim=ctx.dim)
        chunks = [chunk.contiguous() for chunk in chunks]

        local_grad = torch.zeros_like(chunks[ctx.rank])

        dist.reduce_scatter(local_grad, chunks, op=dist.ReduceOp.SUM, group=ctx.group)
        return local_grad, None, None


def scatter_sequence(x: torch.Tensor, *, dim: int = 1, group=None) -> torch.Tensor:
    """Scatter sequence positions across the ranks in ``group``.

    The future implementation should validate divisibility, return the local
    sequence slice, and define the inverse gather behavior in backward.
    """
    return ScatterSequence.apply(x, group, dim)


def gather_sequence(x: torch.Tensor, *, dim: int = 1, group=None) -> torch.Tensor:
    """Gather sequence slices from all ranks in ``group``.

    The future implementation should preserve autograd so gradients are
    returned to the corresponding local sequence slices.
    """
    return GatherSequence.apply(x, group, dim)


def sequence_parallel_layer(module, x: torch.Tensor, *, sequence_dim: int = 1, group=None):
    """Apply a token-local module to a sequence-parallel input.

    This helper is intended for operations such as RMSNorm or dropout that
    can run independently on each local sequence slice. It should not be used
    as a naive replacement for full-sequence attention.
    """
    return module(x)


class SequenceParallelTensorParallelMLP(nn.Module):
    """Scaffold for composing sequence parallelism with a TP MLP.

    Expected layout for ``forward``:

    1. scatter the full sequence across ``group``;
    2. apply the token-local normalization/module on each local slice;
    3. gather the sequence before the TP MLP if the TP layers expect a
       replicated sequence;
    4. run the column-parallel expansion and row-parallel projection;
    5. scatter and gather according to the chosen output contract.

    The implementation is intentionally left for the learner.
    """

    def __init__(self, local_module: nn.Module, tp_mlp: nn.Module, *, sequence_dim: int = 1, group=None):
        super().__init__()
        self.local_module = local_module
        self.tp_mlp = tp_mlp
        self.sequence_dim = sequence_dim
        self.group = group

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        local_x = scatter_sequence(x, dim=self.sequence_dim, group=self.group)
        local_x = self.local_module(local_x)
        full_x = gather_sequence(local_x, dim=self.sequence_dim, group=self.group)
        out = self.tp_mlp(full_x)
        return out
