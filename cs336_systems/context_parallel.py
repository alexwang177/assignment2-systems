"""Learning scaffolds for context-parallel attention.

This module intentionally does not implement the algorithms yet.  The intended
learning order is:

1. all-gather attention;
2. variable-length all-gather attention;
3. ring attention;
4. Ulysses (all-to-all) attention;
5. hybrid ring/Ulysses attention.

Tensor layout for fixed-length attention is ``[..., heads, sequence, head_dim]``.
The sequence dimension is partitioned across a context-parallel process group.
For varlen attention, packed Q/K/V tensors use ``[tokens, heads, head_dim]`` and
``cu_seqlens`` describes the individual sequences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

ContextParallelMode = Literal["all_gather", "ring", "ulysses", "hybrid"]
SUPPORTED_CONTEXT_PARALLEL_MODES = ("all_gather", "ring", "ulysses", "hybrid")


class AllGatherWithReduceScatter(torch.autograd.Function):
    """Boilerplate for an autograd-aware all-gather.

    Forward contract:
        local ``x`` -> the rank-ordered concatenation of every rank's ``x``.

    Backward contract:
        Split ``grad_output`` into rank-ordered chunks, sum corresponding
        chunks across the process group with reduce-scatter, and return only
        this rank's local gradient.

    This is the key gradient rule for gathered K/V: every rank contributes to
    the gradient of every K/V partition.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, group=None, dim: int = -2):
        world_size = dist.get_world_size(group)
        ctx.world_size = world_size
        ctx.dim = dim
        ctx.group = group

        lst = [torch.zeros_like(x) for _ in range(world_size)]
        dist.all_gather(lst, x, group=group)

        return torch.concat(lst, dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        lst = list(torch.chunk(grad_output, chunks=ctx.world_size, dim=ctx.dim))
        lst = [chunk.contiguous() for chunk in lst]
        local_grad = torch.zeros_like(lst[0])
        dist.reduce_scatter(local_grad, lst, group=ctx.group)

        return (local_grad, None, None)


def all_gather_autograd(x: torch.Tensor, *, group=None, dim: int = -2) -> torch.Tensor:
    """Apply the custom autograd all-gather once its Function is implemented."""
    return AllGatherWithReduceScatter.apply(x, group, dim)


@dataclass(frozen=True)
class ContextParallelConfig:
    """Configuration shared by the CP implementations.

    ``causal`` refers to global sequence positions, not positions local to a
    rank.  Implementations must therefore accept or derive a global offset for
    each local query/key block.
    """

    mode: ContextParallelMode = "all_gather"
    causal: bool = True

    def __post_init__(self) -> None:
        if self.mode not in SUPPORTED_CONTEXT_PARALLEL_MODES:
            raise ValueError(f"Unknown context-parallel mode: {self.mode}")


class ContextParallelAttention(nn.Module):
    """Dispatch point for the CP attention algorithms.

    This is deliberately only an interface.  Keep the algorithm-specific
    functions below separate while learning; this wrapper can be completed
    after their individual correctness tests pass.
    """

    def __init__(
        self,
        mode: ContextParallelMode = "all_gather",
        *,
        group=None,
        causal: bool = True,
    ):
        super().__init__()
        self.config = ContextParallelConfig(mode=mode, causal=causal)
        self.group = group

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        query_start: int = 0,
        key_start: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        """Compute CP attention for local Q/K/V blocks.

        ``query_start`` and ``key_start`` are global sequence offsets.  They
        are required for correct causal masking when blocks are distributed.
        """
        raise NotImplementedError("Implement the selected context-parallel attention algorithm")


def exchange_context_partition(x: torch.Tensor, *, group=None, async_op: bool = False):
    """Exchange one local K/V context partition with a ring neighbor.

    This is the communication primitive needed by ring attention.  Decide and
    document the ring direction, ownership order, and asynchronous work-handle
    contract before implementing it.
    """
    raise NotImplementedError


def all_gather_attention(
    query: torch.Tensor, # [batch, heads, local_sequence, head_dim]
    key: torch.Tensor,   # [batch, heads, local_sequence, head_dim]
    value: torch.Tensor, # [batch, heads, local_sequence, head_dim]
    *,
    group=None,
    causal: bool = True,
    key_start: int = 0,
) -> torch.Tensor:
    """Stage 1: gather K/V, then compute attention for local queries."""

    if not dist.is_initialized():
        rank = 0
        world_size = 1
    else:
        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)

    local_seq_len = query.shape[-2]
    full_seq_len = local_seq_len * world_size

    query_start = rank * local_seq_len

    query_positions = query_start + torch.arange(start=0, end=local_seq_len, device=query.device)
    key_positions = key_start + torch.arange(start=0, end=full_seq_len, device=query.device)

    causal_mask = None
    if causal:
        causal_mask = query_positions[:, None] >= key_positions[None, :]
        causal_mask = causal_mask[None, None, :, :]

    if world_size > 1:
        # All gather the local K tensors and concatenate along sequence dimension
        full_key = all_gather_autograd(key, group=group, dim=-2)
        # All gather the local V tensors and concatenate along sequence dimension
        full_value = all_gather_autograd(value, group=group, dim=-2)
    else:
        full_key = key
        full_value = value

    return F.scaled_dot_product_attention(
        query=query,
        key=full_key,
        value=full_value,
        attn_mask=causal_mask
    )


def varlen_all_gather_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    group=None,
    causal: bool = True,
) -> torch.Tensor:
    """Stage 2: all-gather CP for packed variable-length sequences."""
    raise NotImplementedError


def ring_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    group=None,
    causal: bool = True,
    query_start: int = 0,
    key_start: int = 0,
) -> torch.Tensor:
    """Stage 3: circulate K/V blocks and combine them with online softmax."""
    raise NotImplementedError


def ulysses_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    group=None,
    causal: bool = True,
) -> torch.Tensor:
    """Stage 4: redistribute sequence/head layout with all-to-all."""
    raise NotImplementedError


def hybrid_ring_ulysses_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    ring_group=None,
    ulysses_group=None,
    causal: bool = True,
) -> torch.Tensor:
    """Stage 5: compose Ulysses redistribution with ring attention."""
    raise NotImplementedError
