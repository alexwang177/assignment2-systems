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
from torch import nn

ContextParallelMode = Literal["all_gather", "ring", "ulysses", "hybrid"]
SUPPORTED_CONTEXT_PARALLEL_MODES = ("all_gather", "ring", "ulysses", "hybrid")


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
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    group=None,
    causal: bool = True,
    query_start: int = 0,
    key_start: int = 0,
) -> torch.Tensor:
    """Stage 1: gather K/V, then compute attention for local queries."""
    raise NotImplementedError


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
