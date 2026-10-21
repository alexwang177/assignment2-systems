"""Boilerplate for context parallelism.

Context parallelism partitions long-context attention work across ranks.  The
future implementation should specify how query, key, value, and KV-cache
partitions communicate.
"""

from __future__ import annotations

import torch
from torch import nn


class ContextParallelAttention(nn.Module):
    """Attention module whose context dimension is distributed across ranks."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("Define context partitioning and attention collectives")

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs):
        raise NotImplementedError


def exchange_context_partition(x: torch.Tensor, *, group=None):
    """Exchange the context partition needed by the local attention rank."""
    raise NotImplementedError
