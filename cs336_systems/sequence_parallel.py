"""Boilerplate for sequence parallelism.

Sequence parallelism partitions sequence/token positions across ranks while
coordinating operations whose inputs or outputs span the sequence dimension.
"""

from __future__ import annotations

import torch


def scatter_sequence(x: torch.Tensor, *, dim: int = 1, group=None) -> torch.Tensor:
    """Return the local sequence slice."""
    raise NotImplementedError


def gather_sequence(x: torch.Tensor, *, dim: int = 1, group=None) -> torch.Tensor:
    """Gather sequence slices from all ranks."""
    raise NotImplementedError


def sequence_parallel_layer(module, x: torch.Tensor, *, sequence_dim: int = 1, group=None):
    """Apply a module to a sequence-parallel input."""
    raise NotImplementedError
