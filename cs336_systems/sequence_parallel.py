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


def scatter_sequence(x: torch.Tensor, *, dim: int = 1, group=None) -> torch.Tensor:
    """Scatter sequence positions across the ranks in ``group``.

    The future implementation should validate divisibility, return the local
    sequence slice, and define the inverse gather behavior in backward.
    """
    raise NotImplementedError


def gather_sequence(x: torch.Tensor, *, dim: int = 1, group=None) -> torch.Tensor:
    """Gather sequence slices from all ranks in ``group``.

    The future implementation should preserve autograd so gradients are
    returned to the corresponding local sequence slices.
    """
    raise NotImplementedError


def sequence_parallel_layer(module, x: torch.Tensor, *, sequence_dim: int = 1, group=None):
    """Apply a token-local module to a sequence-parallel input.

    This helper is intended for operations such as RMSNorm or dropout that
    can run independently on each local sequence slice. It should not be used
    as a naive replacement for full-sequence attention.
    """
    raise NotImplementedError


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
        raise NotImplementedError
