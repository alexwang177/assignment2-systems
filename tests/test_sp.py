"""Unit tests and behavioral contracts for sequence parallelism."""

import pytest
import torch
from torch import nn

from cs336_systems.sequence_parallel import gather_sequence, scatter_sequence, sequence_parallel_layer


def test_sequence_parallel_interfaces_are_unimplemented():
    x = torch.randn(2, 4, 8)
    for operation in (scatter_sequence, gather_sequence):
        with pytest.raises(NotImplementedError):
            operation(x)
    with pytest.raises(NotImplementedError):
        sequence_parallel_layer(nn.Identity(), x)


@pytest.mark.xfail(raises=NotImplementedError, strict=False, reason="Sequence parallel collectives are starter interfaces")
def test_sequence_scatter_and_gather_are_identity_on_single_rank():
    x = torch.randn(2, 5, 8)
    local = scatter_sequence(x)
    reconstructed = gather_sequence(local)
    assert torch.equal(reconstructed, x)
