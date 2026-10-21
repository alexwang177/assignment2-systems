"""Unit tests and behavioral contracts for tensor parallelism."""

import pytest
import torch
from torch import nn

from cs336_systems.tensor_parallel import ColumnParallelLinear, RowParallelLinear, get_tensor_parallel_group


def test_tensor_parallel_interfaces_are_unimplemented():
    with pytest.raises(NotImplementedError):
        ColumnParallelLinear(8, 16)
    with pytest.raises(NotImplementedError):
        RowParallelLinear(8, 16)
    with pytest.raises(NotImplementedError):
        get_tensor_parallel_group()


@pytest.mark.xfail(raises=NotImplementedError, strict=False, reason="Tensor parallel layers are starter interfaces")
def test_column_parallel_linear_matches_reference_on_single_rank():
    torch.manual_seed(0)
    reference = nn.Linear(8, 16)
    parallel = ColumnParallelLinear(8, 16)
    parallel.load_state_dict(reference.state_dict())
    x = torch.randn(3, 8)
    assert torch.allclose(parallel(x), reference(x), atol=1e-6, rtol=1e-6)


@pytest.mark.xfail(raises=NotImplementedError, strict=False, reason="Tensor parallel layers are starter interfaces")
def test_row_parallel_linear_matches_reference_on_single_rank():
    torch.manual_seed(0)
    reference = nn.Linear(8, 16)
    parallel = RowParallelLinear(8, 16)
    parallel.load_state_dict(reference.state_dict())
    x = torch.randn(3, 8)
    assert torch.allclose(parallel(x), reference(x), atol=1e-6, rtol=1e-6)
