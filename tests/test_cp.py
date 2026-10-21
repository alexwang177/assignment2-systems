"""Unit tests and behavioral contracts for context parallelism."""

import pytest
import torch

from cs336_systems.context_parallel import ContextParallelAttention, exchange_context_partition


def test_context_parallel_interfaces_are_unimplemented():
    with pytest.raises(NotImplementedError):
        ContextParallelAttention()
    x = torch.randn(2, 4, 8)
    with pytest.raises(NotImplementedError):
        exchange_context_partition(x)


@pytest.mark.xfail(raises=NotImplementedError, strict=False, reason="Context parallel attention is a starter interface")
def test_context_parallel_attention_matches_scaled_dot_product_attention_on_single_rank():
    torch.manual_seed(0)
    query = torch.randn(2, 2, 4, 8)
    key = torch.randn(2, 2, 4, 8)
    value = torch.randn(2, 2, 4, 8)
    attention = ContextParallelAttention()
    actual = attention(query, key, value)
    expected = torch.nn.functional.scaled_dot_product_attention(query, key, value)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
