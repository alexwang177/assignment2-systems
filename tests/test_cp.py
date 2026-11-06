"""Progressive contracts for learning context-parallel attention.

Implement these tests in order.  The algorithm tests are initially marked
expected-failure so the repository remains runnable while the learner works
through the stages.
"""

import pytest
import torch

from cs336_systems.context_parallel import (
    SUPPORTED_CONTEXT_PARALLEL_MODES,
    ContextParallelAttention,
    ContextParallelConfig,
    all_gather_attention,
    exchange_context_partition,
    hybrid_ring_ulysses_attention,
    ring_attention,
    ulysses_attention,
    varlen_all_gather_attention,
)


def _inputs(sequence_length=8):
    torch.manual_seed(0)
    query = torch.randn(1, 2, sequence_length, 4, requires_grad=True)
    key = torch.randn(1, 2, sequence_length, 4, requires_grad=True)
    value = torch.randn(1, 2, sequence_length, 4, requires_grad=True)
    return query, key, value


def test_cp_modes_and_config_are_explicit():
    assert SUPPORTED_CONTEXT_PARALLEL_MODES == ("all_gather", "ring", "ulysses", "hybrid")
    assert ContextParallelConfig().mode == "all_gather"
    assert ContextParallelConfig(mode="ring", causal=False).causal is False
    with pytest.raises(ValueError):
        ContextParallelConfig(mode="unknown")


def test_cp_wrapper_is_configurable_without_initializing_distributed():
    attention = ContextParallelAttention(mode="ring", causal=False)
    assert attention.config.mode == "ring"
    assert attention.config.causal is False


@pytest.mark.xfail(raises=NotImplementedError, strict=False, reason="Stage 1: implement all-gather CP")
def test_stage_1_all_gather_matches_reference_on_one_rank():
    query, key, value = _inputs()
    actual = all_gather_attention(query, key, value, causal=False)
    expected = torch.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=False)
    torch.testing.assert_close(actual, expected)


@pytest.mark.xfail(raises=NotImplementedError, strict=False, reason="Stage 1: implement the CP wrapper")
def test_stage_1_wrapper_matches_reference_on_one_rank():
    query, key, value = _inputs()
    actual = ContextParallelAttention(causal=False)(query, key, value)
    expected = torch.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=False)
    torch.testing.assert_close(actual, expected)


@pytest.mark.xfail(raises=NotImplementedError, strict=False, reason="Stage 2: implement packed varlen attention")
def test_stage_2_varlen_all_gather_preserves_sequence_boundaries():
    query, key, value = _inputs(sequence_length=5)
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)
    actual = varlen_all_gather_attention(query[0].transpose(0, 1), key[0].transpose(0, 1), value[0].transpose(0, 1), cu_seqlens, cu_seqlens, causal=False)
    assert actual.shape == (5, 2, 4)


@pytest.mark.xfail(raises=NotImplementedError, strict=False, reason="Stage 3: implement ring communication and online softmax")
def test_stage_3_ring_attention_matches_reference():
    query, key, value = _inputs()
    actual = ring_attention(query, key, value, causal=False)
    expected = torch.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=False)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.xfail(raises=NotImplementedError, strict=False, reason="Stage 3: implement the ring exchange primitive")
def test_stage_3_ring_exchange_has_a_defined_contract():
    exchange_context_partition(torch.randn(2, 2, 4, 4))


@pytest.mark.xfail(raises=NotImplementedError, strict=False, reason="Stage 4: implement Ulysses all-to-all layout changes")
def test_stage_4_ulysses_attention_matches_reference():
    query, key, value = _inputs()
    actual = ulysses_attention(query, key, value, causal=False)
    expected = torch.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=False)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.xfail(raises=NotImplementedError, strict=False, reason="Stage 5: implement hybrid CP")
def test_stage_5_hybrid_ring_ulysses_matches_reference():
    query, key, value = _inputs()
    actual = hybrid_ring_ulysses_attention(query, key, value, causal=False)
    expected = torch.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=False)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
