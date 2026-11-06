"""Progressive contracts for learning context-parallel attention.

Implement these tests in order.  The algorithm tests are initially marked
expected-failure so the repository remains runnable while the learner works
through the stages.
"""

import pytest
import torch
import torch.multiprocessing as mp

from cs336_systems.context_parallel import (
    SUPPORTED_CONTEXT_PARALLEL_MODES,
    AllGatherWithReduceScatter,
    all_gather_autograd,
    ContextParallelAttention,
    ContextParallelConfig,
    all_gather_attention,
    exchange_context_partition,
    hybrid_ring_ulysses_attention,
    ring_attention,
    ulysses_attention,
    varlen_all_gather_attention,
)
from tests.common import _cleanup_process_group, _setup_process_group


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


def test_custom_all_gather_function_has_forward_and_backward_hooks():
    assert hasattr(AllGatherWithReduceScatter, "forward")
    assert hasattr(AllGatherWithReduceScatter, "backward")


def _all_gather_backward_worker(rank, world_size):
    device = _setup_process_group(rank, world_size, backend="gloo")

    local_x = torch.ones(2, 2, 4, 4, device=device, requires_grad=True)
    full_x = all_gather_autograd(local_x, group=None, dim=-2)
    rank_weight = torch.full_like(full_x, rank + 1)
    (full_x * rank_weight).sum().backward()

    # Each local shard contributes to every rank's gathered output. The
    # corresponding gradient chunks are therefore summed across ranks.
    expected_grad = torch.full_like(local_x, sum(range(1, world_size + 1)))
    torch.testing.assert_close(local_x.grad, expected_grad)
    _cleanup_process_group()


def test_custom_all_gather_backward_reduce_scatters_gradients():
    mp.spawn(
        _all_gather_backward_worker,
        args=(2,),
        nprocs=2,
        join=True,
    )


def _all_gather_attention_backward_worker(rank, world_size, causal):
    device = _setup_process_group(rank, world_size, backend="gloo")
    torch.manual_seed(456)

    batch, heads, sequence, head_dim = 2, 2, 8, 4
    full_query = torch.randn(batch, heads, sequence, head_dim, device=device, requires_grad=True)
    full_key = torch.randn(batch, heads, sequence, head_dim, device=device, requires_grad=True)
    full_value = torch.randn(batch, heads, sequence, head_dim, device=device, requires_grad=True)
    grad_output = torch.randn(batch, heads, sequence, head_dim, device=device)

    reference_output = torch.nn.functional.scaled_dot_product_attention(
        full_query,
        full_key,
        full_value,
        is_causal=causal,
    )
    (reference_output * grad_output).sum().backward()

    expected_query_grad = full_query.grad.chunk(world_size, dim=-2)[rank]
    expected_key_grad = full_key.grad.chunk(world_size, dim=-2)[rank]
    expected_value_grad = full_value.grad.chunk(world_size, dim=-2)[rank]

    local_query = full_query.detach().chunk(world_size, dim=-2)[rank].contiguous().requires_grad_()
    local_key = full_key.detach().chunk(world_size, dim=-2)[rank].contiguous().requires_grad_()
    local_value = full_value.detach().chunk(world_size, dim=-2)[rank].contiguous().requires_grad_()

    actual_output = all_gather_attention(
        local_query,
        local_key,
        local_value,
        causal=causal,
    )
    local_grad_output = grad_output.chunk(world_size, dim=-2)[rank]
    (actual_output * local_grad_output).sum().backward()

    torch.testing.assert_close(local_query.grad, expected_query_grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(local_key.grad, expected_key_grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(local_value.grad, expected_value_grad, atol=1e-5, rtol=1e-5)
    _cleanup_process_group()


@pytest.mark.parametrize("causal", [False, True])
def test_all_gather_attention_backward_matches_full_attention(causal):
    mp.spawn(
        _all_gather_attention_backward_worker,
        args=(2, causal),
        nprocs=2,
        join=True,
    )


@pytest.mark.xfail(raises=NotImplementedError, strict=False, reason="Stage 1: implement all-gather CP")
def test_stage_1_all_gather_matches_reference_on_one_rank():
    query, key, value = _inputs()
    actual = all_gather_attention(query, key, value, causal=False)
    expected = torch.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=False)
    torch.testing.assert_close(actual, expected)


def _all_gather_forward_worker(rank, world_size, causal):
    device = _setup_process_group(rank, world_size, backend="gloo")
    torch.manual_seed(123)

    batch, heads, sequence, head_dim = 2, 2, 8, 4
    query = torch.randn(batch, heads, sequence, head_dim, device=device)
    key = torch.randn(batch, heads, sequence, head_dim, device=device)
    value = torch.randn(batch, heads, sequence, head_dim, device=device)

    local_query = query.chunk(world_size, dim=-2)[rank].contiguous()
    local_key = key.chunk(world_size, dim=-2)[rank].contiguous()
    local_value = value.chunk(world_size, dim=-2)[rank].contiguous()

    actual = all_gather_attention(local_query, local_key, local_value, causal=causal)
    expected = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        is_causal=causal,
    )
    expected = expected.chunk(world_size, dim=-2)[rank]

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    _cleanup_process_group()


@pytest.mark.parametrize("causal", [False, True])
def test_stage_1_all_gather_matches_reference_across_two_cpu_processes(causal):
    mp.spawn(
        _all_gather_forward_worker,
        args=(2, causal),
        nprocs=2,
        join=True,
    )


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
