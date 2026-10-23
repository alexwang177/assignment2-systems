"""Tests and learning contracts for sequence parallelism.

These tests use the common SP+TP layout:

* sequence-parallel activations have shape ``[batch, sequence/P, hidden]``;
* token-local operations run on the local sequence slice;
* sequence activations are gathered before a TP operation that expects the
  sequence to be replicated;
* the result can be scattered again after the TP operation.

The distributed tests are expected to fail while the starter helpers remain
unimplemented.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from cs336_basics.model import Linear, RMSNorm
from cs336_systems.sequence_parallel import (
    SequenceParallelTensorParallelMLP,
    gather_sequence,
    scatter_sequence,
    sequence_parallel_layer,
)
from cs336_systems.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from tests.common import _cleanup_process_group, _setup_process_group


WORLD_SIZE = 2


def test_sequence_parallel_interfaces_are_unimplemented():
    x = torch.randn(2, 4, 8)
    for operation in (scatter_sequence, gather_sequence):
        with pytest.raises(NotImplementedError):
            operation(x)
    with pytest.raises(NotImplementedError):
        sequence_parallel_layer(nn.Identity(), x)
    composition = SequenceParallelTensorParallelMLP(nn.Identity(), nn.Identity())
    with pytest.raises(NotImplementedError):
        composition(x)


def _sp_scatter_gather_worker(rank, world_size):
    device = _setup_process_group(rank, world_size, backend="gloo")
    torch.manual_seed(0)

    batch, sequence, hidden = 2, 8, 6
    x = torch.randn(batch, sequence, hidden, device=device)
    local_x = scatter_sequence(x, dim=1, group=None)

    assert local_x.shape == (batch, sequence // world_size, hidden)

    reconstructed = gather_sequence(local_x, dim=1, group=None)
    assert torch.allclose(reconstructed, x, atol=1e-6, rtol=1e-6)

    _cleanup_process_group()


@pytest.mark.xfail(strict=False, reason="Sequence parallel collectives are starter interfaces")
def test_sequence_scatter_gather_round_trip_across_ranks():
    mp.spawn(_sp_scatter_gather_worker, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)


def _sp_gradient_worker(rank, world_size):
    device = _setup_process_group(rank, world_size, backend="gloo")
    torch.manual_seed(1)

    x = torch.randn(2, 8, 6, device=device).requires_grad_()
    reference_x = x.detach().clone().requires_grad_()
    grad_output = torch.randn_like(x)

    expected = reference_x * 2
    (expected * grad_output).sum().backward()

    local_x = scatter_sequence(x, dim=1, group=None)
    reconstructed = gather_sequence(local_x * 2, dim=1, group=None)
    ((reconstructed * grad_output).sum() / world_size).backward()

    assert torch.allclose(x.grad, reference_x.grad, atol=1e-6, rtol=1e-6)
    _cleanup_process_group()


@pytest.mark.xfail(strict=False, reason="Sequence parallel autograd collectives are starter interfaces")
def test_sequence_scatter_gather_backward_matches_reference():
    mp.spawn(_sp_gradient_worker, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)


def _copy_column_shard(tp_layer, reference_layer, rank, world_size):
    with torch.no_grad():
        tp_layer.linear.weight.copy_(reference_layer.weight.chunk(world_size, dim=0)[rank])


def _copy_row_shard(tp_layer, reference_layer, rank, world_size):
    with torch.no_grad():
        tp_layer.linear.weight.copy_(reference_layer.weight.chunk(world_size, dim=1)[rank])


def _sp_tp_mlp_worker(rank, world_size):
    device = _setup_process_group(rank, world_size, backend="gloo")
    torch.manual_seed(2)

    hidden, intermediate = 8, 16
    reference_norm = RMSNorm(hidden).to(device)
    reference_w1 = Linear(hidden, intermediate).to(device)
    reference_w2 = Linear(intermediate, hidden).to(device)

    tp_norm = RMSNorm(hidden).to(device)
    tp_w1 = ColumnParallelLinear(hidden, intermediate, gather_output=False).to(device)
    tp_w2 = RowParallelLinear(intermediate, hidden, input_is_parallel=True).to(device)

    with torch.no_grad():
        tp_norm.weight.copy_(reference_norm.weight)
    _copy_column_shard(tp_w1, reference_w1, rank, world_size)
    _copy_row_shard(tp_w2, reference_w2, rank, world_size)

    x = torch.randn(2, 8, hidden, device=device)
    expected = reference_w2(torch.nn.functional.silu(reference_w1(reference_norm(x))))

    tp_mlp = nn.Sequential(tp_w1, nn.SiLU(), tp_w2)
    tp_block = SequenceParallelTensorParallelMLP(
        local_module=tp_norm,
        tp_mlp=tp_mlp,
        sequence_dim=1,
        group=None,
    )

    # The composition itself is intentionally left for the learner to write
    # in SequenceParallelTensorParallelMLP.forward().
    actual = tp_block(x)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    _cleanup_process_group()


@pytest.mark.xfail(strict=False, reason="Sequence-parallel and tensor-parallel composition is a starter exercise")
def test_sequence_parallel_composes_with_tensor_parallel_mlp():
    mp.spawn(_sp_tp_mlp_worker, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)
