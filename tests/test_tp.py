"""Unit and distributed correctness tests for tensor parallel linear layers.

Conventions used by these tests follow the common Megatron-style API:

* ColumnParallelLinear receives replicated input and partitions output features.
  ``gather_output=True`` returns the full output on every rank.
* RowParallelLinear receives input already partitioned along the feature
  dimension when ``input_is_parallel=True`` and all-reduces partial outputs.
* The starter layers are bias-free, matching ``cs336_basics.Linear``.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from cs336_systems.tensor_parallel import ColumnParallelLinear, RowParallelLinear, get_tensor_parallel_group
from tests.common import _cleanup_process_group, _setup_process_group


WORLD_SIZE = 2


def test_tensor_parallel_group_interface_is_unimplemented():
    with pytest.raises(NotImplementedError):
        get_tensor_parallel_group()


def _copy_column_weight_shard(layer, full_weight, rank, world_size):
    with torch.no_grad():
        layer.linear.weight.copy_(full_weight.chunk(world_size, dim=0)[rank])


def _copy_row_weight_shard(layer, full_weight, rank, world_size):
    with torch.no_grad():
        layer.linear.weight.copy_(full_weight.chunk(world_size, dim=1)[rank])


def _column_parallel_worker(rank, world_size, gather_output):
    device = _setup_process_group(rank, world_size, backend="gloo")
    torch.manual_seed(123)

    in_features, out_features = 6, 10
    reference = nn.Linear(in_features, out_features, bias=False, device=device)
    layer = ColumnParallelLinear(in_features, out_features, gather_output=gather_output).to(device)
    _copy_column_weight_shard(layer, reference.weight, rank, world_size)

    x = torch.randn(3, 4, in_features, device=device).requires_grad_()
    reference_x = x.detach().clone().requires_grad_()
    expected_full = reference(reference_x)

    # Define one global reference loss. For gather_output=False, each rank
    # receives only its corresponding slice of this gradient. For
    # gather_output=True, every rank computes the replicated loss, so the TP
    # loss below is averaged across ranks.
    grad_output_full = torch.randn_like(expected_full)
    expected_loss = (expected_full * grad_output_full).sum()
    expected_loss.backward()
    actual = layer(x)

    if gather_output:
        assert actual.shape == expected_full.shape
        assert torch.allclose(actual, expected_full, atol=1e-5, rtol=1e-5)
    else:
        expected_local = expected_full.detach().chunk(world_size, dim=-1)[rank]
        assert actual.shape == expected_local.shape
        assert torch.allclose(actual, expected_local, atol=1e-5, rtol=1e-5)

    if gather_output:
        actual_loss = (actual * grad_output_full).sum() / world_size
    else:
        grad_output_local = grad_output_full.chunk(world_size, dim=-1)[rank]
        actual_loss = (actual * grad_output_local).sum()
    actual_loss.backward()

    expected_weight_grad = reference.weight.grad.chunk(world_size, dim=0)[rank]
    assert torch.allclose(x.grad, reference_x.grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(layer.linear.weight.grad, expected_weight_grad, atol=1e-5, rtol=1e-5)

    _cleanup_process_group()


def _row_parallel_worker(rank, world_size):
    device = _setup_process_group(rank, world_size, backend="gloo")
    torch.manual_seed(123)

    in_features, out_features = 10, 6
    reference = nn.Linear(in_features, out_features, bias=False, device=device)
    layer = RowParallelLinear(in_features, out_features, input_is_parallel=True).to(device)
    _copy_row_weight_shard(layer, reference.weight, rank, world_size)

    full_x = torch.randn(3, 4, in_features, device=device)
    reference_x = full_x.detach().clone().requires_grad_()
    local_x = full_x.chunk(world_size, dim=-1)[rank].detach().clone().requires_grad_()
    grad_output = torch.randn(3, 4, out_features, device=device)

    expected = reference(reference_x)
    (expected * grad_output).sum().backward()

    actual = layer(local_x)
    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    (actual * grad_output).sum().backward()

    expected_input_grad = reference_x.grad.chunk(world_size, dim=-1)[rank]
    expected_weight_grad = reference.weight.grad.chunk(world_size, dim=1)[rank]
    assert torch.allclose(local_x.grad, expected_input_grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(layer.linear.weight.grad, expected_weight_grad, atol=1e-5, rtol=1e-5)

    _cleanup_process_group()


@pytest.mark.parametrize("gather_output", [False, True])
def test_column_parallel_forward_backward_matches_reference(gather_output):
    mp.spawn(_column_parallel_worker, args=(WORLD_SIZE, gather_output), nprocs=WORLD_SIZE, join=True)


def test_row_parallel_forward_backward_matches_reference():
    mp.spawn(_row_parallel_worker, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)


def test_tensor_parallel_layers_reject_nondivisible_dimensions():
    with pytest.raises(ValueError):
        ColumnParallelLinear(6, 9)
    with pytest.raises(ValueError):
        RowParallelLinear(9, 6)
