"""Unit tests and behavioral contracts for expert parallelism."""

import pytest
import torch

from cs336_systems.expert_parallel import ExpertParallelMoE, combine_expert_outputs, dispatch_tokens


def test_expert_parallel_interfaces_are_unimplemented():
    with pytest.raises(NotImplementedError):
        ExpertParallelMoE(num_experts=4, hidden_size=8, expert_size=16)
    x = torch.randn(4, 8)
    indices = torch.zeros(4, dtype=torch.long)
    with pytest.raises(NotImplementedError):
        dispatch_tokens(x, indices)
    with pytest.raises(NotImplementedError):
        combine_expert_outputs(x, torch.ones(4))


@pytest.mark.xfail(raises=NotImplementedError, strict=False, reason="Expert routing is a starter interface")
def test_expert_dispatch_and_combine_preserve_token_order():
    tokens = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    expert_indices = torch.tensor([2, 0, 1, 2, 1, 0])
    dispatched = dispatch_tokens(tokens, expert_indices)
    combined = combine_expert_outputs(dispatched, torch.ones(6))
    assert torch.equal(combined, tokens)
