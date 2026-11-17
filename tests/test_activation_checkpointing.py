"""Behavioral contracts for activation checkpointing variants."""

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import CheckpointPolicy

from cs336_systems.activation_checkpointing import FullAC, RegionAC, SelectiveAC, activation_memory_note


class BasicTransformerBlock(nn.Module):
    """Small CPU-friendly pre-norm Transformer-shaped block for the exercises."""

    def __init__(self, hidden_size: int = 8, expansion: int = 16):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.attention = nn.Linear(hidden_size, hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, expansion),
            nn.GELU(),
            nn.Linear(expansion, hidden_size),
        )

    def forward(self, x):
        x = x + self.attention(self.norm(x))
        return x + self.feed_forward(self.norm(x))


def _inputs():
    torch.manual_seed(0)
    return torch.randn(2, 4, 8, requires_grad=True)


def _policy(*_):
    """Starter selective policy: initially recompute every operation."""
    return CheckpointPolicy.PREFER_RECOMPUTE


def test_activation_memory_note_describes_the_tradeoff():
    note = activation_memory_note().lower()
    assert "activation" in note
    assert "recomput" in note


def test_wrappers_validate_region_bounds():
    with pytest.raises(ValueError):
        RegionAC(nn.Sequential(BasicTransformerBlock()), 1, 1)
    with pytest.raises(ValueError):
        RegionAC(nn.Sequential(BasicTransformerBlock()), -1, 1)


@pytest.mark.xfail(raises=NotImplementedError, strict=False, reason="FullAC is an exercise stub")
def test_full_ac_matches_forward_and_gradients():
    torch.manual_seed(0)
    reference = BasicTransformerBlock()
    checkpointed = FullAC(BasicTransformerBlock())
    checkpointed.module.load_state_dict(reference.state_dict())

    x_reference = _inputs()
    x_checkpointed = x_reference.detach().clone().requires_grad_()
    y_reference = reference(x_reference)
    y_checkpointed = checkpointed(x_checkpointed)
    y_reference.square().sum().backward()
    y_checkpointed.square().sum().backward()

    assert torch.allclose(y_checkpointed, y_reference)
    assert torch.allclose(x_checkpointed.grad, x_reference.grad)
    for actual, expected in zip(checkpointed.parameters(), reference.parameters()):
        assert torch.allclose(actual.grad, expected.grad)


@pytest.mark.xfail(raises=NotImplementedError, strict=False, reason="SelectiveAC is an exercise stub")
def test_selective_ac_preserves_forward_and_gradients():
    block = SelectiveAC(BasicTransformerBlock(), policy_fn=_policy)
    output = block(_inputs())
    output.square().sum().backward()
    assert output.shape == (2, 4, 8)


@pytest.mark.xfail(raises=NotImplementedError, strict=False, reason="RegionAC is an exercise stub")
def test_region_ac_checkpoints_only_a_transformer_stack_region():
    stack = nn.Sequential(*(BasicTransformerBlock() for _ in range(4)))
    model = RegionAC(stack, start=1, end=3)
    output = model(_inputs())
    output.square().sum().backward()
    assert output.shape == (2, 4, 8)
