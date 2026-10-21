"""Unit tests and behavioral contracts for activation checkpointing."""

import pytest
import torch

from cs336_systems.activation_checkpointing import activation_memory_note, checkpoint, checkpoint_sequential


def test_activation_checkpointing_interfaces_are_unimplemented():
    x = torch.randn(2, 4, requires_grad=True)
    with pytest.raises(NotImplementedError):
        checkpoint(lambda value: value.square(), x)
    with pytest.raises(NotImplementedError):
        checkpoint_sequential([lambda value: value + 1], segments=1, input=x)


def test_activation_memory_note_describes_the_tradeoff():
    note = activation_memory_note()
    assert "activation" in note.lower()
    assert "recomput" in note.lower()


@pytest.mark.xfail(raises=NotImplementedError, strict=False, reason="Activation checkpointing is a starter interface")
def test_checkpoint_matches_uncheckpointed_forward_and_gradient():
    torch.manual_seed(0)
    weight = torch.randn(4, 4, requires_grad=True)
    x = torch.randn(2, 4, requires_grad=True)

    checkpointed = checkpoint(lambda value: value @ weight, x)
    loss = checkpointed.square().sum()
    loss.backward()

    assert checkpointed.shape == (2, 4)
    assert x.grad is not None
    assert weight.grad is not None


@pytest.mark.xfail(raises=NotImplementedError, strict=False, reason="Sequential checkpointing is a starter interface")
def test_checkpoint_sequential_preserves_function_composition():
    x = torch.randn(2, 4, requires_grad=True)
    functions = [lambda value: value + 1, torch.relu, lambda value: value * 2]
    actual = checkpoint_sequential(functions, segments=2, input=x)
    expected = functions[2](functions[1](functions[0](x)))
    assert torch.allclose(actual, expected)
