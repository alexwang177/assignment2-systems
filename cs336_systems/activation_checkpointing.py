"""Boilerplate for activation checkpointing exercises."""

from __future__ import annotations

from collections.abc import Callable

import torch


def checkpoint(function: Callable, *args, use_reentrant: bool = False, **kwargs):
    """Run ``function`` with activation checkpointing.

    The implementation should use ``torch.utils.checkpoint`` and document the
    tradeoff between lower activation memory and recomputation during backward.
    """
    raise NotImplementedError


def checkpoint_sequential(functions, segments: int, input: torch.Tensor, **kwargs):
    """Checkpoint a sequence of functions in ``segments`` chunks."""
    raise NotImplementedError


def activation_memory_note() -> str:
    """Return a short description of the memory/compute tradeoff."""
    return "Checkpointing saves fewer activations and recomputes forward work during backward."
