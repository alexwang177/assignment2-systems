"""Activation-checkpointing exercises for small Transformer modules.

The public wrappers below are intentionally incomplete.  Implement them using
``torch.utils.checkpoint`` while keeping the module APIs unchanged.

Variants used in this exercise:

* :class:`FullAC`: checkpoint the complete module invocation.
* :class:`SelectiveAC`: checkpoint the invocation with a selective-checkpoint
  policy, allowing chosen operations to be saved instead of recomputed.
* :class:`RegionAC`: checkpoint only a contiguous region of an
  ``nn.Sequential`` stack; layers outside that region run normally.

All variants should preserve forward values and parameter/input gradients.
They trade activation memory for extra forward computation during backward.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import functools

import torch
from torch import nn
import torch.utils.checkpoint as checkpoint_utils


class FullAC(nn.Module):
    """Apply activation checkpointing to an entire module.

    TODO: call the non-reentrant PyTorch checkpoint implementation from
    ``forward``.  Preserve arbitrary positional and keyword arguments.
    """

    def __init__(self, module: nn.Module, *, use_reentrant: bool = False):
        super().__init__()
        self.module = module
        self.use_reentrant = use_reentrant

    def forward(self, *args, **kwargs):
        return checkpoint(
            self.module,
            *args,
            use_reentrant=self.use_reentrant,
            **kwargs,
        )


class SelectiveAC(nn.Module):
    """Checkpoint a module using a selective operation policy.

    ``policy_fn`` is intended for
    ``torch.utils.checkpoint.create_selective_checkpoint_contexts``.  The
    policy decides which operations are saved and which are recomputed.
    """

    def __init__(self, module: nn.Module, policy_fn: Callable, *, use_reentrant: bool = False):
        super().__init__()
        self.module = module
        self.policy_fn = policy_fn
        self.use_reentrant = use_reentrant

    def forward(self, *args, **kwargs):
        return checkpoint(
            self.module,
            *args,
            use_reentrant=self.use_reentrant,
            context_fn=functools.partial(
                checkpoint_utils.create_selective_checkpoint_contexts,
                self.policy_fn,
            ),
            **kwargs,
        )


class RegionAC(nn.Module):
    """Checkpoint selected layers in a sequential Transformer stack.

    The half-open interval ``[start, end)`` identifies the checkpointed
    region.  ``module`` must be an ``nn.Sequential`` (or another module with
    indexed children); layers before and after the region are not wrapped.
    """

    def __init__(self, module: nn.Sequential, start: int, end: int, *, use_reentrant: bool = False):
        super().__init__()
        if start < 0 or end <= start:
            raise ValueError("RegionAC requires 0 <= start < end")
        self.module = module
        self.start = start
        self.end = end
        self.use_reentrant = use_reentrant

    def forward(self, *args, **kwargs):
        
        x = args[0]
        for i in range(len(self.module)):
            if i >= self.start and i < self.end:
                x = checkpoint(
                    self.module[i],
                    x,
                    use_reentrant=self.use_reentrant,
                    **kwargs,
                )
            else:
                x = self.module[i](x, **kwargs)

        return x



def checkpoint(function: Callable, *args, use_reentrant: bool = False, **kwargs):
    """Low-level checkpointing helper for one callable.

    This is useful for experimenting before integrating one of the module
    wrappers above.  ``kwargs`` should be forwarded to the callable.
    """
    return checkpoint_utils.checkpoint(
        function,
        *args,
        use_reentrant=use_reentrant,
        **kwargs,
    )


def checkpoint_sequential(functions: Sequence[Callable], segments: int, input: torch.Tensor, **kwargs):
    """Low-level helper for checkpointing a sequence in ``segments`` chunks."""
    return checkpoint_utils.checkpoint_sequential(
        functions,
        segments,
        input,
        **kwargs,
    )


def activation_memory_note() -> str:
    """Return the central activation-checkpointing tradeoff."""
    return "Checkpointing saves fewer activations and recomputes forward work during backward."
