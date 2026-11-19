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

The ``toy_nonreentrant_checkpoint`` exercise below is intentionally narrower
than PyTorch's implementation.  It is for studying the mechanism, not for
production use.
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

class Handle:
    def __init__(self, slot):
        self.slot = slot


def toy_nonreentrant_checkpoint(function: Callable, *args, **kwargs):
    """Educational scaffold for a small non-reentrant checkpoint prototype.

    Suggested implementation steps:

    1. Validate the deliberately small scope: tensor positional arguments, one
       tensor output, no in-place mutation, and a deterministic ``function``.
    2. Save detached copies of the input tensors, plus enough metadata to
       reconstruct the call during backward.  Do not save the intermediate
       activations produced inside ``function``.
    3. Create a recomputation closure that reruns ``function`` from those saved
       inputs with gradients enabled.  Guard it against recursively applying
       the same checkpoint hooks.
    4. Use ``torch.autograd.graph.saved_tensors_hooks`` around the original
       forward.  Its pack hook should replace tensors saved by autograd with
       lightweight handles; its unpack hook should trigger recomputation and
       return the requested reconstructed tensor.
    5. Make sure recomputation builds a connected autograd graph.  Merely
       running ``function`` under ``no_grad`` will not work for this exercise.
    6. Preserve and restore RNG state if you decide to support random
       operations such as dropout; otherwise document determinism as a
       limitation.
    7. Compare output values and input/parameter gradients against an
       uncheckpointed call and against
       ``torch.utils.checkpoint.checkpoint(..., use_reentrant=False)``.

    This is intentionally much smaller than PyTorch's implementation.  Do not
    try to reproduce its nested structures, early-stop recomputation, device
    handling, mutation checks, or selective policies initially.
    """
    for x in args:
        assert isinstance(x, torch.Tensor)

    saved_inputs = tuple(x.detach().clone() for x in args)
    requires_grad = tuple(x.requires_grad for x in args)

    saved_handles = []
    recomputed_tensors = []

    def recompute_pack(x: torch.Tensor):
        recomputed_tensors.append(x)
        return x

    def recompute_unpack(value):
        return value

    def recompute():
        recompute_inputs = tuple(
            x.detach().clone().requires_grad_(needs_grad)
            for x, needs_grad in zip(saved_inputs, requires_grad)
        )

        with torch.enable_grad():
            with torch.autograd.graph.saved_tensors_hooks(recompute_pack, recompute_unpack):
                output = function(*recompute_inputs, **kwargs)

        return output   
    
    def pack(x: torch.Tensor):
        handle = Handle(len(saved_handles))
        saved_handles.append(handle)
        return handle

    def unpack(handle):
        if not recomputed_tensors:
            recompute()
        
        return recomputed_tensors[handle.slot]
    
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        output = function(*args, **kwargs)
    return output


def activation_memory_note() -> str:
    """Return the central activation-checkpointing tradeoff."""
    return "Checkpointing saves fewer activations and recomputes forward work during backward."
