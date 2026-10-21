"""Boilerplate for expert parallelism in mixture-of-experts models."""

from __future__ import annotations

import torch
from torch import nn


class ExpertParallelMoE(nn.Module):
    """Mixture-of-experts layer with experts distributed across ranks."""

    def __init__(self, num_experts: int, hidden_size: int, expert_size: int):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.expert_size = expert_size
        raise NotImplementedError("Implement routing, token dispatch, and expert combine")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def dispatch_tokens(tokens: torch.Tensor, expert_indices: torch.Tensor, *, group=None):
    """Route tokens to the ranks owning their selected experts."""
    raise NotImplementedError


def combine_expert_outputs(outputs: torch.Tensor, routing_weights: torch.Tensor, *, group=None):
    """Return tokens to their original order and combine routed outputs."""
    raise NotImplementedError
