"""Boilerplate for pipeline parallelism."""

from __future__ import annotations

import torch
from torch import nn


class PipelineStage(nn.Module):
    """One stage of a model partitioned across pipeline ranks."""

    def __init__(self, module: nn.Module, stage_index: int, num_stages: int):
        super().__init__()
        self.module = module
        self.stage_index = stage_index
        self.num_stages = num_stages

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


class PipelineSchedule:
    """Boilerplate interface for a pipeline microbatch schedule."""

    def __init__(self, stages: list[PipelineStage], num_microbatches: int):
        self.stages = stages
        self.num_microbatches = num_microbatches

    def run(self, batch: torch.Tensor):
        raise NotImplementedError("Implement pipeline send/receive and 1F1B scheduling")


def partition_sequential_model(module: nn.Module, num_stages: int) -> list[PipelineStage]:
    """Partition a sequential model into pipeline stages."""
    raise NotImplementedError
