"""Boilerplate for composing multiple parallelism dimensions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParallelismConfig:
    """Logical mesh sizes for an N-dimensional parallel training job."""

    data: int = 1
    tensor: int = 1
    sequence: int = 1
    context: int = 1
    expert: int = 1
    pipeline: int = 1

    @property
    def world_size(self) -> int:
        result = 1
        for size in (self.data, self.tensor, self.sequence, self.context, self.expert, self.pipeline):
            result *= size
        return result


def build_parallel_mesh(config: ParallelismConfig):
    """Construct the process mesh and named subgroups for the configuration."""
    raise NotImplementedError


def validate_parallelism_config(config: ParallelismConfig) -> None:
    """Validate that all mesh dimensions are positive and compatible."""
    if any(size < 1 for size in (config.data, config.tensor, config.sequence, config.context, config.expert, config.pipeline)):
        raise ValueError("All parallelism dimensions must be positive")


def train_nd_parallel(*args, **kwargs):
    """Boilerplate entry point for composing the parallelism strategies."""
    raise NotImplementedError
