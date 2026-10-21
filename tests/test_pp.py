"""Unit tests and behavioral contracts for pipeline parallelism."""

import pytest
import torch
from torch import nn

from cs336_systems.pipeline_parallel import PipelineSchedule, PipelineStage, partition_sequential_model


def test_pipeline_parallel_interfaces_are_unimplemented():
    stage = nn.Identity()
    schedule = PipelineSchedule([], num_microbatches=2)
    with pytest.raises(NotImplementedError):
        schedule.run(torch.randn(2, 4))
    with pytest.raises(NotImplementedError):
        partition_sequential_model(stage, num_stages=1)


def test_pipeline_stage_delegates_to_its_module():
    stage = PipelineStage(nn.Linear(4, 3), stage_index=0, num_stages=2)
    x = torch.randn(5, 4)
    assert stage(x).shape == (5, 3)
    assert stage.stage_index == 0
    assert stage.num_stages == 2


@pytest.mark.xfail(raises=NotImplementedError, strict=False, reason="Pipeline scheduling is a starter interface")
def test_pipeline_schedule_preserves_batch_shape():
    stage = PipelineStage(nn.Identity(), stage_index=0, num_stages=1)
    schedule = PipelineSchedule([stage], num_microbatches=2)
    x = torch.randn(4, 8)
    assert schedule.run(x).shape == x.shape
