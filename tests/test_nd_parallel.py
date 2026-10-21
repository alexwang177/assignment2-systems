"""Unit tests and behavioral contracts for N-dimensional parallelism."""

import pytest

from cs336_systems.nd_parallel import ParallelismConfig, build_parallel_mesh, train_nd_parallel, validate_parallelism_config


def test_nd_parallel_config_and_interfaces():
    config = ParallelismConfig(data=2, tensor=2, pipeline=2)
    assert config.world_size == 8
    validate_parallelism_config(config)
    with pytest.raises(NotImplementedError):
        build_parallel_mesh(config)
    with pytest.raises(NotImplementedError):
        train_nd_parallel(config=config)


def test_nd_parallel_rejects_nonpositive_dimensions():
    with pytest.raises(ValueError):
        validate_parallelism_config(ParallelismConfig(data=0))


@pytest.mark.xfail(raises=NotImplementedError, strict=False, reason="N-D process mesh is a starter interface")
def test_nd_mesh_uses_configured_world_size():
    config = ParallelismConfig(data=2, tensor=2)
    mesh = build_parallel_mesh(config)
    assert mesh.world_size == config.world_size


@pytest.mark.xfail(raises=NotImplementedError, strict=False, reason="N-D training is a starter interface")
def test_nd_training_entry_point_returns_a_training_result():
    result = train_nd_parallel(config=ParallelismConfig(), steps=0)
    assert result is not None
