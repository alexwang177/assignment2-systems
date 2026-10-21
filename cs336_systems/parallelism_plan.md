# Parallelism learning plan

This repository currently contains working DDP/FSDP exercises and starter
interfaces for the remaining topics in `AGENTS.md`.

## Suggested implementation order

1. Activation checkpointing
2. Tensor parallelism
3. Sequence parallelism
4. Pipeline parallelism
5. Context parallelism
6. Expert parallelism
7. N-dimensional composition

The starter tests in `tests/test_parallelism_boilerplate.py` intentionally test
only the interfaces and configuration validation. Replace each
`NotImplementedError` expectation with CPU/Gloo correctness tests as the
corresponding technique is implemented.

## CPU/MPS constraint

The exercises should use the Gloo backend for multi-process CPU tests. MPS is
useful for local tensor operations, but does not provide a general replacement
for CPU/Gloo process-group tests.
