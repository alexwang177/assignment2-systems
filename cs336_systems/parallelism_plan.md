# Parallelism learning plan

This repository currently contains working DDP/FSDP exercises and starter
interfaces for the remaining topics in `AGENTS.md`.

## Suggested implementation order

1. Activation checkpointing
2. Tensor parallelism
3. Sequence parallelism
4. Pipeline parallelism
5. Context parallelism
   - fixed-length all-gather attention
   - packed variable-length attention
   - ring attention with online softmax
   - Ulysses/all-to-all attention
   - hybrid ring/Ulysses attention
6. Expert parallelism
7. N-dimensional composition

The context-parallel tests in `tests/test_cp.py` are ordered as a learning
sequence. The initial tests validate configuration; later tests are expected
failures until each algorithm is implemented. Replace the expected-failure
markers with CPU/Gloo multi-process correctness tests as each stage is
completed. Keep variable-length attention separate from the communication
algorithm: it is a packed-input layout that can be combined with all-gather,
ring, or Ulysses attention.

## CPU/MPS constraint

The exercises should use the Gloo backend for multi-process CPU tests. MPS is
useful for local tensor operations, but does not provide a general replacement
for CPU/Gloo process-group tests.
