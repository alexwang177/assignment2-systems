# FSDP Implementation Plan

## Goal

Implement a simplified Fully-Sharded Data Parallel wrapper that:

- shards selected model parameters across ranks;
- all-gathers full parameters before computation;
- reduce-scatters gradients after backward;
- keeps master weights in FP32;
- optionally uses a lower-precision compute dtype;
- works with a standard PyTorch optimizer;
- supports CPU/Gloo execution for local testing.

The implementation should be compatible with the tests in `tests/test_fsdp.py`.

---

## Relevant Test Requirements

The tests use:

- `world_size = 2`;
- the `gloo` backend;
- CPU when CUDA is unavailable;
- a toy model containing:
  - `Embedding`;
  - `Linear`;
  - `RMSNorm`;
- `compute_dtype=None`;
- `compute_dtype=torch.float16`.

The tests check:

1.  FSDP training matches a non-parallel baseline.
2.  Parameters can be reconstructed after sharding.
3.  Gradients have the expected local shapes.
4.  Replicated parameters have synchronized gradients.
5.  FP32 master weights remain correct when computation uses FP16.

---

## Design Decisions

### Parameters to shard

Shard:

- `cs336_basics.model.Linear` weights;
- `cs336_basics.model.Embedding` weights.

Do not shard:

- `RMSNorm` parameters;
- parameters that are not explicitly supported by the implementation.

Replicated parameters must still have their gradients synchronized across ranks.

### Parameter ownership

For every sharded parameter, store metadata describing:

- original full shape;
- local shard shape;
- shard range or offset;
- owning rank;
- master parameter shard;
- temporary gathered full parameter;
- compute dtype, if applicable.

Every rank must derive the same metadata and use the same collective order.

---

## Suggested Internal Components

### FSDP wrapper

Responsibilities:

- store the wrapped module;
- discover supported layers;
- replace or wrap supported layers;
- expose local parameters to the standard optimizer;
- delegate arbitrary positional and keyword arguments in `forward`;
- track pending communication operations;
- implement `finish_gradient_synchronization`.

### Sharded layer wrapper

Responsibilities:

- store only the local master-weight shard persistently;
- gather full weights before forward computation;
- cast gathered weights to the compute dtype if requested;
- run the original layer operation;
- release temporary full weights when safe;
- arrange for backward-time weight availability;
- reduce-scatter gradients into local shard gradients.

### Replicated parameter handling

Responsibilities:

- keep replicated parameters on every rank;
- synchronize their gradients after backward;
- ensure all ranks have identical replicated parameter updates.

---

## Implementation Stages

### Stage 1: Process-group and wrapper setup

- Confirm that the process group is initialized.
- Record rank and world size.
- Store the wrapped module.
- Make `forward` support arbitrary positional and keyword arguments.
- Identify all supported `Linear` and `Embedding` modules.
- Identify replicated parameters such as norm parameters.

Sanity checks:

- every rank discovers the same module structure;
- every rank assigns the same sharding metadata;
- every rank uses the same parameter traversal order.

---

### Stage 2: Implement parameter sharding

For each supported sharded weight:

- determine the full parameter shape;
- choose a deterministic sharding layout;
- split the full parameter consistently across ranks;
- retain only the local master-weight shard;
- preserve the original full shape in metadata.

Verify:

- local shard sizes are correct;
- all ranks agree on shard offsets;
- concatenating or gathering shards reconstructs the original parameter;
- master shards remain FP32.

Use a tiny parameter and two ranks for debugging before testing the full model.

---

### Stage 3: Implement full-parameter gathering

Before a sharded layer executes:

- all ranks participate in an all-gather;
- the gathered shards reconstruct the full weight;
- the result is available in the expected shape;
- the gathered weight is cast to `compute_dtype` if requested.

Initially use synchronous communication for correctness. Add asynchronous communication only after the synchronous version is stable.

Sanity checks:

- compare gathered weights against an unsharded model;
- verify dtype and shape;
- verify all ranks reconstruct the same full weight.

---

### Stage 4: Implement forward computation

For each supported layer:

1.  gather the full weight;
2.  cast it for computation if necessary;
3.  run the original layer operation;
4.  release temporary gathered storage when it is no longer needed.

Do not replace the FP32 master shard with the lower-precision compute tensor permanently.

Verify:

- `compute_dtype=None` uses FP32 computation;
- `compute_dtype=torch.float16` uses FP16 computation where expected;
- master parameters remain FP32;
- forward outputs are numerically close to the baseline.

---

### Stage 5: Implement backward-time behavior

The backward pass must ensure that the relevant full weight is available when the layer’s backward computation needs it.

For the first correctness-oriented version:

- allow the autograd graph to retain the gathered weight as needed;
- avoid freeing tensors before backward uses them;
- prioritize correctness over memory efficiency.

Later, improve memory usage by freeing full gathered weights as soon as backward no longer needs them.

---

### Stage 6: Implement gradient reduce-scatter

For every sharded parameter gradient:

- all ranks participate in the same reduce-scatter operation;
- the result on each rank has the local shard shape;
- gradients are averaged consistently;
- the local gradient dtype matches the local master parameter dtype.

For replicated parameters:

- use an all-reduce;
- ensure every rank receives the same gradient.

Verify:

- sharded gradients have local shard shapes;
- replicated gradients are identical across ranks;
- gradients are available before `optimizer.step()`;
- gradients have the expected dtype.

---

### Stage 7: Implement synchronization finishing

Implement `finish_gradient_synchronization` so that it:

- waits for all outstanding asynchronous operations;
- ensures all local gradients are ready;
- clears completed communication handles;
- does not perform the optimizer step itself.

The training sequence should remain:

```text
forward
backward
finish_gradient_synchronization
optimizer.step
```

────────────────────────────────────────────────────────────────────────────────

### Stage 8: Implement full-parameter gathering for tests

Implement fsdp_gather_full_params so that it:

- reconstructs complete sharded parameters;
- returns replicated parameters directly;
- returns a mapping using the original parameter names;
- does not permanently change the sharded model layout;
- works with both FP32 and mixed-precision modes.

Use this function to compare FSDP parameters with the non-parallel baseline after every update.

────────────────────────────────────────────────────────────────────────────────

Mixed-Precision Requirements

When compute_dtype is provided:

- keep master parameter shards in FP32;
- cast weights before communication or computation as required;
- keep optimizer updates in FP32;
- ensure gradients are converted to the master-weight dtype before optimizer updates;
- use relaxed comparison tolerances for FP16 tests.

Do not permanently convert the master weights to FP16.

────────────────────────────────────────────────────────────────────────────────

Collective Communication Invariants

All ranks must:

- call the same collective operations;
- call them in the same order;
- use compatible shapes and dtypes;
- participate even when the current rank is not the source or owner;
- use matching metadata.

A rank-specific early return inside a collective sequence can cause a deadlock.

────────────────────────────────────────────────────────────────────────────────

Debugging Strategy

### Tiny single-layer test

Start with one supported Linear layer and two ranks.

Check:

- initial shard values;
- gathered full weight;
- local gradient shape;
- reduce-scattered gradient;
- optimizer update;
- reconstructed full parameter.

### Replicated parameter test

Use a small norm-like parameter and verify:

- each rank retains the full parameter;
- gradients are all-reduced;
- optimizer updates remain identical.

### Mixed-precision test

Verify separately:

- master weight dtype;
- gathered compute weight dtype;
- gradient dtype;
- optimizer state dtype;
- final parameter tolerance.

────────────────────────────────────────────────────────────────────────────────

Recommended Test Order

Run focused tests in this order:

1.  FSDP FP32 correctness.
2.  FSDP gradient synchronization.
3.  FSDP FP16 correctness.
4.  The full FSDP test file repeatedly.

Command:

```text
  uv run pytest tests/test_fsdp.py
```

Run it several times because asynchronous communication bugs may be intermittent.

────────────────────────────────────────────────────────────────────────────────

Mac-Specific Notes

The repository tests use Gloo and CPU fallback, so the correctness tests should run on a Mac.

However:

- CPU/Gloo will not provide meaningful GPU overlap measurements;
- MPS is not a substitute for CUDA/NCCL;
- use small toy models while debugging;
- prioritize shape, dtype, collective-order, and parameter-equivalence checks.

────────────────────────────────────────────────────────────────────────────────

Final Completion Checklist

- [ ] Wrapper delegates arbitrary forward arguments.
- [ ] Supported layers are discovered consistently on every rank.
- [ ] Sharded master weights remain FP32.
- [ ] Full weights are gathered before computation.
- [ ] Temporary gathered weights are released appropriately.
- [ ] Sharded gradients are reduce-scattered.
- [ ] Replicated gradients are all-reduced.
- [ ] Mixed-precision computation works.
- [ ] Standard optimizers can consume local parameters.
- [ ] Full parameters can be reconstructed for comparison.
- [ ] Communication handles are awaited and cleared.
- [ ] All FSDP tests pass repeatedly.
