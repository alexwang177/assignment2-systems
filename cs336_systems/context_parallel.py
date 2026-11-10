"""Learning scaffolds for context-parallel attention.

This module intentionally does not implement the algorithms yet.  The intended
learning order is:

1. all-gather attention;
2. variable-length all-gather attention;
3. ring attention;
4. Ulysses (all-to-all) attention;
5. hybrid ring/Ulysses attention.

Stretch goals:

6. variable-length ring attention;
7. variable-length Ulysses attention.

Tensor layout for fixed-length attention is ``[..., heads, sequence, head_dim]``.
The sequence dimension is partitioned across a context-parallel process group.
For varlen attention, packed Q/K/V tensors use ``[tokens, heads, head_dim]`` and
``cu_seqlens`` describes the individual sequences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

ContextParallelMode = Literal["all_gather", "ring", "ulysses", "hybrid"]
SUPPORTED_CONTEXT_PARALLEL_MODES = ("all_gather", "ring", "ulysses", "hybrid")


class AllGatherWithReduceScatter(torch.autograd.Function):
    """Boilerplate for an autograd-aware all-gather.

    Forward contract:
        local ``x`` -> the rank-ordered concatenation of every rank's ``x``.

    Backward contract:
        Split ``grad_output`` into rank-ordered chunks, sum corresponding
        chunks across the process group with reduce-scatter, and return only
        this rank's local gradient.

    This is the key gradient rule for gathered K/V: every rank contributes to
    the gradient of every K/V partition.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, group=None, dim: int = -2):
        world_size = dist.get_world_size(group)
        ctx.world_size = world_size
        ctx.dim = dim
        ctx.group = group

        lst = [torch.zeros_like(x) for _ in range(world_size)]
        dist.all_gather(lst, x, group=group)

        return torch.concat(lst, dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        lst = list(torch.chunk(grad_output, chunks=ctx.world_size, dim=ctx.dim))
        lst = [chunk.contiguous() for chunk in lst]
        local_grad = torch.zeros_like(lst[0])
        dist.reduce_scatter(local_grad, lst, group=ctx.group)

        return (local_grad, None, None)


def all_gather_autograd(x: torch.Tensor, *, group=None, dim: int = -2) -> torch.Tensor:
    """Apply the custom autograd all-gather once its Function is implemented."""
    return AllGatherWithReduceScatter.apply(x, group, dim)


@dataclass(frozen=True)
class ContextParallelConfig:
    """Configuration shared by the CP implementations.

    ``causal`` refers to global sequence positions, not positions local to a
    rank.  Implementations must therefore accept or derive a global offset for
    each local query/key block.
    """

    mode: ContextParallelMode = "all_gather"
    causal: bool = True

    def __post_init__(self) -> None:
        if self.mode not in SUPPORTED_CONTEXT_PARALLEL_MODES:
            raise ValueError(f"Unknown context-parallel mode: {self.mode}")


class ContextParallelAttention(nn.Module):
    """Dispatch point for the CP attention algorithms.

    This is deliberately only an interface.  Keep the algorithm-specific
    functions below separate while learning; this wrapper can be completed
    after their individual correctness tests pass.
    """

    def __init__(
        self,
        mode: ContextParallelMode = "all_gather",
        *,
        group=None,
        causal: bool = True,
    ):
        super().__init__()
        self.config = ContextParallelConfig(mode=mode, causal=causal)
        self.group = group

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        query_start: int = 0,
        key_start: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        """Compute CP attention for local Q/K/V blocks.

        ``query_start`` and ``key_start`` are global sequence offsets.  They
        are required for correct causal masking when blocks are distributed.
        """
        raise NotImplementedError("Implement the selected context-parallel attention algorithm")


def all_gather_attention(
    query: torch.Tensor, # [batch, heads, local_sequence, head_dim]
    key: torch.Tensor,   # [batch, heads, local_sequence, head_dim]
    value: torch.Tensor, # [batch, heads, local_sequence, head_dim]
    *,
    group=None,
    causal: bool = True,
    key_start: int = 0,
) -> torch.Tensor:
    """Stage 1: gather K/V, then compute attention for local queries."""

    if not dist.is_initialized():
        rank = 0
        world_size = 1
    else:
        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)

    local_seq_len = query.shape[-2]
    full_seq_len = local_seq_len * world_size

    query_start = rank * local_seq_len

    query_positions = query_start + torch.arange(start=0, end=local_seq_len, device=query.device)
    key_positions = key_start + torch.arange(start=0, end=full_seq_len, device=query.device)

    causal_mask = None
    if causal:
        causal_mask = query_positions[:, None] >= key_positions[None, :]
        causal_mask = causal_mask[None, None, :, :]

    if world_size > 1:
        # All gather the local K tensors and concatenate along sequence dimension
        full_key = all_gather_autograd(key, group=group, dim=-2)
        # All gather the local V tensors and concatenate along sequence dimension
        full_value = all_gather_autograd(value, group=group, dim=-2)
    else:
        full_key = key
        full_value = value

    return F.scaled_dot_product_attention(
        query=query,
        key=full_key,
        value=full_value,
        attn_mask=causal_mask
    )


def _local_varlen_attention(
    query: torch.Tensor,  # [total_tokens, heads, head_dim]
    key: torch.Tensor,    # [total_tokens, heads, head_dim]
    value: torch.Tensor,  # [total_tokens, heads, head_dim]
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    query_position_starts: list[int],
    causal=False,
):
    num_seq = len(cu_seqlens_q) - 1
    local_outputs = []

    for i in range(num_seq):
        q_start = cu_seqlens_q[i].item()
        q_end = cu_seqlens_q[i + 1].item()

        k_start = cu_seqlens_k[i].item()
        k_end = cu_seqlens_k[i + 1].item()

        q_i = query[q_start:q_end] # [T, H, D]
        k_i = key[k_start:k_end]   # [T, H, D]
        v_i = value[k_start:k_end] # [T, H, D]

        q_i = q_i[None, :, :, :]  # [1, T, H, D]
        q_i = q_i.transpose(1, 2) # [1, H, T, D]

        k_i = k_i[None, :, :, :]  # [1, T, H, D]
        k_i = k_i.transpose(1, 2) # [1, H, T, D]

        v_i = v_i[None, :, :, :]  # [1, T, H, D]
        v_i = v_i.transpose(1, 2) # [1, H, T, D]

        # Create the mask
        causal_mask = None
        if causal:
            q_positions = query_position_starts[i] + torch.arange(q_end - q_start)
            k_positions = torch.arange(k_end - k_start)
            causal_mask = q_positions[:, None] >= k_positions[None, :]

        o_i = F.scaled_dot_product_attention(
            query=q_i,
            key=k_i,
            value=v_i,
            is_causal=False,
            attn_mask=causal_mask
        ) # [1, H, T, D]

        o_i = o_i.squeeze(0)      # [H, T, D]
        o_i = o_i.transpose(0, 1) # [T, H, D]
        local_outputs.append(o_i)

    packed_out = torch.concat(local_outputs, dim=0)
    return packed_out


def _repack_rank_major(
    x, # [tokens, num_heads, head_dim]
    local_cu_seqlens: torch.Tensor, # [num_sequences + 1]
    world_size: int,
) -> torch.Tensor:
    local_total_tokens = local_cu_seqlens[-1].item()
    rank_blocks = torch.split(x, split_size_or_sections=local_total_tokens, dim=0)

    seq_blocks = []
    num_seq = len(local_cu_seqlens) - 1

    for i in range(num_seq):
        start = local_cu_seqlens[i].item()
        end = local_cu_seqlens[i + 1].item()

        chunks = []

        for rank_block in rank_blocks:
            chunks.append(rank_block[start:end])

        full_seq = torch.concat(chunks, dim=0)
        seq_blocks.append(full_seq)

    return torch.concat(seq_blocks, dim=0)

def varlen_all_gather_attention(
    query: torch.Tensor,        # [total_local_query_tokens, num_heads, head_dim]
    key: torch.Tensor,          # [total_local_kv_tokens, num_heads, head_dim]
    value: torch.Tensor,        # [total_local_kv_tokens, num_heads, value_dim]
    cu_seqlens_q: torch.Tensor, # [num_sequences + 1]
    cu_seqlens_k: torch.Tensor, # [num_sequences + 1]
    *,
    group=None,
    causal: bool = True,
) -> torch.Tensor:
    """Stage 2: all-gather CP for packed variable-length sequences.

    Example with two logical sequences and two CP ranks:

        local Q/K/V lengths per sequence: [3, 2]
        local cu_seqlens:                 [0, 3, 5]
        global K/V lengths:               [6, 4]
        global cu_seqlens_k:              [0, 6, 10]

        rank 0 query offsets:              [0, 0]
        rank 1 query offsets:              [3, 2]
    """

    # Shape: [num_sequences + 1]. For example, [0, 3, 5] describes
    # sequence 0 as [0:3] and sequence 1 as [3:5].
    assert len(cu_seqlens_k) == len(cu_seqlens_q)
    num_seq = len(cu_seqlens_q) - 1  # scalar: number of logical sequences

    # Shape: [num_sequences]. Example: diff([0, 3, 5]) -> [3, 2].
    local_q_lengths = torch.diff(cu_seqlens_q)
    local_k_lengths = torch.diff(cu_seqlens_k)

    # Initial self-attention contract: local Q and local K/V lengths match.
    assert len(local_k_lengths) == len(local_q_lengths)

    # Scalars describing the active CP process group.
    world_size = dist.get_world_size(group)  # example: 2
    rank = dist.get_rank(group)              # example: 0 or 1

    # Shape: [num_sequences]. Example: [3, 2] * 2 -> [6, 4].
    global_k_lengths = world_size * local_k_lengths

    # Shape: [num_sequences]. Global position of each rank's first local
    # query token. Example for rank 1: [3, 2].
    query_position_starts = rank * local_q_lengths

    # Shape: [num_sequences + 1]. Example: [6, 4] -> [0, 6, 10].
    global_cu_seqlens_k = torch.cat(
        (
            global_k_lengths.new_zeros(1),
            torch.cumsum(global_k_lengths, dim=0),
        )
    )

    # Gather KV tensors
    global_key = all_gather_autograd(key, group=group, dim=0) # [total_global_key_tokens, num_heads, head_dim]
    global_key = _repack_rank_major(global_key, cu_seqlens_k, world_size)

    global_value = all_gather_autograd(value, group=group, dim=0) # [total_global_value_tokens, num_heads, head_dim]
    global_value = _repack_rank_major(global_value, cu_seqlens_k, world_size)

    # query: [total_local_query_tokens, heads, head_dim]
    # key/value: [total_global_kv_tokens, heads, head_dim] after gathering
    # cu_seqlens_q: [num_sequences + 1], local Q boundaries
    # global_cu_seqlens_k: [num_sequences + 1], global K/V boundaries
    return _local_varlen_attention(
        query,
        global_key,
        global_value,
        cu_seqlens_q,
        global_cu_seqlens_k,
        causal=causal,
        query_position_starts=query_position_starts.tolist()
    )


def exchange_context_partition(x: torch.Tensor, *, group=None):
    """Exchange one local K/V context partition with a ring neighbor.

    This is the communication primitive needed by ring attention.  Decide and
    document the ring direction, ownership order, and asynchronous work-handle
    contract before implementing it.
    """
    return _exchange_kv_block(x, group)


def _exchange_kv_block(x, group=None):
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    send_to = (rank + 1) % world_size
    receive_from = (rank - 1) % world_size

    recv_buffer = torch.empty_like(x)

    send_op = dist.P2POp(dist.isend, x, send_to, group)
    recv_op = dist.P2POp(dist.irecv, recv_buffer, receive_from, group)

    work_handles = dist.batch_isend_irecv([send_op, recv_op])
    for h in work_handles:
        h.wait()

    return recv_buffer
    
"""

 Ring loop state

 For each rank, initialize:

 ```text
   running_max: [batch, heads, local_query_length]
   running_sum: [batch, heads, local_query_length]
   running_output: [batch, heads, local_query_length, value_dim]
 ```

 Initialize:

 ```text
   running_max = -inf
   running_sum = 0
   running_output = 0
 ```

 For each ring step

 With the current K/V block:

 1. Compute scores:
    ```text
      scores = Q @ K_currentᵀ / sqrt(head_dim)
    ```

 2. Compute block statistics:
    ```text
      block_max = max(scores, dim=key_dimension)

      exp_scores = exp(scores - block_max[..., None])

      block_sum = sum(exp_scores, dim=key_dimension)

      block_output = exp_scores @ V_current
    ```

 3. Merge the block into the running state:
    ```text
      new_max = max(running_max, block_max)

      old_scale = exp(running_max - new_max)
      block_scale = exp(block_max - new_max)

      running_sum =
          old_scale * running_sum
          + block_scale * block_sum

      running_output =
          old_scale[..., None] * running_output
          + block_scale[..., None] * block_output

      running_max = new_max
    ```

 4. Exchange K/V for the next iteration.

 Do not exchange after the final iteration.

 5. Normalize at the end:

 ```text
   output = running_output / running_sum[..., None]
 ```

"""

def ring_attention(
    query: torch.Tensor, # [batch, heads, local_queries, head_dim]
    key: torch.Tensor,   # [batch, heads, local_keys, head_dim]
    value: torch.Tensor, # [batch, heads, local_values, head_dim]
    *,
    group=None,
    causal: bool = True,
    query_start: int = 0,
    key_start: int = 0,
) -> torch.Tensor:
    """Stage 3: circulate K/V blocks and combine them with online softmax."""
    batch, heads, local_sequence_len, head_dim = query.shape
    value_dim = value.shape[-1]
    
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    running_max = torch.full((batch, heads, local_sequence_len,), float("-inf"))    # [batch, heads, local_queries]
    running_denom = torch.zeros((batch, heads, local_sequence_len,))                # [batch, heads, local_queries]
    running_output = torch.zeros((batch, heads, local_sequence_len, value_dim,))    # [batch, heads, local_queries, value_dim]

    for i in range(world_size):
        
        # 1. Compute scores
        key_T = key.transpose(-2, -1) # [batch, heads, head_dim, local_keys]
        attn_scores = (query @ key_T) / head_dim ** 0.5 # [batch, heads, local_queries, local_keys]

        # 2. Compute statistics for the current KV block
        block_max = torch.amax(attn_scores, dim=-1) # [batch, heads, local_queries]
        exp_attn_scores = torch.exp(attn_scores - block_max[..., None]) # [batch, heads, local_sequence, local_sequence]
        block_denom = torch.sum(exp_attn_scores, dim=-1) # [batch, heads, local_queries]
        block_output = exp_attn_scores @ value # [batch, heads, local_values, head_dim]

        # 3. Merge the block into the running state
        new_max = torch.maximum(running_max, block_max) # [batch, heads, local_queries]

        old_scale = torch.exp(running_max - new_max)
        block_scale = torch.exp(block_max - new_max)

        running_denom = old_scale * running_denom + block_scale * block_denom
        running_output = old_scale[..., None] * running_output + block_scale[..., None] * block_output

        running_max = new_max

        if i < world_size - 1:
            # Exchange KV blocks
            key = _exchange_kv_block(key, group=group)
            value = _exchange_kv_block(value, group=group)

    output = running_output / running_denom[..., None]
    return output


def ulysses_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    group=None,
    causal: bool = True,
) -> torch.Tensor:
    """Stage 4: redistribute sequence/head layout with all-to-all."""
    raise NotImplementedError


def hybrid_ring_ulysses_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    ring_group=None,
    ulysses_group=None,
    causal: bool = True,
) -> torch.Tensor:
    """Stage 5: compose Ulysses redistribution with ring attention."""
    raise NotImplementedError


def varlen_ring_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    group=None,
    causal: bool = True,
) -> torch.Tensor:
    """Stretch goal: ring attention for packed variable-length sequences."""
    raise NotImplementedError


def varlen_ulysses_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    group=None,
    causal: bool = True,
) -> torch.Tensor:
    """Stretch goal: Ulysses attention for packed variable-length sequences."""
    raise NotImplementedError
