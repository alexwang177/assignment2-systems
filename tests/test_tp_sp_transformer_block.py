"""A small forward-only example of composing TP and SP in one block.

This test intentionally keeps attention sequence-replicated: ordinary
self-attention needs access to every key/value position. Sequence parallelism
is used around the token-local RMSNorm operations, while tensor parallelism is
used for the attention and SwiGLU linear layers.
"""

from __future__ import annotations

import torch
import torch.multiprocessing as mp
from einops import rearrange
from torch import nn

from cs336_basics.model import RMSNorm, TransformerBlock, scaled_dot_product_attention
from cs336_systems.sequence_parallel import (
    SequenceParallelTensorParallelMLP,
    gather_sequence,
    scatter_sequence,
    sequence_parallel_layer,
)
from cs336_systems.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from tests.common import _cleanup_process_group, _setup_process_group


WORLD_SIZE = 2


class TPAttention(nn.Module):
    """TP over heads/features, with the full sequence replicated."""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert num_heads % WORLD_SIZE == 0
        self.local_num_heads = num_heads // WORLD_SIZE
        self.head_dim = d_model // num_heads

        # Each projection shards its output features.
        self.q_proj = ColumnParallelLinear(d_model, d_model, gather_output=False)
        self.k_proj = ColumnParallelLinear(d_model, d_model, gather_output=False)
        self.v_proj = ColumnParallelLinear(d_model, d_model, gather_output=False)

        # Each rank owns a partial input feature range; the output is summed.
        self.output_proj = RowParallelLinear(d_model, d_model, input_is_parallel=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = rearrange(
            self.q_proj(x), "... seq (heads d) -> ... heads seq d", heads=self.local_num_heads
        )
        k = rearrange(
            self.k_proj(x), "... seq (heads d) -> ... heads seq d", heads=self.local_num_heads
        )
        v = rearrange(
            self.v_proj(x), "... seq (heads d) -> ... heads seq d", heads=self.local_num_heads
        )

        sequence_length = x.shape[-2]
        positions = torch.arange(sequence_length, device=x.device)
        mask = positions[:, None] >= positions[None, :]
        mask = mask.reshape((1,) * (x.ndim - 2) + (sequence_length, sequence_length))

        local_attention = scaled_dot_product_attention(q, k, v, mask=mask)
        local_attention = rearrange(local_attention, "... heads seq d -> ... seq (heads d)")
        return self.output_proj(local_attention)


class TPSwiGLU(nn.Module):
    """Tensor-parallel SwiGLU: TP projections, local elementwise work."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = ColumnParallelLinear(d_model, d_ff, gather_output=False)
        self.w3 = ColumnParallelLinear(d_model, d_ff, gather_output=False)
        self.w2 = RowParallelLinear(d_ff, d_model, input_is_parallel=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # w1 and w3 each produce an intermediate shard on this rank.
        local_hidden = torch.nn.functional.silu(self.w1(x)) * self.w3(x)
        # w2 consumes the intermediate shard and all-reduces its output.
        return self.w2(local_hidden)


class TPSequenceParallelTransformerBlock(nn.Module):
    """Pre-norm block with SP around norms and TP in attention/FFN."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = TPAttention(d_model, num_heads)

        self.ffn = SequenceParallelTensorParallelMLP(
            local_module=RMSNorm(d_model),
            tp_mlp=TPSwiGLU(d_model, d_ff),
            sequence_dim=1,
            group=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SP around the first norm. Attention receives the full sequence.
        local_x = scatter_sequence(x, dim=1)
        local_x = sequence_parallel_layer(self.ln1, local_x, sequence_dim=1)
        normalized_x = gather_sequence(local_x, dim=1)

        x = x + self.attn(normalized_x)

        # SequenceParallelTensorParallelMLP performs:
        # scatter -> local RMSNorm -> gather -> TP SwiGLU.
        return x + self.ffn(x)


def _copy_column(local_layer, reference_layer, rank: int):
    with torch.no_grad():
        local_layer.linear.weight.copy_(reference_layer.weight.chunk(WORLD_SIZE, dim=0)[rank])


def _copy_row(local_layer, reference_layer, rank: int):
    with torch.no_grad():
        local_layer.linear.weight.copy_(reference_layer.weight.chunk(WORLD_SIZE, dim=1)[rank])


def _copy_weights(tp_block, reference_block, rank: int):
    for local_layer, reference_layer in (
        (tp_block.attn.q_proj, reference_block.attn.q_proj),
        (tp_block.attn.k_proj, reference_block.attn.k_proj),
        (tp_block.attn.v_proj, reference_block.attn.v_proj),
        (tp_block.ffn.tp_mlp.w1, reference_block.ffn.w1),
        (tp_block.ffn.tp_mlp.w3, reference_block.ffn.w3),
    ):
        _copy_column(local_layer, reference_layer, rank)

    for local_layer, reference_layer in (
        (tp_block.attn.output_proj, reference_block.attn.output_proj),
        (tp_block.ffn.tp_mlp.w2, reference_block.ffn.w2),
    ):
        _copy_row(local_layer, reference_layer, rank)

    with torch.no_grad():
        tp_block.ln1.weight.copy_(reference_block.ln1.weight)
        tp_block.ffn.local_module.weight.copy_(reference_block.ln2.weight)


def _worker(rank: int, world_size: int):
    device = _setup_process_group(rank, world_size, backend="gloo")
    torch.manual_seed(42)

    d_model, num_heads, d_ff = 8, 2, 16
    reference = TransformerBlock(d_model, num_heads, d_ff, positional_encoder=None).to(device)
    actual_block = TPSequenceParallelTransformerBlock(d_model, num_heads, d_ff).to(device)
    _copy_weights(actual_block, reference, rank)

    x = torch.randn(2, 8, d_model, device=device)
    expected = reference(x)
    actual = actual_block(x)

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    _cleanup_process_group()


def test_tensor_and_sequence_parallel_transformer_block_forward():
    mp.spawn(_worker, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)
