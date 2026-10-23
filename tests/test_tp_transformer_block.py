"""Forward test for tensor-parallel attention and FFN composition."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from einops import rearrange

from cs336_basics.model import RMSNorm, TransformerBlock, scaled_dot_product_attention
from cs336_systems.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from tests.common import _cleanup_process_group, _setup_process_group


WORLD_SIZE = 2


class TPAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert num_heads % WORLD_SIZE == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.local_num_heads = num_heads // WORLD_SIZE
        self.head_dim = d_model // num_heads
        self.q_proj = ColumnParallelLinear(d_model, d_model, gather_output=False)
        self.k_proj = ColumnParallelLinear(d_model, d_model, gather_output=False)
        self.v_proj = ColumnParallelLinear(d_model, d_model, gather_output=False)
        self.output_proj = RowParallelLinear(d_model, d_model, input_is_parallel=True)

    def forward(self, x):
        q = rearrange(self.q_proj(x), "... seq (heads d) -> ... heads seq d", heads=self.local_num_heads)
        k = rearrange(self.k_proj(x), "... seq (heads d) -> ... heads seq d", heads=self.local_num_heads)
        v = rearrange(self.v_proj(x), "... seq (heads d) -> ... heads seq d", heads=self.local_num_heads)

        seq_len = x.shape[-2]
        positions = torch.arange(seq_len, device=x.device)
        mask = positions[:, None] >= positions[None, :]
        mask = mask.reshape((1,) * (x.ndim - 2) + (seq_len, seq_len))
        local_attention = scaled_dot_product_attention(q, k, v, mask=mask)
        local_attention = rearrange(local_attention, "... heads seq d -> ... seq (heads d)")
        return self.output_proj(local_attention)


class TPTransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.attn = TPAttention(d_model, num_heads)
        self.ffn_w1 = ColumnParallelLinear(d_model, d_ff, gather_output=False)
        self.ffn_w3 = ColumnParallelLinear(d_model, d_ff, gather_output=False)
        self.ffn_w2 = RowParallelLinear(d_ff, d_model, input_is_parallel=True)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)

    def forward(self, x):
        x_attn = self.attn(self.ln1(x))
        attn_output = x + x_attn
        x_norm = self.ln2(attn_output)
        x_ffn = self.ffn_w2(torch.nn.functional.silu(self.ffn_w1(x_norm)) * self.ffn_w3(x_norm))
        return attn_output + x_ffn


def _copy_column(local_layer, reference_layer, rank):
    with torch.no_grad():
        local_layer.linear.weight.copy_(reference_layer.weight.chunk(WORLD_SIZE, dim=0)[rank])


def _copy_row(local_layer, reference_layer, rank):
    with torch.no_grad():
        local_layer.linear.weight.copy_(reference_layer.weight.chunk(WORLD_SIZE, dim=1)[rank])


def _copy_block_weights(tp_block, reference, rank):
    for tp_layer, ref_layer in (
        (tp_block.attn.q_proj, reference.attn.q_proj),
        (tp_block.attn.k_proj, reference.attn.k_proj),
        (tp_block.attn.v_proj, reference.attn.v_proj),
        (tp_block.ffn_w1, reference.ffn.w1),
        (tp_block.ffn_w3, reference.ffn.w3),
    ):
        _copy_column(tp_layer, ref_layer, rank)

    _copy_row(tp_block.attn.output_proj, reference.attn.output_proj, rank)
    _copy_row(tp_block.ffn_w2, reference.ffn.w2, rank)

    with torch.no_grad():
        tp_block.ln1.weight.copy_(reference.ln1.weight)
        tp_block.ln2.weight.copy_(reference.ln2.weight)


def _worker(rank, world_size):
    device = _setup_process_group(rank, world_size, backend="gloo")
    torch.manual_seed(42)

    d_model, num_heads, d_ff = 8, 2, 16
    reference = TransformerBlock(d_model, num_heads, d_ff, positional_encoder=None).to(device)
    tp_block = TPTransformerBlock(d_model, num_heads, d_ff).to(device)
    _copy_block_weights(tp_block, reference, rank)

    x = torch.randn(2, 4, d_model, device=device)
    expected = reference(x)
    actual = tp_block(x)

    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)

    _cleanup_process_group()


def test_tensor_parallel_transformer_block_forward():
    mp.spawn(_worker, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)
