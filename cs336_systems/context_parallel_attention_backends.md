# Context-Parallel Attention Backends

Context parallelism and the attention backend are separate choices.

A context-parallel implementation decides **how attention work is distributed**
across ranks. The attention backend decides **how each rank computes attention**
and how its inputs and masks are represented.

A useful mental model is:

```text
CP communication strategy + attention backend + sequence representation
```

Examples:

```text
AllGatherKV + SDPA + fixed-length tensors
AllGatherKV + FlexAttention + fixed-length tensors
AllGatherKV + varlen attention
Ulysses + SDPA
Ulysses + FlexAttention
Ulysses + varlen attention
```

## 1. Standard fixed-length attention

The standard backend usually uses:

```python
torch.nn.functional.scaled_dot_product_attention(...)
```

The tensors are dense and padded to a common sequence length:

```text
Q: [batch, heads, query_length, head_dim]
K: [batch, heads, key_length, head_dim]
V: [batch, heads, key_length, head_dim]
```

The operation is:

```text
scores = Q @ K.T / sqrt(head_dim)
probabilities = softmax(scores + mask)
output = probabilities @ V
```

PyTorch may select different internal kernels, such as the math, memory-efficient,
or FlashAttention-style implementation. The public API remains SDPA.

This is the best reference backend for learning because it is simple and easy to
compare against the ordinary non-distributed implementation.

### CP usage

With all-gather context parallelism:

```text
Q_local: [batch, heads, local_sequence, head_dim]
K_full:  [batch, heads, global_sequence, head_dim]
V_full:  [batch, heads, global_sequence, head_dim]
```

Each rank computes only its local output rows.

## 2. FlexAttention

FlexAttention is PyTorch's programmable attention API:

```python
torch.nn.attention.flex_attention.flex_attention(...)
```

Compared with SDPA, FlexAttention allows more control over the attention pattern,
including:

- custom score modifications through `score_mod`;
- block-level masks through `BlockMask`;
- sliding-window attention;
- document boundaries;
- prefix-LM and other non-standard causal patterns;
- block-sparse attention patterns.

A dense SDPA mask has one entry for every query/key pair:

```text
[query_length, key_length]
```

A FlexAttention block mask describes which blocks can interact, allowing the
kernel to skip entire regions of work.

### CP usage

CP must construct a mask whose indices correctly map:

```text
local query positions -> global query positions
global key/value positions -> global key positions
```

This is particularly useful for complex CP layouts, permutations, sliding windows,
and variable global attention patterns.

## 3. Variable-length attention

Variable-length, or varlen, attention avoids padding every sequence to the same
length. Tokens are packed into one buffer:

```text
Q: [total_tokens, heads, head_dim]
K: [total_tokens, heads, head_dim]
V: [total_tokens, heads, head_dim]
```

Cumulative sequence lengths identify the individual sequences. For example:

```text
cu_seqlens = [0, 5, 12, 15]
```

means:

```text
sequence 0: tokens [0:5]
sequence 1: tokens [5:12]
sequence 2: tokens [12:15]
```

The attention kernel must prevent tokens from one sequence from attending to tokens
from another sequence.

Varlen attention is primarily an **input representation and kernel interface**,
not a CP communication topology. It can be combined with all-gather CP, Ulysses,
or other strategies.

### CP usage

A CP varlen implementation must preserve:

- sequence boundaries;
- local token ownership;
- global position IDs;
- causal relationships;
- per-sequence metadata after communication.

This is more complicated than fixed-length CP because a tensor split is no longer
necessarily aligned with a single dense sequence.

## TorchTitan naming convention

TorchTitan composes the CP strategy with the attention backend. Names generally
follow this pattern:

```text
[CP strategy][attention backend]InnerAttention
```

Examples:

### `AllGatherKVFlexInnerAttention`

```text
AllGatherKV CP + FlexAttention + fixed-length inputs
```

Each rank keeps local queries, all-gathers K/V, creates a rank-local/global-KV
FlexAttention mask, and computes local outputs.

### `AllGatherKVVarlenInnerAttention`

```text
AllGatherKV CP + variable-length packed attention
```

### `UlyssesInnerAttention`

```text
Ulysses CP + standard fixed-length inner attention
```

Ulysses redistributes the sequence/head layout, commonly transforming:

```text
[T / CP, H, ...] -> [T, H / CP, ...]
```

Attention then runs locally over the redistributed layout, after which the output
is redistributed back.

### `UlyssesFlexInnerAttention`

```text
Ulysses CP + FlexAttention
```

### `UlyssesVarlenInnerAttention`

```text
Ulysses CP + variable-length packed attention
```

## What does `InnerAttention` mean?

`InnerAttention` is the core attention operation after Q/K/V projection and before
the output projection:

```text
x
  -> q_proj, k_proj, v_proj
  -> InnerAttention(Q, K, V, metadata)
  -> output_proj
```

It generally does not own:

- Q/K/V projections;
- the output projection;
- residual connections;
- the MLP.

In this repository, the closest equivalent is:

```python
scaled_dot_product_attention(Q, K, V, mask)
```

## Important distinction

The suffixes describe different concerns:

| Name component | Meaning |
|---|---|
| `AllGatherKV` | K/V communication strategy |
| `Ulysses` | all-to-all sequence/head redistribution |
| `Flex` | programmable FlexAttention backend |
| `Varlen` | packed variable-length input representation |
| `InnerAttention` | attention core, after projections |

Therefore, a class without `Flex` or `Varlen`, such as
`UlyssesInnerAttention`, generally means standard fixed-length attention. The
exact kernel is determined by its base class, but it is conceptually the ordinary
SDPA-style path.

## Recommended learning order

For this repository, study the backends in this order:

1. Fixed-length SDPA all-gather CP.
2. Distributed backward for all-gather CP.
3. Fixed-length Ulysses with SDPA.
4. FlexAttention masks and block masks.
5. Varlen attention without CP.
6. Varlen all-gather CP.
7. Fixed-length ring attention.
8. Fixed-length Ulysses attention.
9. **Stretch goal:** varlen Ulysses CP.
10. **Stretch goal:** varlen ring CP.
