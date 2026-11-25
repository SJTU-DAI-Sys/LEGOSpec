# Attention configuration

A layer selects its attention family and variant. Dense and latent layers also select a history-access pattern.

## Families

- `dense`: `mha`, `gqa`, or `mqa`, distinguished by query and KV head counts.
- `latent`: `mla`, with latent KV representations. Training expands K/V; evaluation without gradients uses absorbed MLA and a latent cache.
- `linear`: `gdn` or `kda`. GDN uses per-head decay; KDA uses per-key-channel decay within each head.

For example:

```json
{"attention": {"family": "dense", "variant": "gqa", "access": "swa", "window_size": 512}}
```

Shared projection settings belong in `attention_configs.<variant>`. Draft head layouts may differ from the target's layout. Explicit profiles control the draft projections; omitted geometry inherits target defaults. Query head count must be divisible by KV head count, and head dimension must be even.

## History access

Dense and latent attention support the combinations accepted by the recipe schema:

- `full`: all visible confirmed history.
- `swa`: a sliding window of confirmed history.
- `dsa`: history selected by an indexer.

Shared access settings belong in `access_configs.full`, `access_configs.swa`, or `access_configs.dsa`. For example:

```json
{
  "attention_configs": {
    "gqa": {"num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128}
  },
  "access_configs": {
    "swa": {"window_size": 512}
  }
}
```

DSA currently requires DSA in every layer, the SDPA backend, and a non-selector proposal. Absorbed MLA with DSA retains a latent cache but currently applies a visibility mask; this path does not imply sparse-compute acceleration.

## GDN/KDA block attention

Linear layers do not specify `access` or a separate block-attention switch:

```json
{"attention": {"family": "linear", "variant": "kda"}}
```

Every GDN/KDA layer, including the first, includes bidirectional attention within the current block. History and block reads share Q/K/V/O projections. A sigmoid gate per token and value head scales the block read before the shared output projection:

```text
normalized history read + gate × block read
    → shared output projection → residual → MLP
```

Set `attention_configs.gdn.gate_init` or `attention_configs.kda.gate_init` to control the initial gate value; the default is 0.1. Only confirmed context writes recurrent history. Block reads do not update that state.

Internal Q/KV depthwise convolutions are part of the linear mixer. An optional layer-level `local_operator` controls external grouped convolution separately. The supplied GDN/KDA five-layer recipes omit external convolution. FLA KDA supports key head dimensions up to 256 in the current integration.

## Choose an execution backend

`attention_backend` selects eager, SDPA, or FlexAttention execution for supported recipes. Backend settings belong to the training launcher rather than the architecture's family/access fields. GDN/KDA use the configured recurrent backend and its kernel dispatch.

See [runtime installation](SGLANG_RUNTIME.md), [recipes](../configs/README.md), and [model structure](../legospec/modeling/DESIGN.md).
