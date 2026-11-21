"""Frozen Qwen3 scoring with one prefix KV per document, shared by all anchors.

No branch is appended to the prefix cache. Prefixes use the target model's
standard causal forward. Branches use the project's guarded FlexAttention path
on CUDA and a dense SDPA numerical oracle on CPU.
"""

from dataclasses import dataclass, field

import torch
from torch.nn import functional as F
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb


@dataclass
class AnchorPrefix:
    key_values: list
    valid: torch.Tensor
    anchors: torch.Tensor
    keep: torch.Tensor
    anchor_hidden: torch.Tensor
    context_hidden: dict = field(default_factory=dict)

    def release(self):
        self.key_values.clear()
        self.context_hidden.clear()


def _attend(q, k, v, mask_mod, scale):
    if q.is_cuda:
        from litespec.modeling.draft.flex_attention import (
            compile_friendly_flex_attention, create_flex_block_mask, flex_kernel_options,
        )
        mask = create_flex_block_mask(mask_mod, B=q.size(0), H=None,
            Q_LEN=q.size(-2), KV_LEN=k.size(-2), device=q.device, backend="TRITON")
        return compile_friendly_flex_attention(q, k, v, block_mask=mask,
            enable_gqa=q.size(1) != k.size(1), scale=scale,
            kernel_options=flex_kernel_options("TRITON"))
    from torch.nn.attention.flex_attention import create_mask

    mask = create_mask(mask_mod, q.size(0), 1, q.size(-2), k.size(-2), device=q.device)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                         enable_gqa=q.size(1) != k.size(1), scale=scale)


class Qwen3AnchorScorer:
    """Plain service object: target parameters never become FSDP children."""

    def __init__(self, causal_lm):
        config = causal_lm.config
        if config.model_type != "qwen3" or any(
            kind != "full_attention" for kind in (getattr(config, "layer_types", None) or [])
        ):
            raise ValueError("Anchor OPD supports dense, full-attention Qwen3 targets only")
        if getattr(config, "rope_scaling", None) or (
            (getattr(config, "rope_parameters", None) or {}).get("rope_type", "default") != "default"
        ):
            raise ValueError("Anchor OPD currently requires standard Qwen3 RoPE")
        self.model = causal_lm.requires_grad_(False).eval()
        self.prefill_calls = 0
        self.branch_calls = 0

    def _qkv(self, attention, hidden, positions):
        shape = (*hidden.shape[:-1], -1, attention.head_dim)
        q = attention.q_norm(attention.q_proj(hidden).view(shape)).transpose(1, 2)
        k = attention.k_norm(attention.k_proj(hidden).view(shape)).transpose(1, 2)
        v = attention.v_proj(hidden).view(shape).transpose(1, 2)
        cos, sin = self.model.model.rotary_emb(hidden, positions)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        return q, k, v

    @staticmethod
    def _finish(layer, residual, attended):
        attended = attended.transpose(1, 2).reshape(*residual.shape[:-1], -1)
        hidden = residual + layer.self_attn.o_proj(attended)
        return hidden + layer.mlp(layer.post_attention_layernorm(hidden))

    @torch.no_grad()
    def prefill_anchors(self, input_ids, valid, anchors, keep, *, capture_layer_ids=()):
        if input_ids.ndim != 2 or valid.shape != input_ids.shape:
            raise ValueError("Prefix token IDs and validity must have matching [N,S] axes")
        if anchors.shape != keep.shape or anchors.size(0) != input_ids.size(0):
            raise ValueError("Anchor axes must match prefix batch")
        if (anchors[keep] < 0).any() or (anchors[keep] >= input_ids.size(1)).any():
            raise ValueError("Anchor outside source document")
        if not valid.bool().gather(1, anchors).masked_fill(~keep, True).all():
            raise ValueError("Active anchors must point to valid tokens")
        length = int(anchors.masked_fill(~keep, 0).max().item()) + 1
        valid = valid[:, :length].bool()
        safe_anchors = anchors.masked_fill(~keep, 0)
        captured, handles = {}, []
        for index in capture_layer_ids:
            def capture(module, args, output, layer_id=index):
                captured[layer_id] = output[0] if isinstance(output, tuple) else output
            handles.append(self.model.model.layers[index].register_forward_hook(capture))
        try:
            output = self.model.model(input_ids=input_ids[:, :length], attention_mask=valid,
                                      use_cache=True, output_hidden_states=False)
        finally:
            for handle in handles:
                handle.remove()
        cache = output.past_key_values
        if hasattr(cache, "to_legacy_cache"):
            caches = list(cache.to_legacy_cache())
        else:
            caches = [(layer.keys, layer.values) for layer in cache.layers]
        hidden = output.last_hidden_state
        anchor_hidden = hidden.gather(1, safe_anchors[..., None].expand(-1, -1, hidden.size(-1)))
        self.prefill_calls += 1
        return AnchorPrefix(caches, valid, safe_anchors, keep, anchor_hidden, captured)

    @torch.no_grad()
    def score_sampled_blocks(self, prefix, draft_ids, *, cached_anchor_hidden=None):
        n, a, k = draft_ids.shape
        if prefix.anchors.shape != (n, a) or len(prefix.key_values) != len(self.model.model.layers):
            raise ValueError("Expired or incompatible anchor prefix")
        first = prefix.anchor_hidden if cached_anchor_hidden is None else cached_anchor_hidden
        if first.shape != (n, a, self.model.config.hidden_size):
            raise ValueError("Cached anchor hidden must have [N,A,D] axes")
        if k == 1:
            self.branch_calls += 1
            return first.unsqueeze(-2)
        slots = k - 1
        hidden = self.model.get_input_embeddings()(draft_ids[..., :-1].reshape(n, -1))
        positions = (prefix.anchors[..., None] + 1 + torch.arange(slots, device=draft_ids.device)).reshape(n, -1)
        prefix_length = prefix.valid.size(1)

        def branch_mask(b, h, q, key):
            branch = (q // slots).clamp_max(a - 1)
            active = prefix.keep[b, branch]
            source = (key < prefix_length) & prefix.valid[b, key.clamp(0, prefix_length - 1)]
            source = source & (key <= prefix.anchors[b, branch])
            own = (key >= prefix_length) & ((key - prefix_length) // slots == branch)
            own = own & ((key - prefix_length) % slots <= q % slots)
            return (active & (source | own)) | (~active & (key == 0))

        for layer, (prefix_k, prefix_v) in zip(self.model.model.layers, prefix.key_values):
            q, k_branch, v_branch = self._qkv(layer.self_attn, layer.input_layernorm(hidden), positions)
            keys = torch.cat((prefix_k, k_branch), dim=-2)
            values = torch.cat((prefix_v, v_branch), dim=-2)
            hidden = self._finish(layer, hidden, _attend(q, keys, values, branch_mask, layer.self_attn.scaling))
        tail = self.model.model.norm(hidden).reshape(n, a, slots, -1)
        self.branch_calls += 1
        return torch.cat((first.unsqueeze(-2), tail), dim=-2)
