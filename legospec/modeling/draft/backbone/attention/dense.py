from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

"""MHA, GQA and MQA: head geometry, projections and softmax attention."""

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config, rotate_half

from legospec.modeling.draft.backbone.attention.access.visibility import (
    build_access,
)
from legospec.modeling.draft.backbone.attention.backends.flex import (
    compile_friendly_flex_attention,
    flex_kernel_options,
)
from legospec.modeling.draft.backbone.attention.base import (
    AttentionBase,
    AttentionForward,
)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GQAProjection(AttentionBase):
    """GQA/MHA projections over the family's context-then-draft KV layout."""

    def _init_projections(self, config: Qwen3Config) -> None:
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(
            getattr(config, "attention_input_size", config.hidden_size),
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            getattr(config, "attention_input_size", config.hidden_size),
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            getattr(config, "attention_input_size", config.hidden_size),
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, config.rms_norm_eps)

    def _compute_qkv(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        v = torch.cat([v_ctx, v_noise], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        return q, k, v


class GQAAttention(AttentionForward, GQAProjection):
    def __init__(self, config, layer_idx, recipe, separate_context=False):
        super().__init__(config, layer_idx)
        self.is_dsa = recipe.get("access") == "dsa"
        self.backend = recipe.get("backend", "sdpa")
        self.flex_backend = recipe.get("flex_backend", "TRITON")
        self.window_size = (
            recipe.get("window_size")
            if recipe.get("access") == "sliding_window"
            else None
        )
        self.access = build_access(recipe)
        self.separate_context = separate_context
        if separate_context:
            self.context_key_projection = deepcopy(self.k_proj)
            self.context_value_projection = deepcopy(self.v_proj)

    def _compute_qkv(
        self, draft_hidden_states, context_hidden_states, position_embeddings
    ):
        if not self.separate_context:
            return super()._compute_qkv(
                draft_hidden_states, context_hidden_states, position_embeddings
            )
        batch_size, length = draft_hidden_states.shape[:2]
        query_states = self.q_norm(
            self.q_proj(draft_hidden_states).view(batch_size, length, -1, self.head_dim)
        ).transpose(1, 2)
        key_states = torch.cat(
            (
                self.context_key_projection(context_hidden_states),
                self.k_proj(draft_hidden_states),
            ),
            dim=1,
        )
        value_states = torch.cat(
            (
                self.context_value_projection(context_hidden_states),
                self.v_proj(draft_hidden_states),
            ),
            dim=1,
        )
        key_states = self.k_norm(
            key_states.view(
                batch_size, -1, self.config.num_key_value_heads, self.head_dim
            )
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, -1, self.config.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, *position_embeddings
        )
        return query_states, key_states, value_states

    def forward_stepwise(
        self, fused, position_embeddings, key, value, mask, use_flex=False
    ):
        batch, length, _ = fused.shape
        q = self.q_norm(
            self.q_proj(fused).view(
                batch, length, self.config.num_attention_heads, self.head_dim
            )
        ).transpose(1, 2)
        k = self.k_norm(
            self.k_proj(fused).view(
                batch, length, self.config.num_key_value_heads, self.head_dim
            )
        ).transpose(1, 2)
        v = (
            self.v_proj(fused)
            .view(batch, length, self.config.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        cos, sin = position_embeddings
        cos, sin = cos[:, None], sin[:, None]
        q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
        key, value = torch.cat((key, k), -2), torch.cat((value, v), -2)
        if use_flex:
            attended = compile_friendly_flex_attention(
                q,
                key.contiguous(),
                value.contiguous(),
                block_mask=mask,
                enable_gqa=True,
                kernel_options=flex_kernel_options(self.flex_backend),
            )
        else:
            # Explicit mask also works for cached multi-token rebuilds.
            groups = self.config.num_attention_heads // self.config.num_key_value_heads
            k_all, v_all = (
                key.repeat_interleave(groups, 1),
                value.repeat_interleave(groups, 1),
            )
            if self.backend == "eager":
                scores = (
                    q.float() @ k_all.float().transpose(-1, -2)
                ) * self.head_dim**-0.5
                probabilities = (
                    scores.masked_fill(~mask, -torch.inf).softmax(-1).nan_to_num()
                )
                attended = probabilities.to(v_all.dtype) @ v_all
            else:
                attended = F.scaled_dot_product_attention(
                    q, k_all, v_all, attn_mask=mask
                )
        return (
            self.o_proj(attended.transpose(1, 2).reshape(batch, length, -1)),
            key,
            value,
        )
