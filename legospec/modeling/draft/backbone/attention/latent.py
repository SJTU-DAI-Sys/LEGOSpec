from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

"""MLA projections, expanded training, absorbed inference and latent cache."""

import math
from copy import deepcopy

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config, rotate_half

from legospec.modeling.draft.backbone.attention.access.visibility import (
    build_access,
)
from legospec.modeling.draft.backbone.attention.base import (
    AttentionBase,
    AttentionForward,
)
from legospec.modeling.draft.backbone.state import MLALayerState


def _require_bool_config(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"DFlash {field} must be a boolean, got {value!r}")
    return value


def _resolve_mla_rope_interleaved(config: Qwen3Config) -> bool:
    """Rotation convention from the standard MLA ``rope_interleave`` field."""

    return _require_bool_config(
        getattr(config, "rope_interleave", True),
        "config.rope_interleave",
    )


def _rotate_half_interleaved(x: torch.Tensor) -> torch.Tensor:
    """Rotate consecutive pairs, the DeepSeek-style MLA RoPE convention."""

    paired = x.reshape(*x.shape[:-1], -1, 2)
    first, second = paired.unbind(dim=-1)
    return torch.stack((-second, first), dim=-1).flatten(-2)


def apply_mla_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    interleaved: bool,
) -> torch.Tensor:
    if interleaved:
        half = cos.shape[-1] // 2
        cos = cos[..., :half].repeat_interleave(2, dim=-1)
        sin = sin[..., :half].repeat_interleave(2, dim=-1)
        rotated = _rotate_half_interleaved(x)
    else:
        rotated = rotate_half(x)
    return x * cos.unsqueeze(1) + rotated * sin.unsqueeze(1)


class MLAProjection(AttentionBase):
    """Multi-head Latent Attention projections for DFlash-family drafts.

    Standard MLA parameterization: an optional low-rank Q path, a shared
    compressed KV latent, and partial RoPE (interleaved or NeoX, from the
    standard ``rope_interleave`` field). K/V are expanded per head for
    training so the mode runs through the same masks and attention backends
    as :class:`GQAProjection`.
    """

    def _init_projections(self, config: Qwen3Config) -> None:
        self.num_heads = int(config.num_attention_heads)
        self.num_key_value_groups = 1
        self.q_lora_rank = (
            None
            if getattr(config, "q_lora_rank", None) is None
            else int(config.q_lora_rank)
        )
        self.kv_lora_rank = int(config.kv_lora_rank)
        self.qk_nope_head_dim = int(config.qk_nope_head_dim)
        self.qk_rope_head_dim = int(config.qk_rope_head_dim)
        self.v_head_dim = int(config.v_head_dim)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.head_dim = self.qk_head_dim
        self.scaling = self.qk_head_dim**-0.5
        # DeepSeek YaRN applies mscale_all_dim to the full QK logits; the
        # rotary attention factor separately scales only the partial-RoPE slice.
        rope_parameters = config.rope_parameters
        if rope_parameters.get("rope_type", "default") != "default":
            mscale_all_dim = rope_parameters.get("mscale_all_dim", 0)
            if mscale_all_dim:
                factor = rope_parameters["factor"]
                mscale = (
                    1.0
                    if factor <= 1
                    else 0.1 * mscale_all_dim * math.log(factor) + 1.0
                )
                self.scaling *= mscale * mscale
        self.rope_interleaved = _resolve_mla_rope_interleaved(config)

        hidden_size = int(config.hidden_size)
        bias = bool(config.attention_bias)
        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(
                hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
            )
        else:
            self.q_a_proj = nn.Linear(hidden_size, self.q_lora_rank, bias=bias)
            self.q_a_layernorm = Qwen3RMSNorm(
                self.q_lora_rank,
                config.rms_norm_eps,
            )
            self.q_b_proj = nn.Linear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
            )
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=bias,
        )
        self.kv_a_layernorm = Qwen3RMSNorm(
            self.kv_lora_rank,
            config.rms_norm_eps,
        )
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            hidden_size,
            bias=bias,
        )

    def _project_q(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.q_lora_rank is None:
            return self.q_proj(hidden_states)
        return self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))

    def _project_kv(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len = hidden_states.shape[:2]
        kv_compressed, k_rope = self.kv_a_proj_with_mqa(hidden_states).split(
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        kv = self.kv_b_proj(self.kv_a_layernorm(kv_compressed)).view(
            bsz,
            seq_len,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        k_nope, value = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        return k_nope, k_rope, value

    def _compute_qkv(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, q_len = hidden_states.shape[:2]
        query = self._project_q(hidden_states).view(
            bsz,
            q_len,
            self.num_heads,
            self.qk_head_dim,
        )
        q_nope, q_rope = query.split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1,
        )
        ctx_k_nope, ctx_k_rope, ctx_value = self._project_kv(target_hidden)
        noise_k_nope, noise_k_rope, noise_value = self._project_kv(hidden_states)
        k_nope = torch.cat((ctx_k_nope, noise_k_nope), dim=1)
        k_rope = torch.cat((ctx_k_rope, noise_k_rope), dim=1)
        v = torch.cat((ctx_value, noise_value), dim=1)
        # The model-level rotary carries qk_rope_head_dim; queries take the
        # trailing positions exactly like apply_rotary_pos_emb.
        cos, sin = position_embeddings
        q_rope = apply_mla_rope(
            q_rope.transpose(1, 2),
            cos[:, -q_len:],
            sin[:, -q_len:],
            interleaved=self.rope_interleaved,
        )
        k_rope = apply_mla_rope(
            k_rope.unsqueeze(1),
            cos,
            sin,
            interleaved=self.rope_interleaved,
        ).expand(-1, self.num_heads, -1, -1)
        q = torch.cat((q_nope.transpose(1, 2), q_rope), dim=-1)
        k = torch.cat((k_nope.transpose(1, 2), k_rope), dim=-1)
        return q, k, v.transpose(1, 2)


class MLAAttention(AttentionForward, MLAProjection):
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
            self.context_kv_a_proj = deepcopy(self.kv_a_proj_with_mqa)
            self.context_kv_norm = deepcopy(self.kv_a_layernorm)
            self.context_kv_b_proj = deepcopy(self.kv_b_proj)

    def forward(
        self, draft_hidden_states, context_hidden_states, position_embeddings, inputs
    ):
        # Evaluation with gradients (e.g. training diagnostics) retains the training path.
        if self.training or torch.is_grad_enabled():
            return super().forward(
                draft_hidden_states, context_hidden_states, position_embeddings, inputs
            )
        batch_size, query_length = draft_hidden_states.shape[:2]
        context_length = context_hidden_states.size(1)
        query = (
            self._project_q(draft_hidden_states)
            .view(batch_size, query_length, self.num_heads, self.qk_head_dim)
            .transpose(1, 2)
        )
        query_nope, query_rope = query.split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], -1
        )
        cos, sin = position_embeddings
        query_rope = apply_mla_rope(
            query_rope,
            cos[:, -query_length:],
            sin[:, -query_length:],
            interleaved=self.rope_interleaved,
        )
        context_projection = (
            self.context_kv_a_proj if self.separate_context else self.kv_a_proj_with_mqa
        )
        context_norm = (
            self.context_kv_norm if self.separate_context else self.kv_a_layernorm
        )
        context_latent, context_rope = context_projection(context_hidden_states).split(
            [self.kv_lora_rank, self.qk_rope_head_dim], -1
        )
        context_latent = context_norm(context_latent)
        draft_latent, draft_rope = self.kv_a_proj_with_mqa(draft_hidden_states).split(
            [self.kv_lora_rank, self.qk_rope_head_dim], -1
        )
        draft_latent = self.kv_a_layernorm(draft_latent)
        key_rope = apply_mla_rope(
            torch.cat((context_rope, draft_rope), 1).unsqueeze(1),
            cos,
            sin,
            interleaved=self.rope_interleaved,
        ).squeeze(1)
        context_rope, draft_rope = key_rope.split([context_length, query_length], 1)
        positions = inputs.context_position_ids
        if positions is None:
            positions = torch.arange(context_length, device=query.device)[None].expand(
                batch_size, -1
            )
        valid = inputs.context_valid_mask
        if valid is None:
            valid = torch.ones_like(positions, dtype=torch.bool)
        documents = inputs.document_ids
        if documents is None:
            documents = torch.zeros_like(positions)
        if inputs.draft_state is not None:
            previous = inputs.draft_state.layers.get(self.layer_idx)
            if previous is not None:
                if (
                    context_length
                    and previous.positions.numel()
                    and (
                        positions[:, :1]
                        <= previous.positions.masked_fill(~previous.valid, -1).amax(
                            -1, keepdim=True
                        )
                    ).any()
                ):
                    raise ValueError(
                        "Cached forward requires strictly new confirmed context positions"
                    )
                context_latent = torch.cat((previous.kv_latent, context_latent), 1)
                context_rope = torch.cat((previous.key_rope, context_rope), 1)
                positions = torch.cat((previous.positions, positions), 1)
                valid = torch.cat((previous.valid, valid), 1)
                documents = torch.cat((previous.document_ids, documents), 1)
            cache_valid = valid & (positions < inputs.anchor_positions[:, :1])
            cache = MLALayerState(
                context_latent, context_rope, positions, cache_valid, documents
            )
            if self.window_size is not None:
                width = min(self.window_size, positions.size(1))
                indices = (
                    positions.masked_fill(~cache_valid, -1)
                    .argsort(1, descending=True)[:, :width]
                    .flip(-1)
                )
                cache = MLALayerState(
                    context_latent.gather(
                        1, indices[..., None].expand(-1, -1, self.kv_lora_rank)
                    ),
                    context_rope.gather(
                        1, indices[..., None].expand(-1, -1, self.qk_rope_head_dim)
                    ),
                    positions.gather(1, indices),
                    cache_valid.gather(1, indices),
                    documents.gather(1, indices),
                )
            inputs.draft_state.layers[self.layer_idx] = cache
        context_weights = (
            self.context_kv_b_proj.weight
            if self.separate_context
            else self.kv_b_proj.weight
        ).view(
            self.num_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank
        )
        draft_weights = self.kv_b_proj.weight.view(
            self.num_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank
        )
        # Keep separate context/draft weights, but normalize both score sets together.
        scores = []
        for latent, rope, weights in (
            (context_latent, context_rope, context_weights),
            (draft_latent, draft_rope, draft_weights),
        ):
            absorbed_query = torch.einsum(
                "bhqd,hdr->bhqr", query_nope, weights[:, : self.qk_nope_head_dim]
            )
            scores.append(
                torch.einsum("bhqr,bsr->bhqs", absorbed_query, latent)
                + torch.einsum("bhqd,bsd->bhqs", query_rope, rope)
            )
        visibility = self.access(
            inputs,
            positions,
            valid,
            documents,
            query_length,
            **(
                {"window_size": self.window_size}
                if self.window_size is not None
                else {}
            ),
        )
        query_visible = visibility.any(-1)
        visibility[..., 0] |= ~query_visible
        probability = (
            (torch.cat(scores, -1).float() * self.scaling)
            .masked_fill(~visibility, -torch.inf)
            .softmax(-1)
            .to(query.dtype)
        )
        context_probability, draft_probability = probability.split(
            [context_latent.size(1), query_length], -1
        )
        output = None
        for attention, latent, weights in (
            (context_probability, context_latent, context_weights),
            (draft_probability, draft_latent, draft_weights),
        ):
            latent_output = torch.einsum("bhqs,bsr->bhqr", attention, latent)
            contribution = torch.einsum(
                "bhqr,hdr->bhqd", latent_output, weights[:, self.qk_nope_head_dim :]
            )
            output = contribution if output is None else output + contribution
        output = output.transpose(1, 2).reshape(batch_size, query_length, -1)
        return self.o_proj(output) * query_visible.transpose(1, 2).to(output.dtype)

    def _compute_qkv(
        self, draft_hidden_states, context_hidden_states, position_embeddings
    ):
        if not self.separate_context:
            return super()._compute_qkv(
                draft_hidden_states, context_hidden_states, position_embeddings
            )
        batch_size, length = draft_hidden_states.shape[:2]
        query_states = self._project_q(draft_hidden_states).view(
            batch_size, length, self.num_heads, self.qk_head_dim
        )
        query_nope, query_rope = query_states.split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], -1
        )
        latent, context_rope = self.context_kv_a_proj(context_hidden_states).split(
            [self.kv_lora_rank, self.qk_rope_head_dim], -1
        )
        expanded = self.context_kv_b_proj(self.context_kv_norm(latent)).view(
            batch_size, -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        context_nope, context_value_states = expanded.split(
            [self.qk_nope_head_dim, self.v_head_dim], -1
        )
        draft_nope, draft_rope, draft_value = self._project_kv(draft_hidden_states)
        key_nope = torch.cat((context_nope, draft_nope), 1).transpose(1, 2)
        key_rope = torch.cat((context_rope, draft_rope), 1).unsqueeze(1)
        cos, sin = position_embeddings
        query_rope = apply_mla_rope(
            query_rope.transpose(1, 2),
            cos[:, -length:],
            sin[:, -length:],
            interleaved=self.rope_interleaved,
        )
        key_rope = apply_mla_rope(
            key_rope, cos, sin, interleaved=self.rope_interleaved
        ).expand(-1, self.num_heads, -1, -1)
        return (
            torch.cat((query_nope.transpose(1, 2), query_rope), -1),
            torch.cat((key_nope, key_rope), -1),
            torch.cat((context_value_states, draft_value), 1).transpose(1, 2),
        )
