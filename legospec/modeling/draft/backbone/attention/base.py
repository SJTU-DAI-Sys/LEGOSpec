"""Shared dense/latent attention lifecycle: projections, KV commit and execution."""

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from legospec.modeling.draft.backbone.state import LayerState


class AttentionBase(nn.Module):
    """Shared scaffold for the DFlash-family attention modes.

    Subclasses own only the projection parameterization: ``_init_projections``
    builds the weights and must define ``scaling``, ``num_key_value_groups``,
    and ``o_proj`` (the shared forward relies on them); ``_compute_qkv``
    returns rotated ``(q, k, v)`` in ``(batch, heads, seq, dim)`` layout with
    keys ordered context-then-draft. Everything the modes must agree on —
    KV-cache updates, backend dispatch, fully-masked-query zeroing, and the
    output projection — lives here so it is maintained in exactly one place.
    """

    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        if config._attn_implementation == "flex_attention":
            assert config.attention_dropout == 0.0, (
                "DFlash FlexAttention requires attention_dropout=0.0"
            )
        self.is_causal = False
        self.sliding_window = (
            config.sliding_window
            if config.layer_types[layer_idx] == "sliding_attention"
            else None
        )
        self._init_projections(config)
        for attribute in ("scaling", "num_key_value_groups", "o_proj"):
            assert hasattr(self, attribute), (
                f"_init_projections must define {attribute}"
            )

    def _init_projections(self, config: Qwen3Config) -> None:
        raise NotImplementedError

    def _compute_qkv(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class AttentionForward:
    def forward(
        self, draft_hidden_states, context_hidden_states, position_embeddings, inputs
    ):
        query_states, key_states, value_states = self._compute_qkv(
            draft_hidden_states, context_hidden_states, position_embeddings
        )
        context_length = context_hidden_states.size(1)
        batch_size = draft_hidden_states.size(0)
        positions = inputs.context_position_ids
        if positions is None:
            positions = torch.arange(
                context_length, device=draft_hidden_states.device
            ).expand(batch_size, -1)
        valid = inputs.context_valid_mask
        if valid is None:
            valid = torch.ones_like(positions, dtype=torch.bool)
        documents = inputs.document_ids
        if documents is None:
            documents = torch.zeros_like(positions)
        context_key_states, context_value_states = (
            key_states[:, :, :context_length],
            value_states[:, :, :context_length],
        )
        if inputs.draft_state is not None:
            previous = inputs.draft_state.layers.get(self.layer_idx)
            if previous is not None:
                # Caller supplies only new confirmed context; duplicates are rejected.
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
                context_key_states = torch.cat(
                    (previous.key_states, context_key_states), dim=2
                )
                context_value_states = torch.cat(
                    (previous.value_states, context_value_states), dim=2
                )
                positions = torch.cat((previous.positions, positions), dim=1)
                valid = torch.cat((previous.valid, valid), dim=1)
                documents = torch.cat((previous.document_ids, documents), dim=1)
            key_states = torch.cat(
                (context_key_states, key_states[:, :, context_length:]), dim=2
            )
            value_states = torch.cat(
                (context_value_states, value_states[:, :, context_length:]), dim=2
            )
            cache_key, cache_value, cache_positions, cache_valid, cache_documents = (
                context_key_states,
                context_value_states,
                positions,
                valid,
                documents,
            )
            cache_valid = valid & (positions < inputs.anchor_positions[:, :1])
            if self.window_size is not None:
                # Compact each sample independently: even ragged batches physically
                # retain at most W confirmed keys, with absolute positions intact.
                width = min(self.window_size, positions.size(1))
                priority = positions.masked_fill(~cache_valid, -1)
                indices = priority.argsort(dim=1, descending=True)[:, :width]
                indices = indices.flip(-1)
                cache_key = context_key_states.gather(
                    2,
                    indices[:, None, :, None].expand(
                        -1, context_key_states.size(1), -1, context_key_states.size(-1)
                    ),
                )
                cache_value = context_value_states.gather(
                    2,
                    indices[:, None, :, None].expand(
                        -1,
                        context_value_states.size(1),
                        -1,
                        context_value_states.size(-1),
                    ),
                )
                cache_positions = positions.gather(1, indices)
                cache_valid = cache_valid.gather(1, indices)
                cache_documents = documents.gather(1, indices)
            inputs.draft_state.layers[self.layer_idx] = LayerState(
                cache_key.detach(),
                cache_value.detach(),
                cache_positions,
                cache_valid,
                cache_documents,
            )
        if getattr(self, "is_dsa", False) and inputs.selection is not None:
            from legospec.modeling.draft.backbone.attention.access.dsa import (
                gather_attention,
            )

            return gather_attention(
                self,
                query_states,
                key_states,
                value_states,
                inputs,
                positions,
                valid,
                documents,
            )
        if (
            inputs.packed_plan is not None
            and inputs.packed_plan.num_samples > 1
            and self.backend in ("sdpa", "eager")
            and not getattr(self, "is_dsa", False)
        ):
            return self._segmented_attention(
                query_states,
                key_states,
                value_states,
                inputs,
                positions,
                valid,
                draft_hidden_states,
            )
        if (
            self.backend == "flex_attention"
            and inputs.packed_plan is not None
            and not getattr(self, "is_dsa", False)
        ):
            from legospec.modeling.draft.backbone.attention.access.visibility import (
                indexed_visibility,
            )
            from legospec.modeling.draft.backbone.attention.backends.flex import (
                compile_friendly_flex_attention,
                create_flex_block_mask,
                flex_kernel_options,
            )

            mask_mod, query_visible = indexed_visibility(
                inputs,
                positions,
                valid,
                documents,
                draft_hidden_states.size(1),
                self.window_size,
                causal_queries=getattr(self, "causal_queries", False),
            )
            mask = create_flex_block_mask(
                mask_mod,
                B=query_states.size(0),
                H=None,
                Q_LEN=query_states.size(2),
                KV_LEN=key_states.size(2),
                device=str(query_states.device),
                backend=self.flex_backend,
            )
            output = compile_friendly_flex_attention(
                query_states,
                key_states,
                value_states,
                block_mask=mask,
                scale=self.scaling,
                enable_gqa=True,
                kernel_options=flex_kernel_options(self.flex_backend),
            )
            output = output.transpose(1, 2).reshape(*draft_hidden_states.shape[:2], -1)
            return self.o_proj(output) * query_visible.transpose(1, 2).to(output.dtype)
        visibility = self.access(
            inputs,
            positions,
            valid,
            documents,
            draft_hidden_states.size(1),
            **(
                {"causal_queries": True}
                if getattr(self, "causal_queries", False)
                else {}
            ),
            **(
                {"window_size": self.window_size}
                if self.window_size is not None
                else {}
            ),
        )
        # CUDA SDPA may return finite zeros for fully masked query rows yet
        # produce NaN dQ in backward. Give only those unused rows one finite
        # key, then zero their output below. Legal rows and gradients retain
        # exactly the original visibility; no index leaves the device.
        query_visible = visibility.any(-1)
        safe_visibility = visibility.clone()
        if visibility.size(-1):
            safe_visibility[..., 0] |= ~query_visible
        repeats = query_states.size(1) // key_states.size(1)
        if repeats != 1:
            key_states = key_states.repeat_interleave(repeats, dim=1)
            value_states = value_states.repeat_interleave(repeats, dim=1)
        if self.backend == "eager":
            scores = (
                torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scaling
            )
            scores = scores.masked_fill(~safe_visibility, float("-inf"))
            attention_probabilities = (
                torch.softmax(scores, dim=-1, dtype=torch.float32)
                .to(query_states.dtype)
                .nan_to_num()
            )
            attention_probabilities = F.dropout(
                attention_probabilities,
                p=self.attention_dropout,
                training=self.training,
            )
            output = torch.matmul(
                attention_probabilities.to(value_states.dtype), value_states
            )
        elif self.backend == "flex_attention":
            from legospec.modeling.draft.backbone.attention.backends.flex import (
                FLASH_KV_TILE,
                FLASH_Q_TILE,
                TRITON_MASK_TILE,
                compile_friendly_flex_attention,
                create_flex_block_mask,
                flex_kernel_options,
            )

            q_tile, kv_tile = (
                (FLASH_Q_TILE, FLASH_KV_TILE)
                if self.flex_backend.upper() == "FLASH"
                else (TRITON_MASK_TILE, TRITON_MASK_TILE)
            )
            # Block-mask construction visits complete tiles, including padding.
            # Pad with False so the callback only loads a tensor; querying shapes
            # inside mask_mod breaks PyTorch 2.9's dynamic FlexAttention lowering.
            flat_visibility = F.pad(
                safe_visibility[:, 0],
                (0, -visibility.size(3) % kv_tile, 0, -visibility.size(2) % q_tile),
                value=False,
            )

            def mask_mod(b, h, q, k):
                return flat_visibility[b, q, k]

            mask = create_flex_block_mask(
                mask_mod,
                B=query_states.size(0),
                H=None,
                Q_LEN=query_states.size(2),
                KV_LEN=key_states.size(2),
                device=str(query_states.device),
                backend=self.flex_backend,
            )
            output = compile_friendly_flex_attention(
                query_states,
                key_states,
                value_states,
                block_mask=mask,
                scale=self.scaling,
                kernel_options=flex_kernel_options(self.flex_backend),
            )
        else:
            output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=safe_visibility,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=False,
                scale=self.scaling,
            )
        output = output.transpose(1, 2).reshape(*draft_hidden_states.shape[:2], -1)
        return self.o_proj(output) * query_visible.transpose(1, 2).to(output.dtype)

    def _segmented_attention(
        self, queries, keys, values, inputs, positions, valid, hidden
    ):
        """Avoid a dense cross-document Q×KV allocation in SDPA/eager mode."""
        outputs = []
        context_length = positions.size(1)
        context_start = block_start = 0
        block_size = hidden.size(1) // inputs.anchor_positions.size(1)
        for sample in inputs.packed_plan.samples:
            context_end = context_start + sample.length
            block_end = block_start + len(sample.anchors)
            q_start, q_end = block_start * block_size, block_end * block_size
            if q_end > q_start:
                query = queries[:, :, q_start:q_end]
                key = torch.cat(
                    (
                        keys[:, :, context_start:context_end],
                        keys[:, :, context_length + q_start : context_length + q_end],
                    ),
                    dim=2,
                )
                value = torch.cat(
                    (
                        values[:, :, context_start:context_end],
                        values[:, :, context_length + q_start : context_length + q_end],
                    ),
                    dim=2,
                )
                anchor = inputs.anchor_positions[
                    :, block_start:block_end
                ].repeat_interleave(block_size, 1)
                live = inputs.block_keep_mask[
                    :, block_start:block_end
                ].repeat_interleave(block_size, 1)
                pos = positions[:, None, context_start:context_end]
                context = (pos < anchor[..., None]) & valid[
                    :, None, context_start:context_end
                ]
                if self.window_size is not None:
                    context &= pos >= anchor[..., None] - self.window_size
                slots = torch.arange(q_end - q_start, device=query.device)
                own = slots[:, None] // block_size == slots[None] // block_size
                if getattr(self, "causal_queries", False):
                    own &= slots[:, None] >= slots[None]
                current = own[None] & live[:, None]
                mask = (
                    torch.cat((context, current), -1)[:, None] & live[:, None, :, None]
                )
                mask[..., 0] |= ~live[:, None]
                # Arbitrary-mask SDPA GQA may fall back to the quadratic math
                # kernel. Keep its established equal-head fused-kernel path;
                # native GQA is used by the indexed Flex implementation above.
                repeats = query.size(1) // key.size(1)
                if repeats != 1:
                    key, value = (
                        key.repeat_interleave(repeats, 1),
                        value.repeat_interleave(repeats, 1),
                    )
                if self.backend == "sdpa":
                    output = F.scaled_dot_product_attention(
                        query,
                        key,
                        value,
                        attn_mask=mask,
                        dropout_p=self.attention_dropout if self.training else 0.0,
                        scale=self.scaling,
                    )
                else:
                    scores = (query @ key.transpose(-2, -1)) * self.scaling
                    probability = (
                        scores.masked_fill(~mask, float("-inf"))
                        .softmax(-1, dtype=torch.float32)
                        .to(query.dtype)
                    )
                    probability = F.dropout(
                        probability, p=self.attention_dropout, training=self.training
                    )
                    output = probability @ value
                outputs.append(output * live[:, None, :, None].to(output.dtype))
            context_start, block_start = context_end, block_end
        output = (
            torch.cat(outputs, dim=2).transpose(1, 2).reshape(*hidden.shape[:2], -1)
        )
        live = inputs.block_keep_mask.repeat_interleave(block_size, 1)[..., None]
        return self.o_proj(output) * live.to(output.dtype)
