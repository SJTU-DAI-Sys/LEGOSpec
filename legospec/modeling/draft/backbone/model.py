from copy import deepcopy
from dataclasses import replace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3MLP,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)

from legospec.modeling.draft.backbone.attention import build_attention
from legospec.modeling.draft.backbone.attention.access.indexer import (
    AnchorIndexer,
)
from legospec.modeling.draft.backbone.attention.linear.cache import (
    prepare_cache,
)
from legospec.modeling.draft.backbone.attention.linear.events import (
    build_event_layout,
)
from legospec.modeling.draft.backbone.convolution import build_local_operator
from legospec.modeling.draft.backbone.state import DraftState
from legospec.modeling.draft.contracts import BackboneOutput


class Layer(nn.Module):
    def __init__(self, config, layer_idx, recipe):
        super().__init__()
        self.block_size = config.block_size
        self.input_fusion = config.composable_config["backbone"].get(
            "input_fusion", "identity"
        )
        self.recurrent = recipe["attention"]["representation"] in ("gdn", "kda")
        if self.recurrent:
            self.attention = build_attention(config, layer_idx, recipe)
        else:
            local = deepcopy(config)
            if self.input_fusion == "concat_hidden":
                local.attention_input_size = 2 * config.hidden_size
            local.layer_types = ["full_attention"] * config.num_hidden_layers
            local.sliding_window = None
            local._attn_implementation = recipe["attention"].get("backend", "sdpa")
            for key, value in recipe["attention"].items():
                if key in (
                    "q_lora_rank",
                    "kv_lora_rank",
                    "qk_nope_head_dim",
                    "qk_rope_head_dim",
                    "v_head_dim",
                    "rope_interleave",
                ):
                    setattr(local, key, value)
                elif recipe["attention"]["representation"] in (
                    "mha",
                    "gqa",
                    "mqa",
                    "mla",
                ) and key in (
                    "num_attention_heads",
                    "num_key_value_heads",
                    "head_dim",
                ):
                    setattr(local, key, value)
            local.rope_parameters = deepcopy(
                getattr(local, "rope_parameters", None)
                or local.rope_scaling
                or {"rope_type": "default"}
            )
            self.attention = build_attention(
                local,
                layer_idx,
                recipe,
                config.composable_config["context"]["type"] == "gated_fusion",
            )
            self.attention.causal_queries = (
                config.composable_config["backbone"].get("query_attention")
                == "causal_parallel"
            )
            rotary_config = local
            if recipe["attention"]["representation"] == "mla":
                rotary_config = deepcopy(local)
                rotary_config.head_dim = local.qk_rope_head_dim
            self.rotary_emb = Qwen3RotaryEmbedding(rotary_config)

        self.hidden_norm = (
            Qwen3RMSNorm(config.hidden_size, config.rms_norm_eps)
            if self.input_fusion == "concat_hidden"
            else None
        )
        self.input_norm = Qwen3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_norm = Qwen3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = Qwen3MLP(config)
        operator = recipe.get("local_operator", {"type": "none"})
        self.attention_conv = build_local_operator(config, operator)
        self.mlp_conv = build_local_operator(config, operator)
        self.block_size = config.block_size

    def forward(
        self,
        draft_hidden_states,
        context_hidden_states,
        inputs,
        layout=None,
        previous=None,
        mask=None,
        use_flex=False,
    ):
        if self.input_fusion == "concat_hidden":
            return self.forward_stepwise(
                draft_hidden_states,
                context_hidden_states,
                inputs,
                layout,
                previous,
                mask,
                use_flex,
            )
        state = teacher = None
        if self.recurrent:
            draft_hidden_states = self.attention.add_positions(draft_hidden_states)
        else:
            positions = inputs.context_position_ids
            if positions is None:
                positions = torch.arange(
                    context_hidden_states.size(1), device=draft_hidden_states.device
                ).expand(draft_hidden_states.size(0), -1)
            query_positions = (
                inputs.anchor_positions[..., None]
                + torch.arange(self.block_size, device=positions.device)
            ).flatten(1)
            embeddings = self.rotary_emb(
                draft_hidden_states, torch.cat((positions, query_positions), 1)
            )
        residual = draft_hidden_states
        draft_hidden_states = self.input_norm(draft_hidden_states)
        if self.recurrent:
            coefficients = None
            if self.attention_conv is not None:
                draft_hidden_states, coefficients = self.attention_conv.prepare(
                    draft_hidden_states
                )
            query_context = (
                self.input_norm(context_hidden_states)
                if self.attention.mixer.q_mode == "confirmed_history"
                else None
            )
            draft_hidden_states, state = self.attention(
                draft_hidden_states,
                context_hidden_states,
                inputs,
                layout,
                previous,
                query_context,
            )
            if self.attention_conv is not None:
                draft_hidden_states = self.attention_conv.finish(
                    draft_hidden_states, coefficients
                )
        else:
            coefficients = None
            if self.attention_conv is not None:
                draft_hidden_states, coefficients = self.attention_conv.prepare(
                    draft_hidden_states
                )
            draft_hidden_states = self.attention(
                draft_hidden_states, context_hidden_states, embeddings, inputs
            )
            if isinstance(draft_hidden_states, tuple):
                draft_hidden_states, teacher = draft_hidden_states
            if self.attention_conv is not None:
                draft_hidden_states = self.attention_conv.finish(
                    draft_hidden_states, coefficients
                )
        draft_hidden_states = residual + draft_hidden_states
        residual = draft_hidden_states
        draft_hidden_states = self.post_attention_norm(draft_hidden_states)
        if self.mlp_conv is not None:
            draft_hidden_states, coefficients = self.mlp_conv.prepare(
                draft_hidden_states
            )
        draft_hidden_states = self.mlp(draft_hidden_states)
        if self.mlp_conv is not None:
            draft_hidden_states = self.mlp_conv.finish(
                draft_hidden_states, coefficients
            )
        hidden = residual + draft_hidden_states
        return hidden, state, teacher

    def forward_stepwise(
        self, hidden, embeds, positions, key, value, mask, use_flex=False
    ):
        fused = torch.cat((self.input_norm(embeds), self.hidden_norm(hidden)), -1)
        mixed, key, value = self.attention.forward_stepwise(
            fused, self.rotary_emb(hidden, positions), key, value, mask, use_flex
        )
        output = hidden + mixed
        return output + self.mlp(self.post_attention_norm(output)), key, value


class Backbone(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        recipes = config.composable_config["backbone"]["layers"]
        self.block_size = config.block_size
        self.stepwise = (
            config.composable_config["backbone"].get("schedule") == "stepwise"
        )
        self.layers = nn.ModuleList(
            [Layer(config, index, recipe) for index, recipe in enumerate(recipes)]
        )
        self.gdn_indices = {i for i, layer in enumerate(self.layers) if layer.recurrent}
        self.attention_indices = set(range(len(self.layers))) - self.gdn_indices
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.last_event_count = 0
        self.indexer = (
            AnchorIndexer(config) if config.composable_config.get("indexer") else None
        )

    def forward(self, context):
        inputs, original = context.inputs, context.inputs.draft_state
        if len(context.context_hidden_states) != len(self.layers):
            raise ValueError("Context mapping count must equal layer count")
        if original is not None:
            if (
                set(original.layers) - self.attention_indices
                or (original.layers and set(original.layers) != self.attention_indices)
                or (not self.gdn_indices and original.backbone_state is not None)
            ):
                raise ValueError("Cache layers do not match this layer stack")
            if self.gdn_indices and (
                (original.backbone_state is None and original.layers)
                or (
                    original.backbone_state is not None
                    and set(original.layers) != self.attention_indices
                )
            ):
                raise ValueError("Incomplete layer-stack cache")
            working = DraftState(
                dict(original.layers), original.backbone_state, original.indexer_state
            )
            docs = inputs.anchor_document_ids
            docs = (
                torch.zeros_like(inputs.anchor_positions[:, 0])
                if docs is None
                else docs[:, 0]
            )
            # Switch documents without mutating the caller's cache on failure.
            working.layers = {
                i: replace(
                    state, valid=state.valid & (state.document_ids == docs[:, None])
                )
                for i, state in working.layers.items()
            }
            if working.indexer_state is not None:
                state = working.indexer_state
                working.indexer_state = replace(
                    state, valid=state.valid & (state.document_ids == docs[:, None])
                )
            inputs = replace(inputs, draft_state=working)
        previous, updated = (
            prepare_cache(self.layers, self.attention_indices, inputs)
            if self.gdn_indices
            else (None, None)
        )
        layout = (
            build_event_layout(inputs, self.block_size) if self.gdn_indices else None
        )
        self.last_event_count = 0 if layout is None else layout.num_events
        selection, teacher_sum, teacher_count = None, None, 0
        if self.indexer is not None and self.indexer.recipe.get("enabled", True):
            selection = self.indexer(inputs, self.layers[0])
            inputs = replace(inputs, selection=selection)
        hidden = inputs.draft_embeddings
        for index, (layer, features) in enumerate(
            zip(self.layers, context.context_hidden_states)
        ):

            def operation(
                hidden,
                features,
                module=layer,
                previous_state=None if previous is None else previous.get(index),
            ):
                return module(hidden, features, inputs, layout, previous_state)

            hidden, state, teacher = (
                checkpoint(operation, hidden, features, use_reentrant=False)
                if self.gradient_checkpointing and self.training
                else operation(hidden, features)
            )
            if updated is not None and layer.recurrent:
                updated.layers[index] = state
            if teacher is not None:
                teacher_sum = teacher if teacher_sum is None else teacher_sum + teacher
                teacher_count += 1
        auxiliary = (
            self.indexer.auxiliary(selection, teacher_sum / teacher_count)
            if teacher_count
            else None
        )
        hidden = self.norm(hidden)
        if self.gdn_indices:
            keep = inputs.block_keep_mask.repeat_interleave(self.block_size, -1)
            hidden = hidden * keep[..., None].to(hidden.dtype)
        if original is not None:
            original.layers = inputs.draft_state.layers
            original.indexer_state = inputs.draft_state.indexer_state
            original.backbone_state = updated
        return BackboneOutput(hidden, original, auxiliary)

    @property
    def layer(self):
        return self.layers[0]

    def empty_kv(self, hidden):
        attention = self.layer.attention
        shape = (
            hidden.shape[0],
            attention.config.num_key_value_heads,
            0,
            attention.head_dim,
        )
        return hidden.new_empty(shape), hidden.new_empty(shape)
