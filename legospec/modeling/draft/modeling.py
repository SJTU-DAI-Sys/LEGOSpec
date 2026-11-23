"""Thin deployable orchestration of context, backbone and proposal."""

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3PreTrainedModel

from legospec.modeling.draft.backbone import build_backbone
from legospec.modeling.draft.backbone.stepwise import (
    extend_hidden_feedback,
    rollout_hidden_feedback,
)
from legospec.modeling.draft.configuration import LegoSpecConfig
from legospec.modeling.draft.context import build_context
from legospec.modeling.draft.proposal import build_proposal


class LegoSpecModel(Qwen3PreTrainedModel):
    config_class = LegoSpecConfig
    method_config_key = "composable_config"
    _no_split_modules = ["Layer"]
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__(config)
        recipe = config.composable_config
        self.uses_hidden_feedback = recipe["backbone"].get("schedule") == "stepwise"
        self.target_layer_ids = recipe["context"]["target_layer_ids"]
        self.block_size = config.block_size
        self.mask_token_id = recipe.get("mask_token_id")
        execution = config.runtime_config()
        self.context = build_context(execution)
        self.backbone = build_backbone(execution)
        self.proposal = build_proposal(execution)
        self.post_init()
        for module in self.modules():
            reset = getattr(module, "reset_draft_parameters", None)
            if reset is not None:
                reset()

    def forward(self, inputs, *, shared_ops=None):
        if self.uses_hidden_feedback:
            raise ValueError(
                "EAGLE3 uses forward_hidden_feedback or extend_hidden_feedback, not block proposal forward"
            )
        if inputs.mode not in ("features", "teacher_forcing", "sample"):
            raise ValueError("mode must be features, teacher_forcing, or sample")
        if inputs.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.training and inputs.draft_state is not None:
            raise ValueError("Persistent inference state is not accepted in training")
        if inputs.draft_state is not None and inputs.anchor_positions.size(1) != 1:
            raise ValueError("Cached inference expects one current block per sample")
        context = self.context(inputs)
        features = self.backbone(context)
        output = self.proposal(features, context, shared_ops)
        output.backbone_hidden_states = features.draft_hidden_states
        output.indexer_aux = features.indexer_aux
        return output

    @property
    def ttt_length(self):
        return self.block_size - 1

    def forward_hidden_feedback(
        self,
        target_hidden_states,
        input_ids,
        valid,
        anchor_positions,
        shared_ops,
        *,
        packed_plan=None,
    ):
        if not self.uses_hidden_feedback:
            raise ValueError("forward_hidden_feedback requires EAGLE3")
        hidden = self.context.project(target_hidden_states)
        if packed_plan is not None:
            outputs = []
            for sample in range(packed_plan.num_samples):
                start, end = packed_plan.context_offsets[sample : sample + 2].tolist()
                anchors = anchor_positions[:, packed_plan.block_sample_ids == sample]
                # Every sample retains its own recurrent context and TTT state.
                outputs.append(
                    rollout_hidden_feedback(
                        self.backbone,
                        hidden[:, start:end],
                        input_ids[:, start:end],
                        valid[:, start:end],
                        shared_ops.embed_tokens,
                        anchors,
                    )
                )
            return torch.cat(outputs, dim=1)
        return rollout_hidden_feedback(
            self.backbone,
            hidden,
            input_ids,
            valid,
            shared_ops.embed_tokens,
            anchor_positions,
        )

    def score_hidden_feedback(self, hidden, shared_ops, *, projection_rows=None):
        normalized = self.backbone.norm(hidden)
        return self.proposal.score_hidden(
            normalized, shared_ops, projection_rows=projection_rows
        )

    def extend_hidden_feedback(
        self, hidden, token_ids, state, shared_ops, *, target_features=False
    ):
        if target_features:
            hidden = self.context.project(hidden)
        return extend_hidden_feedback(
            self.backbone, hidden, shared_ops.embed_tokens(token_ids), state
        )

    def spec_generate(
        self,
        target,
        input_ids,
        max_new_tokens,
        stop_token_ids=(),
        temperature=0.0,
        top_p=1.0,
        *,
        sample_id="request",
        seed=42,
    ):
        from legospec.inference.generate import generate

        self.eval()
        result = generate(
            target,
            self,
            input_ids,
            sample_id,
            temperature,
            stop_token_ids,
            max_new_tokens,
            top_p,
            seed,
        )
        self._last_decode_stats = result
        return torch.cat(
            (input_ids, input_ids.new_tensor([result["generated_ids"]])), 1
        )

    def get_last_decode_stats(self):
        return dict(self._last_decode_stats)
