# coding=utf-8
"""DSpark draft model entry point and Markov heads."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .dflash import DFlashDraftModel
from .attention import resolve_dflash_attention_mode
from .context_utils import target_lm_head
from .convolution import DFlashGroupedConv, Qwen3DFlash2DecoderLayer
from .registry import register_draft
from .sampling import sample


from .markov import (
    _sample,
    AcceptRatePredictor,
    VanillaMarkovHead,
    GatedMarkovHead,
    RNNMarkovHead,
    build_markov_head,
)


@register_draft
class DSparkDraftModel(DFlashDraftModel):
    method_config_key = "dspark_config"
    """DFlash backbone with DSpark Markov/confidence heads."""

    expected_projector_type = "dspark"
    _no_split_modules = ["Qwen3DFlashDecoderLayer", "Qwen3DFlash2DecoderLayer"]

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear) and getattr(
            module, "_dspark_conv_projection", False
        ):
            # Keep the extra projection at identity without advancing the RNG
            # used to initialize shared backbone/head parameters. Preserve a
            # projection already supplied by a loaded checkpoint.
            if not getattr(module.weight, "_is_hf_initialized", False):
                module.weight.zero_()
            return
        super()._init_weights(module)
        if isinstance(module, DFlashGroupedConv):
            # post_init initializes the child Linear after DFlashGroupedConv's
            # constructor. Preserve loaded checkpoint parameters.
            if not getattr(
                module.kernel_projection.weight, "_is_hf_initialized", False
            ):
                module.kernel_projection.weight.zero_()

    def _build_decoder_layer(self, config, layer_idx, kernels) -> nn.Module:
        method_config = dict(getattr(config, "dspark_config", None) or {})
        enabled = method_config.get("dspark_backbone_conv_enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError(
                "dspark_config.dspark_backbone_conv_enabled must be boolean"
            )
        if not enabled:
            return super()._build_decoder_layer(config, layer_idx, kernels)

        taps = method_config.get("conv_kernel_size")
        group_size = method_config.get("conv_group_size")
        if not isinstance(taps, int) or isinstance(taps, bool):
            raise ValueError(
                "DSpARK full convolution requires dspark_config.conv_kernel_size"
            )
        if not isinstance(group_size, int) or isinstance(group_size, bool):
            raise ValueError(
                "DSpARK full convolution requires dspark_config.conv_group_size"
            )

        def grouped_conv() -> DFlashGroupedConv:
            # nn.Linear.reset_parameters in the convolution constructor draws
            # random numbers despite the projection being zero-initialized.
            # Restore RNG state so a paired same-seed run starts with exactly
            # the same non-convolution weights as the baseline.
            with torch.random.fork_rng(devices=[]):
                conv = DFlashGroupedConv(
                    hidden_size=int(config.hidden_size),
                    block_size=self.block_size,
                    taps=taps,
                    group_size=group_size,
                )
            conv.kernel_projection._dspark_conv_projection = True
            return conv

        return Qwen3DFlash2DecoderLayer(
            config,
            layer_idx,
            kernels,
            attention_conv=grouped_conv(),
            mlp_conv=grouped_conv(),
        )

    def __init__(self, config) -> None:
        dspark_config = dict(getattr(config, "dspark_config", None) or {})
        attention_mode = resolve_dflash_attention_mode(config)
        num_heads = int(config.num_attention_heads)
        num_kv_heads = int(config.num_key_value_heads)
        # MLA carries its own head geometry; the query/KV head-count policy
        # below only constrains the GQA/MHA projections.
        if attention_mode != "mla" and num_heads % num_kv_heads:
            raise ValueError(
                "DSpark requires num_key_value_heads to divide "
                f"num_attention_heads, got {num_kv_heads} and {num_heads}"
            )
        if attention_mode == "gqa" and num_kv_heads >= num_heads:
            raise ValueError(
                "DSpark defaults to GQA and requires num_key_value_heads < "
                "num_attention_heads; set dspark_config.attention_mode='mha' "
                "to opt into equal query/KV head counts"
            )
        # 'mha' head-count consistency is enforced by the shared
        # validate_dflash_attention_config at model init.
        dspark_config["attention_mode"] = attention_mode
        projector_type = dspark_config.get("projector_type")
        if projector_type is None:
            dspark_config["projector_type"] = self.expected_projector_type
            config.dspark_config = dspark_config
        elif projector_type != self.expected_projector_type:
            raise ValueError(
                "DSparkDraftModel requires dspark_config.projector_type='dspark'."
            )
        config.dspark_config = dspark_config
        super().__init__(config)

    def _sample_draft_tokens(
        self,
        target: nn.Module,
        draft_hidden: torch.Tensor,
        block_output_ids: torch.LongTensor,
    ) -> torch.LongTensor:
        """Generate a block with the same Markov correction used in training."""

        proposal_hidden = draft_hidden[:, -self.block_size + 1 :, :]
        base_logits = target_lm_head(target)(proposal_hidden)
        if self.markov_head is None:
            return _sample(base_logits)
        sampled_tokens, _ = self.markov_head.sample_block_tokens(
            base_logits,
            first_prev_token_ids=block_output_ids[:, 0],
            hidden_states=proposal_hidden,
        )
        return sampled_tokens

    def _init_draft_head(self, config, dspark_config: dict) -> None:
        self.markov_head = build_markov_head(config, dspark_config)
        confidence_alpha = float(dspark_config.get("confidence_head_alpha", 0.0) or 0.0)
        self.enable_confidence_head = bool(
            dspark_config.get("enable_confidence_head", confidence_alpha > 0.0)
        )
        self.confidence_head_with_markov = bool(
            dspark_config.get("confidence_head_with_markov", False)
        )
        if self.confidence_head_with_markov and self.markov_head is None:
            raise ValueError(
                "confidence_head_with_markov=True requires markov_rank > 0"
            )

        self.confidence_head = None
        if self.enable_confidence_head:
            input_dim = config.hidden_size
            if self.confidence_head_with_markov:
                input_dim += self.markov_head.markov_rank
            self.confidence_head = AcceptRatePredictor(input_dim=input_dim)

    def apply_logits_head(
        self,
        base_logits: torch.Tensor,
        *,
        prev_token_ids: Optional[torch.Tensor] = None,
        prev_token_embeddings: Optional[torch.Tensor] = None,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        del prev_token_embeddings
        if self.markov_head is None:
            return base_logits
        if prev_token_ids is None:
            raise ValueError("DSparkDraftModel requires prev_token_ids")
        return self.markov_head.apply_block_logits(
            base_logits,
            token_ids=prev_token_ids,
            hidden_states=hidden_states,
        )

    def predict_confidence(
        self,
        hidden_states: torch.Tensor,
        *,
        prev_token_ids: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if self.confidence_head is None:
            return None
        if self.confidence_head_with_markov:
            assert self.markov_head is not None
            if prev_token_ids is None:
                raise ValueError("prev_token_ids is required for Markov confidence")
            prev_embeddings = self.markov_head.get_prev_embeddings(prev_token_ids).to(
                hidden_states.dtype
            )
            hidden_states = torch.cat([hidden_states, prev_embeddings], dim=-1)
        return self.confidence_head(hidden_states).float()


__all__ = [
    "AcceptRatePredictor",
    "DSparkDraftModel",
    "GatedMarkovHead",
    "RNNMarkovHead",
    "VanillaMarkovHead",
    "build_markov_head",
]
