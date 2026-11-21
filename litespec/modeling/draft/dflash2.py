"""DFlash 2 draft architecture.

DFlash 2 keeps the one-pass DFlash backbone and adds two checkpoint-driven
components used by the serving implementation:

* a grouped, dynamic depthwise convolution around every attention and MLP;
* a low-rank candidate selector that re-ranks the target head's top-k tokens.

The module and parameter names intentionally match SGLang's
``DFlash2DraftModel`` so a normal Hugging Face export can be served directly.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from .dflash import DFlashDraftModel
from .attention import Qwen3DFlashDecoderLayer
from .context_utils import target_lm_head
from .kernels import DFlashKernels
from .registry import register_draft


from .convolution import (
    DFlashGroupedConv,
    Qwen3DFlash2DecoderLayer,
)


from .selector import (
    CandidateSelector,
)


@register_draft
class DFlash2DraftModel(DFlashDraftModel):
    method_config_key = "dflash2_config"
    """DFlash backbone with local convolution and candidate path selection."""

    _no_split_modules = ["Qwen3DFlash2DecoderLayer"]
    decoder_layer_class = Qwen3DFlash2DecoderLayer

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        super()._init_weights(module)
        if isinstance(module, DFlashGroupedConv):
            # DFlashDraftModel.post_init() reinitializes child Linear modules.
            # Restore the additive dynamic-kernel branch to an exact no-op after
            # that pass; checkpoint loading can still overwrite these weights.
            nn.init.zeros_(module.kernel_projection.weight)

    def _dflash2_config(self) -> dict:
        return dict(getattr(self.config, "dflash2_config", None) or {})

    def _build_decoder_layer(
        self,
        config: Qwen3Config,
        layer_idx: int,
        kernels: DFlashKernels,
    ) -> nn.Module:
        method_config = dict(getattr(config, "dflash2_config", None) or {})
        taps = method_config.get("conv_kernel_size")
        group_size = method_config.get("conv_group_size")
        if not isinstance(taps, int) or isinstance(taps, bool):
            raise ValueError(
                "DFlash2DraftModel requires dflash2_config.conv_kernel_size"
            )
        if not isinstance(group_size, int) or isinstance(group_size, bool):
            raise ValueError(
                "DFlash2DraftModel requires dflash2_config.conv_group_size"
            )

        def grouped_conv() -> DFlashGroupedConv:
            return DFlashGroupedConv(
                hidden_size=int(config.hidden_size),
                block_size=self.block_size,
                taps=taps,
                group_size=group_size,
            )

        return self.decoder_layer_class(
            config,
            layer_idx,
            kernels,
            attention_conv=grouped_conv(),
            mlp_conv=grouped_conv(),
        )

    def _init_draft_head(self, config: Qwen3Config, dflash2_config: dict) -> None:
        selector_rank = dflash2_config.get("selector_rank")
        selector_top_k = dflash2_config.get("selector_top_k")
        if not isinstance(selector_rank, int) or isinstance(selector_rank, bool):
            raise ValueError("DFlash2DraftModel requires dflash2_config.selector_rank")
        if not isinstance(selector_top_k, int) or isinstance(selector_top_k, bool):
            raise ValueError("DFlash2DraftModel requires dflash2_config.selector_top_k")
        self.candidate_selector = CandidateSelector(
            hidden_size=int(config.hidden_size),
            vocab_size=int(config.vocab_size),
            state_rank=selector_rank,
            top_k=selector_top_k,
            initializer_range=float(config.initializer_range),
        )

    def transform_unary_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply the public DFlash2 unary-logit transform used by SGLang."""

        method_config = self._dflash2_config()
        transformed = logits.float() * float(
            method_config.get("output_multiplier", 1.0)
        )
        softcap = method_config.get("final_logit_softcapping")
        if softcap is not None:
            softcap = float(softcap)
            if softcap <= 0:
                raise ValueError("DFlash2 final_logit_softcapping must be > 0")
            transformed = torch.tanh(transformed / softcap) * softcap
        return transformed

    def _sample_draft_tokens(
        self,
        target: nn.Module,
        draft_hidden: torch.Tensor,
        block_output_ids: torch.LongTensor,
    ) -> torch.LongTensor:
        tokens, _, _ = self._propose_draft_tokens(
            target,
            draft_hidden,
            block_output_ids,
            temperature=0.0,
        )
        return tokens

    def _propose_draft_tokens(
        self,
        target: nn.Module,
        draft_hidden: torch.Tensor,
        block_output_ids: torch.LongTensor,
        temperature: float,
    ) -> tuple[torch.LongTensor, Optional[torch.Tensor], torch.LongTensor]:
        """Return the official sparse selector proposal for verification."""

        draft_prediction_hidden = draft_hidden[:, -self.block_size + 1 :, :]
        unary_logits = self.transform_unary_logits(target_lm_head(target)(draft_prediction_hidden))
        unary_topk, candidate_ids = unary_logits.topk(
            self.candidate_selector.top_k,
            dim=-1,
        )
        token_ids, probabilities = self.candidate_selector.select_path(
            candidate_ids=candidate_ids,
            unary_logits=unary_topk,
            hidden_states=draft_prediction_hidden,
            anchor_token_ids=block_output_ids[:, 0],
            temperature=temperature,
        )
        return token_ids, probabilities, candidate_ids


__all__ = [
    "CandidateSelector",
    "DFlash2DraftModel",
    "DFlashGroupedConv",
    "Qwen3DFlash2DecoderLayer",
]
