import time
from typing import Optional

import torch
from torch import nn
from transformers import DynamicCache
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config, Qwen3PreTrainedModel, Qwen3RotaryEmbedding

from .kernels import DEFAULT_DFLASH_KERNELS, DFlashKernels
from .flex_attention import compile_friendly_flex_attention
from .registry import method_config, register_draft
from .sampling import _sample_probs, _sampling_probs, sample

from .attention import (
    FULL_ATTENTION,
    SLIDING_ATTENTION,
    _VALID_DFLASH_LAYER_TYPES,
    resolve_dflash_attention_layout,
    resolve_dflash_attention_mode,
    _require_bool_config,
    _resolve_mla_rope_interleaved,
    validate_dflash_mla_config,
    validate_dflash_attention_config,
    _rope_config,
    apply_rotary_pos_emb,
    _rotate_half_interleaved,
    apply_mla_rope,
    _prepare_dflash_eager_mask,
    Qwen3DFlashAttentionBase,
    Qwen3DFlashAttention,
    Qwen3DFlashMLAAttention,
    _DFLASH_ATTENTION_CLASSES,
    Qwen3DFlashDecoderLayer,
)


from .context_utils import (
    target_lm_head,
    target_embed_tokens,
    build_target_layer_ids,
    extract_context_feature,
)


def _rejection_sample(
    draft_tokens: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_indices: torch.Tensor,
) -> tuple[int, torch.Tensor]:
    """Verify a sparse DFlash2 proposal without changing the target distribution."""

    gamma = draft_tokens.shape[1]
    target_token_probs = target_probs[:, :gamma].gather(-1, draft_tokens.unsqueeze(-1))[
        ..., 0
    ]
    draft_token_probs = (
        draft_probs * (draft_indices == draft_tokens.unsqueeze(-1))
    ).sum(-1)
    accepted = (
        (torch.rand_like(draft_token_probs) * draft_token_probs < target_token_probs)
        .to(torch.int32)
        .cumprod(-1)
        .sum(-1)[0]
        .item()
    )
    if accepted == gamma:
        return accepted, _sample_probs(target_probs[:, -1])[0]

    residual = target_probs[0, accepted].clone()
    residual.scatter_add_(
        0,
        draft_indices[0, accepted],
        -draft_probs[0, accepted],
    )
    residual.clamp_min_(0)
    total = residual.sum()
    residual = torch.where(
        total > 0,
        residual / total.clamp_min(torch.finfo(residual.dtype).tiny),
        target_probs[0, accepted],
    )
    return accepted, _sample_probs(residual.unsqueeze(0))[0]


@register_draft
class DFlashDraftModel(Qwen3PreTrainedModel):
    method_config_key = "dflash_config"
    config_class = Qwen3Config
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]
    decoder_layer_class = Qwen3DFlashDecoderLayer

    def __init__(
        self,
        config,
        dflash_kernels: Optional[DFlashKernels] = None,
    ) -> None:
        super().__init__(config)
        self.config = config
        if config.architectures != [type(self).__name__]:
            raise ValueError(
                f"{type(self).__name__} requires its own architecture/config"
            )
        self.layer_types, self.sliding_window = resolve_dflash_attention_layout(config)
        self.attention_mode = validate_dflash_attention_config(config)
        kernels = dflash_kernels or DEFAULT_DFLASH_KERNELS
        draft_config = method_config(config)
        block_size = config.block_size
        if not isinstance(block_size, int) or isinstance(block_size, bool):
            raise ValueError("Draft config.block_size must be an integer")
        self.block_size = block_size
        self.layers = nn.ModuleList(
            [
                self._build_decoder_layer(config, layer_idx, kernels)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.target_layer_ids = draft_config.get(
            "target_layer_ids",
            build_target_layer_ids(config.num_target_layers, config.num_hidden_layers),
        )
        self.norm = kernels.make_rms_norm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(
            _rope_config(config, self.attention_mode)
        )
        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = kernels.make_rms_norm(
            config.hidden_size, config.rms_norm_eps
        )
        self.mask_token_id = draft_config.get("mask_token_id", None)
        self.projector_type = draft_config.get("projector_type", None)
        self.pure_draft_prefix_len = draft_config.get("pure_draft_prefix_len", 0)
        self.shift_label = draft_config.get("shift_label", False)
        self._last_decode_stats = {}
        self._init_draft_head(config, draft_config)
        self.post_init()

    def _build_decoder_layer(
        self,
        config: Qwen3Config,
        layer_idx: int,
        kernels: DFlashKernels,
    ) -> nn.Module:
        """Build one backbone layer; architecture variants override this seam."""

        return self.decoder_layer_class(config, layer_idx, kernels)

    def _init_draft_head(self, config, draft_config: dict) -> None:
        del config, draft_config

    def apply_logits_head(
        self,
        base_logits: torch.Tensor,
        *,
        prev_token_ids: Optional[torch.Tensor] = None,
        prev_token_embeddings: Optional[torch.Tensor] = None,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        del prev_token_ids, prev_token_embeddings, hidden_states
        return base_logits

    def apply_markov_logits(
        self,
        base_logits: torch.Tensor,
        *,
        prev_token_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.apply_logits_head(
            base_logits,
            prev_token_ids=prev_token_ids,
            hidden_states=hidden_states,
        )

    def predict_confidence(
        self,
        hidden_states: torch.Tensor,
        *,
        prev_token_ids: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        del hidden_states, prev_token_ids
        return None

    def _sample_draft_tokens(
        self,
        target: nn.Module,
        draft_hidden: torch.Tensor,
        block_output_ids: torch.LongTensor,
    ) -> torch.LongTensor:
        """Sample one speculative block from the draft-model hidden states.

        DFlash predicts the whole suffix in one LM-head call. Draft families
        with an auxiliary logits head can override this boundary without
        duplicating the target-cache and acceptance logic in ``spec_generate``.
        """
        del block_output_ids
        draft_logits = target_lm_head(target)(
            draft_hidden[:, -self.block_size + 1 :, :]
        )
        return sample(draft_logits)

    def _propose_draft_tokens(
        self,
        target: nn.Module,
        draft_hidden: torch.Tensor,
        block_output_ids: torch.LongTensor,
        temperature: float,
    ) -> tuple[torch.LongTensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return tokens and optional sparse proposal probabilities.

        Plain DFlash-family models retain their existing deterministic proposal.
        DFlash2 overrides this hook to return the selector distribution required
        by lossless rejection sampling.
        """

        del temperature
        return (
            self._sample_draft_tokens(target, draft_hidden, block_output_ids),
            None,
            None,
        )

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[object] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer_type, layer in zip(self.layer_types, self.layers):
            layer_attention_mask = (
                attention_mask[layer_type]
                if isinstance(attention_mask, dict)
                else attention_mask
            )
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=layer_attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(hidden_states)

    @torch.inference_mode()
    def spec_generate(
        self,
        target: nn.Module,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: list[int],
        temperature: float,
        top_p: float = 1.0,
        top_k: int = 0,
        target_cache: Optional[Cache] = None,
    ):
        if temperature < 0:
            raise ValueError(f"temperature must be non-negative, got {temperature}")
        if not 0 < top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {top_p}")
        if top_k < 0:
            raise ValueError(f"top_k must be non-negative, got {top_k}")
        self.eval()
        num_input_tokens = input_ids.shape[1]
        max_length = num_input_tokens + max_new_tokens

        block_size = self.block_size
        output_ids = torch.full(
            (1, max_length + block_size),
            self.mask_token_id,
            dtype=torch.long,
            device=target.device,
        )
        position_ids = torch.arange(
            output_ids.shape[1], device=target.device
        ).unsqueeze(0)

        past_key_values_target = (
            DynamicCache() if target_cache is None else target_cache
        )
        past_key_values_draft = DynamicCache()

        # Prefill stage
        output = target(
            input_ids,
            position_ids=position_ids[:, :num_input_tokens],
            past_key_values=past_key_values_target,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True,
        )

        output_ids[:, :num_input_tokens] = input_ids
        output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(
            output.logits, temperature, top_p, top_k
        )
        target_hidden = extract_context_feature(
            output.hidden_states, self.target_layer_ids
        )

        # Decode stage
        acceptance_lengths = []
        start = input_ids.shape[1]
        if input_ids.device.type == "cuda":
            torch.cuda.synchronize(input_ids.device)
        decode_started = time.perf_counter()
        while start < max_length:
            block_output_ids = output_ids[:, start : start + block_size].clone()
            block_position_ids = position_ids[:, start : start + block_size]
            noise_embedding = target_embed_tokens(target)(block_output_ids)
            draft_hidden = self(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids[
                    :, past_key_values_draft.get_seq_length() : start + block_size
                ],
                past_key_values=past_key_values_draft,
                use_cache=True,
                is_causal=False,
            )
            past_key_values_draft.crop(start)
            draft_tokens, draft_probs, draft_indices = self._propose_draft_tokens(
                target,
                draft_hidden,
                block_output_ids,
                temperature,
            )
            block_output_ids[:, 1:] = draft_tokens

            output = target(
                block_output_ids,
                position_ids=block_position_ids,
                past_key_values=past_key_values_target,
                use_cache=True,
                output_hidden_states=True,
            )

            if temperature > 0 and draft_probs is not None:
                if draft_indices is None:
                    raise RuntimeError(
                        "sparse draft probabilities require matching candidate indices"
                    )
                target_probs = _sampling_probs(
                    output.logits,
                    temperature,
                    top_p,
                    top_k,
                )
                acceptance_length, bonus = _rejection_sample(
                    draft_tokens,
                    target_probs,
                    draft_probs,
                    draft_indices,
                )
            else:
                posterior = sample(output.logits, temperature, top_p, top_k)
                acceptance_length = (
                    (block_output_ids[:, 1:] == posterior[:, :-1])
                    .cumprod(dim=1)
                    .sum(dim=1)[0]
                    .item()
                )
                bonus = posterior[:, acceptance_length][0]
            output_ids[:, start : start + acceptance_length + 1] = block_output_ids[
                :, : acceptance_length + 1
            ]
            output_ids[:, start + acceptance_length + 1] = bonus
            start += acceptance_length + 1
            past_key_values_target.crop(start)
            target_hidden = extract_context_feature(
                output.hidden_states, self.target_layer_ids
            )[:, : acceptance_length + 1, :]
            acceptance_lengths.append(acceptance_length + 1)
            if stop_token_ids is not None and any(
                stop_token_id in output_ids[:, num_input_tokens:]
                for stop_token_id in stop_token_ids
            ):
                break
        output_ids = output_ids[:, :max_length]
        output_ids = output_ids[:, output_ids[0] != self.mask_token_id]
        if stop_token_ids is not None:
            stop_token_ids = torch.tensor(stop_token_ids, device=output_ids.device)
            stop_token_indices = torch.isin(
                output_ids[0][num_input_tokens:], stop_token_ids
            ).nonzero(as_tuple=True)[0]
            if stop_token_indices.numel() > 0:
                output_ids = output_ids[
                    :, : num_input_tokens + stop_token_indices[0] + 1
                ]

        if input_ids.device.type == "cuda":
            torch.cuda.synchronize(input_ids.device)
        self._last_decode_stats = {
            "accept_lengths": acceptance_lengths,
            "decode_wall_time": time.perf_counter() - decode_started,
        }
        return output_ids

    def get_last_decode_stats(self) -> dict:
        return dict(self._last_decode_stats)
