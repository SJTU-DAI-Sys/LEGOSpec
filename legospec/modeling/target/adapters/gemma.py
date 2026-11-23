"""Gemma text semantics; execution and TP handling remain in the backends."""

import torch
from torch import nn

from .base import TargetModelAdapter


class GemmaTokenEmbedding(nn.Embedding):
    def forward(self, input):
        embeddings = super().forward(input)
        return embeddings * embeddings.new_tensor(self.embedding_dim**0.5)


class GemmaTargetAdapter(TargetModelAdapter):
    captures_final_layer = True
    supports_final_norm_recompute = True
    softcap_fp32 = True

    def draft_config_values(self):
        values = self.config.to_dict()
        values.pop("per_layer_config", None)
        values["rope_parameters"] = values["rope_parameters"]["full_attention"]
        values["hidden_act"] = values.pop("hidden_activation")
        return values

    def configure_backend(self, backend, kwargs):
        if backend == "sglang" and kwargs.get("attention_backend") not in (
            "triton",
            "fa4",
            "trtllm_mha",
            "ascend",
            "intel_xpu",
            "intel_amx",
            "aiter",
        ):
            kwargs["attention_backend"] = "triton"

    def sglang_components(self, model):
        # The native module already scales by sqrt(H); share it unchanged.
        return model.language_model.embed_tokens, model.lm_head

    def new_embedding(self):
        return GemmaTokenEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            padding_idx=getattr(self.config, "pad_token_id", None),
        )

    def load_final_norm(self, model_path):
        from pathlib import Path

        from safetensors import safe_open

        with safe_open(
            Path(model_path) / "model.safetensors", framework="pt", device="cpu"
        ) as checkpoint:
            weight = checkpoint.get_tensor("model.language_model.norm.weight").to(
                dtype=torch.float32, copy=True
            )
        return GemmaFinalNorm(weight, self.config.rms_norm_eps)


class GemmaFinalNorm(nn.Module):
    """Reconstruct normalized teacher features from cached decoder residuals."""

    def __init__(self, weight, eps):
        super().__init__()
        self.register_buffer("weight", weight)
        self.eps = eps

    def forward(self, residual):
        return nn.functional.rms_norm(
            residual.float(),
            (self.weight.numel(),),
            self.weight,
            self.eps,
        ).to(residual.dtype)
