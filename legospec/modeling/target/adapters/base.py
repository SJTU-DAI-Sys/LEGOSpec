"""Default model semantics used by HF/SGLang and standalone components.

Adapters describe model structure and capture semantics; backends own execution,
TP communication, batching and resource lifetime. No training imports here.
"""

from types import SimpleNamespace

from torch import nn


class TargetModelAdapter:
    captures_final_layer = False
    supports_final_norm_recompute = False
    softcap_fp32 = False

    def __init__(self, config=None):
        self.config = getattr(config, "text_config", config)
        self._capture_handles = []

    @property
    def softcap(self):
        return getattr(self.config, "final_logit_softcapping", None)

    def load_final_norm(self, model_path):
        raise ValueError(
            "This target adapter does not support cached final norm reconstruction"
        )

    def draft_config_values(self):
        return self.config.to_dict()

    def configure_backend(self, backend, kwargs):
        pass

    def hf_backbone(self, model):
        backbone = getattr(model, "base_model", model)
        text = getattr(backbone, "language_model", backbone)
        return getattr(text, "model", text)

    def final_norm(self, backbone):
        for name in ("norm", "final_layernorm", "final_layer_norm", "ln_f"):
            norm = getattr(backbone, name, None)
            if isinstance(norm, nn.Module):
                return norm
        raise ValueError("Target adapter requires an identifiable final norm")

    def sglang_components(self, model):
        return model.get_input_embeddings(), model.lm_head

    def new_embedding(self):
        return nn.Embedding(
            self.config.vocab_size,
            self.config.hidden_size,
            padding_idx=getattr(self.config, "pad_token_id", None),
        )

    def configure_capture(self, model, layer_ids):
        model.set_eagle3_layers_to_capture(layer_ids)

    def install_capture(self, model, layer_ids):
        """Install instance-local hooks; replacing capture policy removes old hooks."""
        self.remove_capture()
        self.configure_capture(model, layer_ids)
        if self.captures_final_layer:
            return
        final_id = len(model.model.layers) - 1
        if final_id not in layer_ids:
            return
        captured = {}

        def capture_raw(module, args, kwargs):
            hidden = args[0] if args else kwargs["x"]
            residual = args[1] if len(args) > 1 else kwargs.get("residual")
            captured["raw"] = hidden.clone() if residual is None else hidden + residual

        self._capture_handles.append(
            self.final_norm(model.model).register_forward_pre_hook(
                capture_raw, with_kwargs=True
            )
        )

        def append_final(module, args, output):
            if isinstance(output, tuple):
                normalized, aux = output
                return normalized, [*aux, captured.pop("raw")]
            return output, [captured.pop("raw")]

        self._capture_handles.append(model.model.register_forward_hook(append_final))

    def remove_capture(self):
        for handle in self._capture_handles:
            handle.remove()
        self._capture_handles.clear()


def get_model_adapter(config):
    from legospec.modeling.config_utils import is_qwen35_model_type

    if isinstance(config, dict):
        outer_type = config.get("model_type", "")
        values = dict(config.get("text_config", config))
        values.setdefault("model_type", outer_type)
        config = SimpleNamespace(**values)
        if outer_type in {"gemma4_unified", "qwen3_5", "qwen3_5_moe"}:
            config.target_source_model_type = outer_type
    config = getattr(config, "text_config", config)
    model_type = getattr(
        config, "target_source_model_type", getattr(config, "model_type", "")
    )
    if model_type in {"gemma4_unified", "gemma4_unified_text"}:
        from .gemma import GemmaTargetAdapter

        return GemmaTargetAdapter(config)
    if is_qwen35_model_type(model_type):
        from .qwen35 import Qwen35TargetAdapter

        return Qwen35TargetAdapter(config)
    return TargetModelAdapter(config)
