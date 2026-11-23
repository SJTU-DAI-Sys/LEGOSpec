"""Qwen3.5/3.6 native SGLang layer-output capture."""

from .base import TargetModelAdapter


class Qwen35TargetAdapter(TargetModelAdapter):
    def configure_backend(self, backend, kwargs):
        if backend != "sglang":
            raise ValueError("Qwen3.5 targets require the SGLang backend")
        kwargs["attention_backend"] = "fa3"

    def configure_capture(self, model, layer_ids):
        # Native DFlash captures inputs to layer i+1; final output is hooked by
        # the base adapter before the final norm.
        for layer in model.model.layers:
            layer._is_layer_to_capture = False
        final_id = len(model.model.layers) - 1
        model.set_dflash_layers_to_capture([i for i in layer_ids if i != final_id])
