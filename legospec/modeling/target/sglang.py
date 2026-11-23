"""SGLang native target execution shared by online training and offline capture."""

import torch

from legospec.modeling.target.training_target_model import (
    TargetObjectiveOutput,
    TrainingTargetModel,
)


class SGLangTrainingTargetModel(TrainingTargetModel):
    def __init__(self, model_runner, *, adapter, engine):
        super().__init__()
        self.model_runner, self.adapter, self.engine = model_runner, adapter, engine
        self.capture_layer_ids = list(engine.layer_ids)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *,
        capture_layer_ids,
        adapter,
        torch_dtype=None,
        device=None,
        cache_dir=None,
        trust_remote_code=False,
        **kwargs,
    ):
        from legospec.modeling.target.sglang_backend.capture import NativeCapture

        engine = NativeCapture(
            pretrained_model_name_or_path,
            capture_layer_ids=capture_layer_ids,
            hidden_size=adapter.config.hidden_size,
            adapter=adapter,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        return cls(engine.runner, adapter=adapter, engine=engine)

    def set_capture_layers(self, layer_ids):
        self.capture_layer_ids = sorted(layer_ids)
        self.engine.layer_ids = self.capture_layer_ids
        self.adapter.install_capture(self.model_runner.model, self.capture_layer_ids)

    def generate_capture_data(self, records):
        return self.engine.generate_capture_data(records)

    @torch.no_grad()
    def generate_training_data(
        self, input_ids, attention_mask, loss_mask, return_logits=True
    ):
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError(
                "Native prefill expects matching batch-first token IDs and masks"
            )
        width = input_ids.shape[1]
        records = []
        for index, (ids, mask) in enumerate(
            zip(input_ids.tolist(), attention_mask.tolist())
        ):
            length = width
            while length > 1 and not mask[length - 1]:
                length -= 1
            records.append(
                dict(source_index=index, input_ids=torch.tensor(ids[:length]))
            )
        captured = self.engine.generate_capture_data(records)
        self.engine.last_token_logits = getattr(captured, "last_token_logits", None)

        def restore(rows):
            # The capture engine uses inference_mode. A fresh tensor outside it
            # is required before trainable projections save teacher features.
            result = rows.new_zeros((len(records), width, rows.shape[-1]))
            for index, (start, stop) in enumerate(
                zip(captured.token_offsets, captured.token_offsets[1:])
            ):
                result[index, : stop - start].copy_(rows[start:stop])
            return result

        hidden = {
            layer: restore(captured.hidden_states[layer])
            for layer in self.capture_layer_ids
        }
        final = restore(captured.final_hidden_state)
        logits = (
            self.build_shared_components(0).lm_head(final) if return_logits else None
        )
        return TargetObjectiveOutput(
            hidden_states=hidden,
            logits=logits,
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            final_hidden_state=final,
        )

    def build_shared_components(self, mask_token_id):
        from legospec.modeling.target.components import build_sglang_components

        if not 0 <= mask_token_id < self.adapter.config.vocab_size:
            raise ValueError("MASK token is outside the target vocabulary")
        if self._components is None:
            self._components = build_sglang_components(
                self.model_runner.model,
                self.adapter,
                mask_token_id,
            )
        return self._components

