from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from legospec.modeling.target.adapters.base import TargetModelAdapter
from legospec.modeling.target.training_target_model import (
    TargetObjectiveOutput,
    TrainingTargetModel,
)


class HFTrainingTargetModel(TrainingTargetModel):
    def __init__(self, model: nn.Module, *, adapter=None, capture_layer_ids=None):
        super().__init__()
        self.model = model
        self.adapter = adapter or TargetModelAdapter(getattr(model, "config", None))
        self.capture_layer_ids = capture_layer_ids

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = True,
        capture_layer_ids=None,
        adapter=None,
        **kwargs,
    ) -> "HFTrainingTargetModel":

        target_model = (
            AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                cache_dir=cache_dir,
                output_hidden_states=True,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )
            .requires_grad_(False)
            .eval()
        )

        if device:
            target_model = target_model.to(device)

        return cls(target_model, adapter=adapter, capture_layer_ids=capture_layer_ids)

    @torch.no_grad()
    def generate_training_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        return_logits: bool = True,
    ) -> TargetObjectiveOutput:
        device = next(self.model.parameters()).device
        input_ids, attention_mask, loss_mask = (
            tensor.to(device, non_blocking=True)
            for tensor in (input_ids, attention_mask, loss_mask)
        )
        backbone = self.adapter.hf_backbone(self.model)
        forward_model = self.model if return_logits else backbone
        norm = self.adapter.final_norm(backbone)
        pre_final = {}

        def capture_pre_norm(module, args):
            pre_final["hidden"] = args[0]

        handle = norm.register_forward_pre_hook(capture_pre_norm)
        try:
            outputs = forward_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        finally:
            handle.remove()
        if "hidden" not in pre_final:
            raise RuntimeError("HF target forward did not execute its final norm")

        # hidden_states[0] = embedding output; hidden_states[i+1] = layer i output
        # Take all layers except embedding (index 0) and concat in feature dim
        normalized_final = outputs.hidden_states[-1]
        all_hidden_states = outputs.hidden_states[1:-1] + (pre_final["hidden"],)
        capture_ids = self.capture_layer_ids
        if capture_ids is None:
            capture_ids = range(len(all_hidden_states))
        hidden_states = {
            layer_id: all_hidden_states[layer_id] for layer_id in capture_ids
        }

        return TargetObjectiveOutput(
            hidden_states=hidden_states,
            logits=outputs.logits if return_logits else None,
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            final_hidden_state=normalized_final,
        )

    def build_shared_components(self, mask_token_id):
        from legospec.modeling.target.components import (
            FlattenedLMHead,
            SharedTargetEmbeddingsAndHead,
        )

        if self._components is None:
            model = self.model.requires_grad_(False).eval()
            head = model.get_output_embeddings()
            if isinstance(head, nn.Linear):
                with torch.device("meta"):
                    shared_head = FlattenedLMHead(
                        head.in_features,
                        head.out_features,
                        bias=head.bias is not None,
                        softcap=self.adapter.softcap,
                    )
                shared_head.weight, shared_head.bias = head.weight, head.bias
                head = shared_head
            self._components = (
                SharedTargetEmbeddingsAndHead(
                    model.get_input_embeddings(),
                    head,
                )
                .requires_grad_(False)
                .eval()
            )
        return self._components

