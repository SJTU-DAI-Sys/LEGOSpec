"""Target feature contract and backend construction, independent of training."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class TargetObjectiveOutput:
    hidden_states: dict[int, torch.Tensor]
    logits: Optional[
        torch.Tensor
    ]  # [batch, seq_len, vocab], optional teacher projection
    input_ids: torch.Tensor  # [batch, seq_len]
    attention_mask: torch.Tensor  # [batch, seq_len]
    loss_mask: torch.Tensor  # [batch, seq_len]
    final_hidden_state: Optional[torch.Tensor] = None  # post-final-norm teacher feature


class TrainingTargetModel(ABC):
    """
    Abstract backend for extracting target-model training features.
    """

    def __init__(self):
        self.capture_layer_ids = None
        self._components = None

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> "TrainingTargetModel":
        """Initialize the target model backend."""

    @abstractmethod
    def generate_training_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        return_logits: bool = True,
    ) -> TargetObjectiveOutput:
        """Generate target hidden states and optional logits for draft training."""

    def set_capture_layers(self, layer_ids: List[int]) -> None:
        """Set which layers' hidden states to capture."""
        self.capture_layer_ids = layer_ids

    @abstractmethod
    def build_shared_components(self, mask_token_id):
        """Frozen views of resident embedding/head, retaining input gradients."""


def get_training_target_model(
    pretrained_model_name_or_path: str,
    backend: str = "sglang",
    torch_dtype=None,
    device=None,
    cache_dir=None,
    capture_layer_ids=None,
    **kwargs,
) -> TrainingTargetModel:
    from legospec.modeling.config_utils import load_text_model_config
    from legospec.modeling.target.adapters.base import get_model_adapter

    config = load_text_model_config(
        pretrained_model_name_or_path,
        cache_dir=cache_dir,
        trust_remote_code=kwargs.get("trust_remote_code", False),
    )
    if capture_layer_ids is None:
        capture_layer_ids = list(range(config.num_hidden_layers))
    else:
        capture_layer_ids = sorted(capture_layer_ids)
    adapter = get_model_adapter(config)
    adapter.configure_backend(backend, kwargs)
    if backend == "sglang":
        from legospec.modeling.target.sglang import SGLangTrainingTargetModel

        backend_class = SGLangTrainingTargetModel
    elif backend == "hf":
        from legospec.modeling.target.hf import HFTrainingTargetModel

        backend_class = HFTrainingTargetModel
    else:
        raise ValueError(f"Invalid backend: {backend}")
    return backend_class.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=torch_dtype,
        device=device,
        cache_dir=cache_dir,
        capture_layer_ids=capture_layer_ids,
        adapter=adapter,
        **kwargs,
    )
