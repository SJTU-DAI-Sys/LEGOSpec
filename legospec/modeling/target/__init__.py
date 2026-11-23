"""Training target-model backends.

SGLang is imported lazily by ``training_target_model`` so the HF backend remains
usable without the optional training dependency.
"""

from legospec.modeling.target.training_target_model import (
    TargetObjectiveOutput,
    TrainingTargetModel,
    get_training_target_model,
)

__all__ = [
    "TargetObjectiveOutput",
    "TrainingTargetModel",
    "HFTrainingTargetModel",
    "SGLangTrainingTargetModel",
    "get_training_target_model",
]

from .hf import HFTrainingTargetModel
from .sglang import SGLangTrainingTargetModel
