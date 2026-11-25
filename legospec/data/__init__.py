"""Conversation datasets and offline hidden-feature caches."""

from legospec.data.dataset import build_training_dataset
from legospec.data.hidden_cache import HiddenCacheError, load_hidden_cache_manifest
from legospec.data.hidden_dataset import (
    RegenFullCollator,
    RegenFullDataset,
    sample_metadata_has_valid_anchor,
)

__all__ = [
    "build_training_dataset",
    "HiddenCacheError",
    "load_hidden_cache_manifest",
    "RegenFullCollator",
    "RegenFullDataset",
    "sample_metadata_has_valid_anchor",
]
