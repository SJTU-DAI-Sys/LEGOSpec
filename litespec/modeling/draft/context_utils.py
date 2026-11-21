"""Shared context utils primitives; independent of draft algorithm implementations.

Extracted without changing numerical operations or parameter names.
Original source modules and AST hashes are recorded in tests/reference/component_extraction.json.
"""
from typing import Optional
import torch
from torch import nn


def target_lm_head(target: nn.Module) -> nn.Module:
    """Resolve the output projection across Qwen3 and Qwen3.5 wrappers."""

    getter = getattr(target, "get_output_embeddings", None)
    head = getter() if callable(getter) else None
    if head is None:
        head = getattr(target, "lm_head", None)
    if head is None:
        raise ValueError("Target model does not expose an output LM head")
    return head


def target_embed_tokens(target: nn.Module) -> nn.Module:
    """Resolve the token embedding across Qwen3 and Qwen3.5 wrappers."""

    getter = getattr(target, "get_input_embeddings", None)
    embedding = getter() if callable(getter) else None
    if embedding is None:
        embedding = getattr(getattr(target, "model", None), "embed_tokens", None)
    if embedding is None:
        raise ValueError("Target model does not expose input token embeddings")
    return embedding


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int):
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start = 1
    end = num_target_layers - 3
    span = end - start
    target_layer_ids = [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]
    return target_layer_ids


def extract_context_feature(
    hidden_states: list[torch.Tensor],
    layer_ids: Optional[list[int]],
) -> torch.Tensor:
    offset = 1
    selected_states = []
    for layer_id in layer_ids:
        selected_states.append(hidden_states[layer_id + offset])
    target_hidden = torch.cat(selected_states, dim=-1)
    return target_hidden

