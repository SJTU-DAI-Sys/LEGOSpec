"""Production sparse attention matches the test-only dense-mask reference."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from dsa_reference import reference_attention
from legospec.modeling.draft.backbone.attention.access.dsa import (
    gather_attention,
)
from legospec.modeling.draft.contracts import SelectionView


def test_gather_matches_reference_outputs_and_gradients():
    torch.manual_seed(42)
    positions = torch.arange(6)[None]
    valid = torch.ones_like(positions, dtype=torch.bool)
    documents = torch.zeros_like(positions)
    indices = torch.tensor([[[0, 2], [1, 4]]])
    selection = SelectionView(
        indices=indices,
        selected_positions=indices,
        selected_valid=torch.ones_like(indices, dtype=torch.bool),
        scores=torch.zeros_like(indices, dtype=torch.float),
        context_positions=positions,
        context_valid=valid,
        context_documents=documents,
        anchor_valid=torch.ones(1, 2, dtype=torch.bool),
        chunk_size=1,
    )
    inputs = SimpleNamespace(
        selection=selection,
        anchor_positions=torch.tensor([[3, 6]]),
        block_keep_mask=torch.ones(1, 2, dtype=torch.bool),
        collect_indexer_teacher=True,
        prediction_mask=None,
    )
    module = SimpleNamespace(
        scaling=0.5, attention_dropout=0.0, training=True, o_proj=nn.Identity()
    )
    tensors = [
        torch.randn(1, 2, 6, 4),
        torch.randn(1, 1, 12, 4),
        torch.randn(1, 1, 12, 4),
    ]
    results = []
    for execute in (gather_attention, reference_attention):
        q, k, v = [t.clone().requires_grad_() for t in tensors]
        output, teacher = execute(module, q, k, v, inputs, positions, valid, documents)
        gradients = torch.autograd.grad(output.square().sum(), (q, k, v))
        results.append((output, teacher, *gradients))
    for actual, expected in zip(*results):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


def test_dsa_reference_is_not_a_public_execution_option():
    from legospec.modeling.draft import LegoSpecConfig

    config = LegoSpecConfig()
    for layer in config.composable_config["backbone"]["layers"]:
        layer["access"] = "dsa"
    config.composable_config["access_configs"] = {"dsa": {"execution": "reference"}}
    with pytest.raises(ValueError):
        config.validate_recipe()
