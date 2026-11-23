"""Component ownership and the installed draft dependency boundary."""

import ast
from pathlib import Path

import pytest
from torch import nn

from legospec import LegoSpecConfig, LegoSpecModel


def test_model_components_do_not_import_training():
    root = Path(__file__).parents[2] / "legospec/modeling/draft"
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("legospec.training"), path
            elif isinstance(node, ast.Import):
                assert all(
                    not n.name.startswith("legospec.training") for n in node.names
                ), path


def test_component_initialization_owns_rnn_identity_prior():
    config = LegoSpecConfig(
        hidden_size=16,
        intermediate_size=32,
        vocab_size=31,
        num_target_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        composable_config={
            "context": {"type": "concat_projection", "target_layer_ids": [0, 2]},
            "backbone": {
                "type": "layer_stack",
                "layers": [
                    {
                        "attention": {
                            "family": "dense",
                            "variant": "gqa",
                            "access": "full",
                        }
                    }
                ]
            },
            "attention_configs": {"gqa": {}},
            "proposal": {"type": "rnn", "rank": 4},
            "objective": {"type": "hard_ce"},
        },
    )
    model = LegoSpecModel(config)
    assert isinstance(model.backbone.layers[0].attention, nn.Module)
    assert not model.proposal.markov_head.markov_w2.weight.count_nonzero()
    assert model.proposal.markov_head.markov_w1.weight.count_nonzero()


def test_string_attention_recipe_is_not_a_public_format():
    with pytest.raises((ValueError, TypeError)):
        LegoSpecConfig(
            composable_config={
                "context": {"type": "concat_projection"},
                "backbone": {"layers": [{"attention": "gqa", "access": "dense"}]},
                "attention_configs": {"gqa": {}},
                "proposal": {"type": "parallel"},
                "objective": {"type": "hard_ce"},
            }
        )
