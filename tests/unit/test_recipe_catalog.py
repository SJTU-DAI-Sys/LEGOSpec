"""Maintained public recipes must use the current four-axis schema."""

import json
from pathlib import Path

import pytest
from legospec import LegoSpecConfig

ROOT = Path(__file__).resolve().parents[2]
RECIPES = [
    p
    for p in sorted((ROOT / "configs").glob("*.json"))
    if "composable_config" in json.loads(p.read_text())
]


@pytest.mark.parametrize("path", RECIPES, ids=lambda p: p.stem)
def test_public_recipe_configuration(path):
    payload = json.loads(path.read_text())
    profile = payload["composable_config"]["attention_configs"].get("gqa", {})
    target = {
        key: profile[key]
        for key in ("num_attention_heads", "num_key_value_heads", "head_dim")
        if key in profile
    }
    config = LegoSpecConfig(**payload, num_target_layers=64, **target)
    assert config.architectures == ["LegoSpecModel"]
    assert config.model_type == "legospec"
    assert config.composable_config["backbone"]["type"] == "layer_stack"
    assert config.composable_config["context"]["type"] in (
        "concat_projection",
        "gated_fusion",
    )
