"""Dependency direction and standalone exports must survive component extraction."""

import ast
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
DRAFT = ROOT / "legospec/modeling/draft"
ALGORITHMS = {"dflash", "dflash2", "dspark", "dlite"}


def test_components_and_composable_do_not_import_algorithms():
    for name in ALGORITHMS:
        assert not (DRAFT / (name + ".py")).exists()
    for path in (ROOT / "legospec").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module not in {
                    "legospec.modeling.draft." + name for name in ALGORITHMS
                }


def tiny_model(proposal):
    from legospec import LegoSpecConfig, LegoSpecModel

    recipe = {
        "context": {"type": "concat_projection", "target_layer_ids": [1]},
        "backbone": {
            "type": "layer_stack",
            "layers": [
                {'attention': {'family': 'dense', 'variant': 'gqa', 'access': 'full'}, 'local_operator': {'type': 'grouped_conv', 'kernel_size': 2, 'group_size': 4}}
            ],
        },
        "attention_configs": {
            "gqa": {"num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8}
        },
        "proposal": {"type": proposal},
        "objective": {"type": "hard_ce", "loss_decay_gamma": None},
    }
    if proposal != "parallel":
        recipe["proposal"]["rank"] = 4
    if proposal == "residual_selector":
        recipe["proposal"]["top_k"] = 3
        recipe["objective"] = {
            "type": "selector_losses",
            "loss_decay_gamma": None,
            "base": {"type": "hard_ce", "weight": 1.0},
            "final": {"type": "hard_ce", "weight": 1.0},
        }
    config = LegoSpecConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=31,
        num_target_layers=4,
        block_size=4,
        composable_config=recipe,
    )
    torch.manual_seed(42)
    return LegoSpecModel(config).eval()


@pytest.mark.parametrize(
    "algorithm", ["parallel", "rnn", "sequential_rnn", "residual_selector"]
)
def test_algorithm_export_loads_outside_checkout(algorithm):
    from legospec.training.runtime.checkpoint import export_draft

    model = tiny_model(algorithm)
    output = ROOT / "tests/artifacts/component_exports"
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output) as folder:
        folder = Path(folder)
        export_draft(model, folder / "model")
        script = """from transformers import AutoModel
import torch
m = AutoModel.from_pretrained('model', trust_remote_code=True, local_files_only=True)
torch.save(m.state_dict(), 'loaded.pt')
"""
        env = dict(
            os.environ,
            PYTHONPATH="",
            CUDA_VISIBLE_DEVICES="",
            HF_MODULES_CACHE=str(folder / "hf_modules"),
            HF_HUB_OFFLINE="1",
        )
        subprocess.run(
            [sys.executable, "-c", script],
            cwd=folder,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        actual = torch.load(folder / "loaded.pt", map_location="cpu", weights_only=True)
        expected = model.state_dict()
        assert actual.keys() == expected.keys()
        for name in actual:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
