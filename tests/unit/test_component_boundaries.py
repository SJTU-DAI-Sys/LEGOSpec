"""Dependency direction and standalone exports must survive component extraction."""
import ast
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest
import torch
from transformers import Qwen3Config

ROOT = Path(__file__).resolve().parents[2]
DRAFT = ROOT / 'litespec/modeling/draft'
ALGORITHMS = {'dflash', 'dflash2', 'dspark', 'dlite'}
COMPONENTS = {'attention', 'convolution', 'selector', 'markov', 'sequential', 'context_utils', 'kernels'}


def test_components_and_composable_do_not_import_algorithms():
    paths = list((DRAFT / 'composable').rglob('*.py')) + [DRAFT / (n+'.py') for n in COMPONENTS]
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split('.')[-1] not in ALGORITHMS, (path, node.lineno)
            elif isinstance(node, ast.Import):
                assert not any(a.name.split('.')[-1] in ALGORITHMS for a in node.names), path


def test_compatibility_exports_point_to_component_owners():
    records = json.loads((ROOT/'tests/reference/component_extraction.json').read_text())
    for row in records:
        old = importlib.import_module('litespec.modeling.draft.'+Path(row['source']).stem)
        new = importlib.import_module('litespec.modeling.draft.'+Path(row['destination']).stem)
        assert getattr(old,row['symbol']) is getattr(new,row['symbol'])


def tiny_model(algorithm):
    from litespec.modeling.draft.registry import resolve_draft
    architecture = {'dflash':'DFlashDraftModel','dflash2':'DFlash2DraftModel',
                    'dspark':'DSparkDraftModel','dlite':'DLiteDraftModel'}[algorithm]
    recipe = dict(target_layer_ids=[1], mask_token_id=0)
    if algorithm == 'dflash2':
        recipe.update(conv_kernel_size=2,conv_group_size=4,selector_rank=4,selector_top_k=3)
    elif algorithm == 'dspark':
        recipe.update(markov_rank=4,markov_head_type='rnn')
    elif algorithm == 'dlite':
        recipe.update(architecture_version='dlite_v1',model_role='student',chs_num_layers=1,
                      sequential_head='rnn',sequential_rank=4)
    config = Qwen3Config(hidden_size=16,intermediate_size=32,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=1,head_dim=8,vocab_size=31,
        num_target_layers=4,block_size=4,architectures=[architecture],
        **{algorithm+'_config':recipe})
    config._attn_implementation = 'eager'
    torch.manual_seed(42)
    return resolve_draft(architecture)(config).eval()


@pytest.mark.parametrize('algorithm', sorted(ALGORITHMS))
def test_algorithm_export_loads_outside_checkout(algorithm):
    from litespec.training.checkpoint import export_draft
    model = tiny_model(algorithm)
    output = ROOT/'tests/artifacts/component_exports'
    output.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output) as folder:
        folder = Path(folder)
        export_draft(model, folder/'model')
        script = '''from transformers import AutoModel
import torch
m = AutoModel.from_pretrained('model', trust_remote_code=True, local_files_only=True)
torch.save(m.state_dict(), 'loaded.pt')
'''
        env = dict(os.environ, PYTHONPATH='', CUDA_VISIBLE_DEVICES='',
                   HF_MODULES_CACHE=str(folder/'hf_modules'),HF_HUB_OFFLINE='1')
        subprocess.run([sys.executable,'-c',script],cwd=folder,env=env,check=True,capture_output=True,text=True)
        actual = torch.load(folder/'loaded.pt',map_location='cpu',weights_only=True)
        expected = model.state_dict()
        assert actual.keys() == expected.keys()
        for name in actual:
            torch.testing.assert_close(actual[name],expected[name],rtol=0,atol=0)
