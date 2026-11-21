"""Audit a config-only SGLang export without modifying the publication files."""
import ast
import copy
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import struct
from numbers import Integral
from typing import Any, List, Optional

os.environ['SGLANG_USE_CPU_ENGINE'] = '1'
ROOT = Path(__file__).resolve().parent
PUBLISH = Path('/share/dai-sys/wanghanzhen/projects/DAC27/LEGOSpec_public/model_publish')
TARGET = '/mnt/dai-sys/wanghanzhen/models/google/gemma-4-12B-it'
original_bytes = (PUBLISH / 'config.json').read_bytes()
original = json.loads(original_bytes)
candidate = copy.deepcopy(original)
candidate['dflash_config'] = candidate.pop('dflash2_config')
candidate['dflash_config']['final_logit_softcapping'] = candidate['final_logit_softcapping']
candidate['dflash_config']['output_multiplier'] = 1.0
candidate['rope_theta'] = candidate['rope_parameters']['rope_theta']
candidate['partial_rotary_factor'] = candidate['rope_parameters']['partial_rotary_factor']
(ROOT / 'config.sglang.json').write_text(json.dumps(candidate, indent=2) + '\n')

# Execute the unmodified, pure config-parser definitions from the pinned
# upstream source, without importing its GPU-serving dependencies.
names = {'_cfg_get', '_get_text_config', '_get_dflash_config',
         '_parse_optional_int', '_parse_dflash_is_mla', 'build_target_layer_ids',
         'DFlashDraftConfig', 'parse_dflash_draft_config'}
tree = ast.parse((ROOT / 'dflash_utils.py').read_text())
nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
assert {n.name for n in nodes} == names
namespace = dict(__name__=__name__, Any=Any, Optional=Optional, List=List,
                 Integral=Integral, dataclass=dataclasses.dataclass,
                 DEFAULT_DFLASH_MASK_TOKEN='<|MASK|>',
                 _DFLASH_ATTENTION_MODES=frozenset({'gqa', 'mha', 'mla'}))
exec(compile(ast.Module(body=nodes, type_ignores=[]), 'pinned_upstream_config_parser', 'exec'), namespace)
parse = namespace['parse_dflash_draft_config']
before, after = parse(draft_hf_config=original), parse(draft_hf_config=candidate)
assert before.selector_rank == 0 and before.conv_kernel_size == 0
assert after.selector_rank == 256 and after.selector_top_k == 16
assert after.conv_kernel_size == 2 and after.conv_group_size == 16
assert after.block_size == 8 and after.mask_token_id == 4
assert after.target_layer_ids == [1, 5, 17, 29, 41]
assert after.final_logit_softcapping == 30.0

# Exercise the exact activation dispatch branch in the upstream constructor.
# This is a source-level negative check, not a full upstream engine launch.
tree = ast.parse((ROOT / 'dflash.py').read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DFlashMLP')
init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
branch = next(n for n in init.body if isinstance(n, ast.If) and 'hidden_act' in ast.unparse(n.test))
try:
    exec(compile(ast.Module(body=[branch], type_ignores=[]), 'upstream_activation_dispatch', 'exec'),
         {'hidden_act': candidate['hidden_act']})
except ValueError as exc:
    activation_error = str(exc)
else:
    raise AssertionError('Expected the pinned upstream to reject GELU')
print('UPSTREAM_CONFIG_PARSER_PASS', flush=True)
print('UPSTREAM_ACTIVATION_BLOCKER', activation_error, flush=True)

# Use the existing, private Gemma-patched SGLang runtime only for a meta-device
# constructor/loader shape check. No GPU allocation or serving run occurs.
import torch
from transformers import Qwen3Config
from sglang.srt.models.dflash import DFlash2DraftModel
from sglang.srt.runtime_context import create_context, get_parallel, use_context
from sglang.srt.server_args import ServerArgs
import sglang.srt.models.dflash as patched_module

with (PUBLISH / 'model.safetensors').open('rb') as f:
    header_len = struct.unpack('<Q', f.read(8))[0]
    header = json.loads(f.read(header_len))
manifest = {k:v for k,v in header.items() if k != '__metadata__'}
assert len(manifest) == 81
assert all(v['dtype'] == 'BF16' for v in manifest.values())
config = Qwen3Config(**candidate)
checks = []
mapping = [('q_proj', 'qkv_proj'), ('k_proj', 'qkv_proj'), ('v_proj', 'qkv_proj'),
           ('gate_proj', 'gate_up_proj'), ('up_proj', 'gate_up_proj')]
for tp in (1, 2):
    args = ServerArgs(model_path=TARGET, device='cpu', attention_backend='triton',
                      tp_size=tp, dtype='bfloat16', disable_cuda_graph=True, disable_radix_cache=True)
    context = create_context(args, role='model_runner')
    with use_context(context), get_parallel().override(
        tp_size=tp, attn_tp_size=tp, tp_rank=0, attn_tp_rank=0, pp_rank=0,
        tp_group=None, attn_tp_group=None
    ):
        with torch.device('meta'):
            model = DFlash2DraftModel(config).to(dtype=torch.bfloat16)
            model.load_weights((name, torch.empty(info['shape'], dtype=torch.bfloat16, device='meta'))
                               for name, info in manifest.items())
        params = dict(model.named_parameters())
        mapped = set()
        for name in manifest:
            for old, new in mapping:
                name = name.replace('.'+old+'.', '.'+new+'.')
            assert name in params, ('unexpected checkpoint tensor', name)
            mapped.add(name)
        assert mapped == set(params), ('unloaded model parameters', set(params)-mapped)
        assert model.draft_config.final_logit_softcapping == 30.0
        checks.append(dict(tp_size=tp, rank=0, checkpoint_tensors=len(manifest),
                           mapped_parameters=len(params), full_name_coverage=True,
                           meta_constructor_and_loader='passed'))
        del model
    print('PATCHED_RUNTIME_META_PASS', tp, flush=True)

assert (PUBLISH / 'config.json').read_bytes() == original_bytes
report = dict(
    upstream=json.loads((ROOT/'upstream.json').read_text()),
    original_config_sha256=hashlib.sha256(original_bytes).hexdigest(),
    candidate_config=str(ROOT/'config.sglang.json'),
    original_parser=dataclasses.asdict(before), candidate_parser=dataclasses.asdict(after),
    upstream_activation_dispatch_error=activation_error,
    patched_runtime_source=patched_module.__file__,
    patched_runtime_source_sha256=hashlib.sha256(Path(patched_module.__file__).read_bytes()).hexdigest(),
    patched_runtime_meta_checks=checks, gpu_inference_tested=False,
    weight_values_changed=False, published_config_changed=False,
    conclusion='Config schema conversion passes; upstream GELU support is required. Patched runtime passes meta loading only.'
)
(ROOT/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2), flush=True)
