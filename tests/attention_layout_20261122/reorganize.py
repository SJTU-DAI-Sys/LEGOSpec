"""Move implementation ownership without changing parameter names or algorithms."""
from pathlib import Path
import ast,shutil,json,re,subprocess
root=Path('/mnt/dai-sys/wanghanzhen/projects/DAC27/LEGOSpec_public')
work=root/'tests/attention_layout_20261122'
base='legospec/modeling/draft/composable/backbone/'
attention=base+'attention/'
module='legospec.modeling.draft.composable.backbone.attention'
original={str(p.relative_to(root)):p.read_text() for tree in ['legospec','tests/unit','scripts'] for p in (root/tree).rglob('*.py')}
common=original[attention+'common.py']
def symbol(source,name):
 node=next(n for n in ast.parse(source).body if getattr(n,'name',None)==name)
 return ast.get_source_segment(source,node)
def definitions(source,names):return '\n\n\n'.join(symbol(source,n) for n in names)+'\n'
# Group projections with their consumer. Keep only genuinely shared softmax execution shared.
updates={}
updates[attention+'softmax.py']='''"""Shared dense/latent attention lifecycle: projections, KV commit and execution."""
import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config
from legospec.modeling.draft.composable.backbone.kernels import BackboneKernels
from legospec.modeling.draft.composable.contracts import LayerState

'''+symbol(common,'AttentionBase').replace('== SLIDING_ATTENTION','== "sliding_attention"')+'\n\n\n'+symbol(original[attention+'base.py'],'AttentionForward')+'\n'
updates[attention+'dense.py']='''"""MHA, GQA and MQA: head geometry, projections and softmax attention."""
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config, rotate_half
from legospec.modeling.draft.composable.backbone.kernels import BackboneKernels
from legospec.modeling.draft.composable.backbone.attention.softmax import AttentionBase, AttentionForward
from legospec.modeling.draft.composable.backbone.attention.access.visibility import build_access
from legospec.modeling.draft.composable.backbone.attention.backends.flex import compile_friendly_flex_attention, flex_kernel_options

'''+definitions(common,['apply_rotary_pos_emb','GQAProjection'])+'\n\n'+symbol(original[attention+'gqa.py'],'GQAAttention')+'\n'
updates[attention+'latent.py']='''"""MLA projections, expanded training, absorbed inference and latent cache."""
from copy import deepcopy
import math
import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config, rotate_half
from legospec.modeling.draft.composable.backbone.kernels import BackboneKernels
from legospec.modeling.draft.composable.contracts import MLALayerState
from legospec.modeling.draft.composable.backbone.attention.softmax import AttentionBase, AttentionForward
from legospec.modeling.draft.composable.backbone.attention.access.visibility import build_access

'''+definitions(common,['_require_bool_config','_resolve_mla_rope_interleaved','_rotate_half_interleaved','apply_mla_rope','MLAProjection'])+'\n\n'+symbol(original[attention+'mla.py'],'MLAAttention')+'\n'
moves={'gdn.py':'linear/recurrence.py','gdn_events.py':'linear/events.py','block.py':'linear/block.py','access.py':'access/visibility.py','sparse.py':'access/dsa.py','flex.py':'backends/flex.py','torch_compat.py':'backends/torch_compat.py'}
for old,new in moves.items():updates[attention+new]=original[attention+old]
updates[attention+'linear/cache.py']=original[attention+'gdn_state.py']+'\n\n'+symbol(original[attention+'gdn_cache.py'],'prepare_cache')+'\n'
updates[attention+'access/indexer.py']=original[base+'indexer.py']
removed=[attention+f for f in ['common.py','base.py','gqa.py','mla.py','gdn_state.py','gdn_cache.py',*moves]]+[base+'indexer.py']
# Absolute module imports are rewritten in source and maintained tests. Historical snapshots stay untouched.
mapping={module+'.'+old[:-3]:module+'.'+new[:-3].replace('/','.') for old,new in moves.items()}
mapping.update({module+'.base':module+'.softmax',module+'.gqa':module+'.dense',module+'.mla':module+'.latent',module+'.gdn_state':module+'.linear.cache',module+'.gdn_cache':module+'.linear.cache','legospec.modeling.draft.composable.backbone.indexer':module+'.access.indexer'})
for rel,s in original.items():
 if rel not in removed:updates.setdefault(rel,s)
# One-consumer rotary config preparation belongs to Layer, not a common algorithm bucket.
p=base+'model.py';s=updates[p]
s=s.replace('from '+module+'.common import _rope_config\n','')
a='''            self.rotary_emb = Qwen3RotaryEmbedding(
                _rope_config(local, recipe["attention"]["representation"])
            )'''
b='''            rotary_config = local
            if recipe["attention"]["representation"] == "mla":
                rotary_config = deepcopy(local)
                rotary_config.head_dim = local.qk_rope_head_dim
            self.rotary_emb = Qwen3RotaryEmbedding(rotary_config)'''
assert a in s;s=s.replace(a,b);updates[p]=s
for rel,s in list(updates.items()):
 # The sole remaining common.py consumer is MLA optimizer geometry.
 s=s.replace(module+'.common',module+'.latent')
 for old,new in sorted(mapping.items(),key=lambda p:-len(p[0])):
  s=re.sub(re.escape(old)+r'(?=[\s.]|$)',lambda _:new,s)
 s=s.replace('from '+module+' import gdn as gated_deltanet','from '+module+'.linear import recurrence as gated_deltanet')
 s=s.replace('from '+module+' import sparse','from '+module+'.access import dsa as sparse')
 ast.parse(s);updates[rel]=s
updates={p:s for p,s in updates.items() if original.get(p)!=s}
# Back up exact touched originals before any move.
for rel in set(updates)|set(removed):
 if rel in original:
  p=work/'backup'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(original[rel])
(work/'git_status_before.txt').write_text(subprocess.check_output(['git','status','--short'],cwd=root,text=True))
for rel,s in updates.items():
 p=root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s)
for rel in removed:(root/rel).unlink()
(work/'files.json').write_text(json.dumps({'updated':list(updates),'removed':removed},indent=2))
print('Updated',len(updates),'files; retired',len(removed),'old locations; backup:',work/'backup')
