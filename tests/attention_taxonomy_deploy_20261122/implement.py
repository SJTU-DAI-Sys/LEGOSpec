from pathlib import Path
import json, shutil, hashlib
root=Path('/mnt/dai-sys/wanghanzhen/projects/DAC27/LEGOSpec_public')
work=root/'tests/attention_taxonomy_deploy_20261122'
source=root.parent/'LEGOSpec_20261111/tests/attention_taxonomy_20261122/runtime'
changes=[]
def save(rel,text):
 p=root/rel
 if p.exists():
  backup=work/'backup'/rel
  if not backup.exists():backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,backup)
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text);changes.append(rel)
def edit(rel,pairs):
 s=(root/rel).read_text()
 for a,b in pairs:
  assert a in s,(rel,a);s=s.replace(a,b)
 save(rel,s)
base='legospec/modeling/draft/composable/'
s=(root/base/'schema.py').read_text()
pairs=[
 ('ATTENTION_TYPES = ("mha", "gqa", "mla", "gdn")','ATTENTION_TYPES = ("mha", "gqa", "mqa", "mla", "gdn", "kda")'),
 ('"gqa": {"num_attention_heads", "num_key_value_heads", "head_dim"},','"gqa": {"num_attention_heads", "num_key_value_heads", "head_dim"},\n    "mqa": {"num_attention_heads", "num_key_value_heads", "head_dim"},'),
 ('"gdn": set(GDN_DEFAULTS),','"gdn": set(GDN_DEFAULTS),\n    "kda": set(GDN_DEFAULTS),'),
 ('def normalize_recipe(config):','def _normalize_legacy(config):'),
 ('if (kind == "gdn") != (access == "linear") or (\n            kind == "mla" and access == "dsa"\n        ):','if (kind in ("gdn", "kda")) != (access == "linear"):'),
 ('if kind != "gdn":','if kind not in ("gdn", "kda"):'),
 ('if kind == "gdn":','if kind in ("gdn", "kda"):'),
 ('if kind in ("mha", "gqa"):','if kind in ("mha", "gqa", "mqa"):'),
 ('else target["num_key_value_heads"],','else 1 if kind == "mqa" else target["num_key_value_heads"],'),
 ('        if kind == "mla" and (','        if kind == "mqa" and profile["num_key_value_heads"] != 1:\n            raise ValueError("MQA requires exactly one KV head")\n        if kind == "kda" and profile["linear_key_head_dim"] > 256 and config.execution_config["gdn_backend"] != "reference":\n            raise ValueError("KDA FLA supports key head dimension <=256")\n        if kind == "mla" and ('),
 ('profile = profiles["gdn"]','profile = profiles[layer["attention"]]'),
 ('"attention": {"representation": "gdn", "access": "linear"},','"attention": {"representation": kind, "access": "linear"},'),
 ('                        **profiles[kind],\n                        "backend"','                        **profiles[kind],\n                        "variant": kind,\n                        "backend"'),
 ('    r = deepcopy(config.composable_config)\n    profiles, accesses = r.pop','    if isinstance(config.composable_config["backbone"]["layers"][0]["attention"], dict):\n        config = _execution_recipe(config)\n    r = deepcopy(config.composable_config)\n    profiles, accesses = r.pop'),
 ('    r["backbone"]["layers"] = layers','    if getattr(config, "_intrinsic_linear_attention", False):\n        for layer in layers:\n            if layer["attention"]["representation"] in ("gdn", "kda"):\n                branch = layer.pop("block_attention")\n                layer["gdn_config"].update(intrinsic_block_attention=True, gate_init=branch["gate_init"])\n    r["backbone"]["layers"] = layers')]
for a,b in pairs:
 assert a in s,a;s=s.replace(a,b)
# Reuse the tested taxonomy boundary inside the existing schema module.
v4=(source/'litespec/modeling/draft/composable/schema_v4.py').read_text()
v4=v4[v4.index('FAMILIES ='):].replace('execution_v3','_execution_recipe').replace('normalize_v3','_normalize_legacy').replace('normalize_v4','normalize_recipe').replace("    result.schema_version = 'composable_v3'\n",'')
v4=v4.replace('def normalize_recipe(config):\n    original', 'def normalize_recipe(config):\n    # String recipes retain their exact historical graph for saved checkpoints.\n    layers = config.composable_config.get("backbone", {}).get("layers") or [{}]\n    if not isinstance(layers[0].get("attention"), dict):\n        return _normalize_legacy(config)\n    original')
save(base+'schema.py',s+'\n\n'+v4)
edit(base+'configuration.py',[
 ('{"attention": "gqa", "access": "dense"} for _ in range(depth)','{"attention": {"family": "dense", "variant": "gqa", "access": "full"}} for _ in range(depth)'),
 ('first["num_attention_heads"] * first["head_dim"]','first["num_attention_heads"] * (first["qk_nope_head_dim"] + first["qk_rope_head_dim"] if first["representation"] == "mla" else first["head_dim"])')])
edit(base+'backbone/model.py',[
 ('recipe["attention"]["representation"] == "gdn"','recipe["attention"]["representation"] in ("gdn", "kda")'),
 ('            self.block_attention = (','            linear = recipe["gdn_config"]\n            block = (dict(share_gdn_projections=True, num_heads=linear["linear_num_value_heads"],\n                          head_dim=linear["linear_key_head_dim"], gate_init=linear["gate_init"])\n                     if linear.get("intrinsic_block_attention") else recipe.get("block_attention"))\n            self.block_attention = ('),
 ('                    recipe["block_attention"],','                    block,'),
 ('if recipe.get("block_attention") is not None','if block is not None'),
 ('                    "gqa",\n                    "mla",','                    "gqa",\n                    "mqa",\n                    "mla",')])
edit(base+'backbone/attention/gdn.py',[
 ('state = state * g[0, t].float().exp()[:, None, None]','decay = g[0, t].float().exp()\n            state = state * (decay[:, None, None] if decay.ndim == 1 else decay[:, :, None])'),
 ('self.hk = recipe["linear_num_key_heads"]','self.variant = recipe.get("variant", "gdn")\n        self.hk = recipe["linear_num_key_heads"]'),
 ('self.in_proj_a = nn.Linear(config.hidden_size, self.hv, bias=False)','self.gate_shape = (self.hv, self.dk) if self.variant == "kda" else (self.hv,)\n        self.in_proj_a = nn.Linear(config.hidden_size, math.prod(self.gate_shape), bias=False)'),
 ('self.dt_bias = nn.Parameter(torch.ones(self.hv))','self.dt_bias = nn.Parameter(torch.ones(self.gate_shape))'),
 ('g = -self.A_log.float().exp() * F.softplus(\n            self.in_proj_a(c).float() + self.dt_bias.float()\n        )','rate = self.A_log.float().exp()\n        if self.variant == "kda":\n            rate = rate[:, None]\n        g = -rate * F.softplus(self.in_proj_a(c).float().reshape(-1, *self.gate_shape) + self.dt_bias.float())'),
 ('g = g.masked_fill(~layout.context_keep[:, None], 0)','g = g.masked_fill(~layout.context_keep.reshape(-1, *([1] * len(self.gate_shape))), 0)'),
 ('ge = g.new_zeros(1, e, self.hv).index_copy','ge = g.new_zeros(1, e, *self.gate_shape).index_copy'),
 ('from fla.ops.gated_delta_rule import chunk_gated_delta_rule','if self.variant == "kda":\n                    from fla.ops.kda import chunk_kda as chunk_gated_delta_rule\n                else:\n                    from fla.ops.gated_delta_rule import chunk_gated_delta_rule')])
edit(base+'backbone/attention/__init__.py',[('"gqa": GQAAttention,','"gqa": GQAAttention, "mqa": GQAAttention,')])
edit(base+'backbone/attention/mla.py', [('self.backend = recipe.get("backend", "sdpa")','self.is_dsa = recipe.get("access") == "dsa"\n        self.backend = recipe.get("backend", "sdpa")')])
edit(base+'backbone/indexer.py', [('source = attention.q_norm(\n            attention.q_proj(u).reshape(n, a, -1, attention.head_dim)\n        )','source = (attention._project_q(u).reshape(n, a, attention.num_heads, attention.qk_head_dim)\n                  if hasattr(attention, "_project_q") else attention.q_norm(\n                      attention.q_proj(u).reshape(n, a, -1, attention.head_dim)))')])
edit('legospec/training/planning.py',[
 ('            pairs = float(k * anchors.sum() + a * k * k)','            if isinstance(attention, dict):\n                access = attention.get("access", "linear")\n                attention = attention["variant"]\n            pairs = float(k * anchors.sum() + a * k * k)'),
 ('window = settings.get("window_size", settings.get("window", 512))','window = (layer["attention"].get("window_size") if isinstance(layer["attention"], dict) else layer.get("window_size")) or settings.get("window_size", settings.get("window", 512))'),
 ('attention in ("gdn", "gated_deltanet")','attention in ("gdn", "kda", "gated_deltanet")'),
 ('work = (length + q) * h * h','work = (length + q) * h * h + (a * k * k * h if isinstance(layer["attention"], dict) else 0)')])
# Migrate only maintained recipe catalog, never checkpoint/native configs.
for p in sorted((root/'configs').glob('*.json')):
 d=json.loads(p.read_text());r=d.get('composable_config')
 if r is None:continue
 for layer in r['backbone']['layers']:
  variant=layer['attention'];family='linear' if variant in ('gdn','kda') else 'latent' if variant=='mla' else 'dense'
  a={'family':family,'variant':variant};access=layer.pop('access')
  if family!='linear':
   a['access']={'dense':'full','sliding_window':'swa','dsa':'dsa'}[access]
   if 'window_size' in layer:a['window_size']=layer.pop('window_size')
  else:
   branch=layer.pop('block_attention',{})
   r['attention_configs'][variant]['gate_init']=branch.get('gate_init',.1)
  layer['attention']=a
 r['access_configs']={ {'dense':'full','sliding_window':'swa','dsa':'dsa'}[k]:v for k,v in r.get('access_configs',{}).items() if k!='linear'}
 save(str(p.relative_to(root)),json.dumps(d,indent=2)+'\n')
p=source/'configs/dflash_kda5_rnn.json';d=json.loads(p.read_text());d.pop('schema_version',None);d['architectures']=['LegoSpecModel'];d['composable_config']['context']['type']='concat_projection'
save('configs/dflash_kda5_rnn.json',json.dumps(d,indent=2)+'\n')
t=(source/'tests/attention_taxonomy_20261122/test_taxonomy.py').read_text().replace('litespec','legospec').replace('ComposableDraftConfig','LegoSpecConfig').replace('ComposableDraftModel','LegoSpecModel').replace('ModelInputs','DraftInputs').replace("schema_version='composable_v4',",'').replace("type='dflash'","type='concat_projection'").replace('attention.gated_deltanet','attention.gdn').replace("if d.get('schema_version')!='composable_v4':continue","if 'composable_config' not in d:continue")
save('tests/attention_taxonomy_deploy_20261122/test_taxonomy.py',t)
(work/'changed_files.json').write_text(json.dumps(changes,indent=2))
print('Updated',len(changes),'files; backup:',work/'backup')
