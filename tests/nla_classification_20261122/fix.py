from pathlib import Path
import ast,shutil,json,hashlib
root=Path('/mnt/dai-sys/wanghanzhen/projects/DAC27/LEGOSpec_public');work=root/'tests/nla_classification_20261122'
base='legospec/modeling/draft/composable/objective/'
changed=[]
def save(rel,text):
 p=root/rel;b=work/'backup'/rel
 if p.exists():b.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,b)
 p.write_text(text);changed.append(rel)
s=(root/base/'block.py').read_text();node=next(n for n in ast.parse(s).body if isinstance(n,ast.ClassDef) and n.name=='NLAObjective');source=ast.get_source_segment(s,node)
lines=s.splitlines(True);del lines[node.lineno-1:node.end_lineno]
s=''.join(lines).replace('"""Negative log overlap, globally normalized by decayed token weights."""','"""Block-coupled D-PACE and expected prefix survival objectives."""').replace('from legospec.modeling.draft.composable.objective.vocab import distribution\n','')
save(base+'block.py',s)
p=root/base/'token.py';s=p.read_text()
if not any(isinstance(n,ast.Import) and any(a.name=='torch' for a in n.names) for n in ast.parse(s).body):s=s.replace('from __future__ import annotations','from __future__ import annotations\n\nimport torch')
s+='\n\n# Per-token negative log overlap, normalized by valid decayed token weights.\n'+source+'\n';save(base+'token.py',s)
for folder in ['legospec','scripts','tests/unit']:
 for p in (root/folder).rglob('*.py'):
  s=p.read_text();t=s.replace('from legospec.modeling.draft.composable.objective.block import NLAObjective','from legospec.modeling.draft.composable.objective.token import NLAObjective')
  if s!=t:save(str(p.relative_to(root)),t)
p=root/base/'DESIGN.md';s=p.read_text().replace('token 放 CE/KL/TV/L1/candidate CE；block 放 NLA/DPACE/E2E-TV','token 放 CE/KL/TV/L1/NLA/candidate CE；block 放 DPACE/E2E-TV');s+='\nNLA 是逐 token 的负 log-overlap，以有效位置权重和为分母，不要求完整 block，不包含跨 token 的 prefix 累乘。lk_alpha 保留为 NLA 的配置别名。\n';save(base+'DESIGN.md',s)
new=next(n for n in ast.parse((root/base/'token.py').read_text()).body if isinstance(n,ast.ClassDef) and n.name=='NLAObjective')
assert ast.dump(new,include_attributes=False)==ast.dump(node,include_attributes=False)
(work/'changes.json').write_text(json.dumps({'changed':changed,'class_ast_unchanged':True},indent=2));print('Moved NLA to token.py; class AST unchanged;',len(changed),'files updated')
