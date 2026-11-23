from pathlib import Path
import ast,json,shutil
r=Path("legospec/modeling/draft")
# Expanded validation is a private representation, not an alternate public format.
p=r/"schema.py";s=p.read_text().replace("_normalize_legacy","_validate_expanded")
s=s.replace('    # String recipes retain their exact historical graph for saved checkpoints.\n    layers = config.composable_config.get("backbone", {}).get("layers") or [{}]\n    if not isinstance(layers[0].get("attention"), dict):\n        return _validate_expanded(config)\n','')
# runtime input is always canonical
s=s.replace('    if isinstance(config.composable_config["backbone"]["layers"][0]["attention"], dict):\n        config = _execution_recipe(config)','    config = _execution_recipe(config)')
p.write_text(s)
# Move expansion and its validation together, public schema remains boundary/default/field policy.
s=p.read_text();a=s.index("def _validate_expanded");b=s.index("FAMILIES =",a)
expanded=s[a:b];s=s[:a]+s[b:]
a=s.index("def _execution_recipe");b=s.index("def normalize_recipe",a);expanded+=s[a:b];s=s[:a]+s[b:]
names=["fields","integer","number","boolean","check_objective","ATTENTION_TYPES","ACCESS_TYPES","PROPOSAL_TYPES","MLA_DEFAULTS","GDN_DEFAULTS","DSA_DEFAULTS","FIELDS","FAMILIES","ACCESS"]
# avoid top-level cycle: runtime imports schema; schema calls runtime lazily
header='"""Expand validated public recipes into executable layer parameters."""\nfrom copy import deepcopy\nimport math\nfrom .schema import '+", ".join(names)+'\nfrom .loss_config import normalize_training_recipe, SELECTOR_TYPES, selector_mode\n\n'
(r/"runtime_config.py").write_text(header+expanded)
s=s.replace('def normalize_recipe(config):','def normalize_recipe(config):\n    from .runtime_config import _execution_recipe')
p.write_text(s)
p=r/"configuration.py";s=p.read_text().replace("    runtime_recipe,\n","");s=s.replace("from legospec.modeling.draft.schema import (","from legospec.modeling.draft.runtime_config import runtime_recipe\nfrom legospec.modeling.draft.schema import (")
# select target layers in config once rather than model mutation/revalidation
s=s.replace("        self.validate_recipe()","""        if self.composable_config["context"].get("target_layer_ids") is None:
            count = 5 if self.composable_config["backbone"].get("schedule") == "stepwise" else self.num_hidden_layers
            if self.num_target_layers < 4:
                selected = list(range(self.num_target_layers))
            elif count == 1:
                selected = [self.num_target_layers // 2]
            else:
                selected = sorted({round(1 + i * (self.num_target_layers - 4) / (count - 1)) for i in range(count)})
            self.composable_config["context"]["target_layer_ids"] = selected
        self.validate_recipe()""");p.write_text(s)
p=r/"modeling.py";s=p.read_text();a=s.index("        config.validate_recipe()");b=s.index('        self.target_layer_ids',a)
s=s[:a]+'''        recipe = config.composable_config
        self.uses_hidden_feedback = recipe["backbone"].get("schedule") == "stepwise"
'''+s[b:];p.write_text(s)
# Migrate current stored recipes to canonical family/access, never production compatibility.
fam={"mha":"dense","gqa":"dense","mqa":"dense","mla":"latent","gdn":"linear","kda":"linear"}
access={"dense":"full","sliding_window":"swa","linear":None,"dsa":"dsa"}
def migrate(value):
 if isinstance(value,list):return [migrate(v) for v in value]
 if not isinstance(value,dict):return value
 v={k:migrate(x) for k,x in value.items()}
 if v.get("attention") in fam if isinstance(v.get("attention"),str) else False:
  kind=v["attention"];a={"family":fam[kind],"variant":kind}
  old=v.pop("access",None)
  if kind not in ("gdn","kda"):
   a["access"]=access.get(old,old or "full")
   if "window_size" in v:a["window_size"]=v.pop("window_size")
  v["attention"]=a;v.pop("block_attention",None)
 if "access_configs" in v:
  v["access_configs"]={access.get(k,k):x for k,x in v["access_configs"].items() if k!="linear"}
 return v
for p in Path("configs").glob("*.json"):
 v=json.loads(p.read_text());p.write_text(json.dumps(migrate(v),indent=2)+"\n")
# literal recipe fixtures
class Migrate(ast.NodeTransformer):
 def visit_Dict(self,node):
  self.generic_visit(node)
  entries={k.value:(i,v) for i,(k,v) in enumerate(zip(node.keys,node.values)) if isinstance(k,ast.Constant) and isinstance(k.value,str)}
  if "attention" in entries:
   i,v=entries["attention"]
   if isinstance(v,ast.Constant) and v.value in fam:
    kind=v.value;a={"family":fam[kind],"variant":kind}
    if kind not in ("gdn","kda"):
     old=entries.get("access",(None,ast.Constant("dense")))[1]
     if isinstance(old,ast.Constant):a["access"]=access.get(old.value,old.value)
    node.values[i]=ast.parse(repr(a),mode="eval").body
    keep=[j for j,k in enumerate(node.keys) if not isinstance(k,ast.Constant) or k.value not in ("access","block_attention")]
    node.keys=[node.keys[j] for j in keep];node.values=[node.values[j] for j in keep]
  if "access_configs" in entries:
   _,child=entries["access_configs"]
   if isinstance(child,ast.Dict):
    keep=[i for i,k in enumerate(child.keys) if not isinstance(k,ast.Constant) or k.value!="linear"]
    child.keys=[ast.Constant(access.get(child.keys[i].value,child.keys[i].value)) if isinstance(child.keys[i],ast.Constant) else child.keys[i] for i in keep]
    child.values=[child.values[i] for i in keep]
  return node
# edit only dict spans to retain comments and test oracle text
for p in Path("tests/unit").glob("*.py"):
 if p.name=="test_gdn_block_attention.py":
  dest=Path("tests/development/legacy_attention");dest.mkdir(parents=True,exist_ok=True);shutil.move(p,dest/p.name);continue
 s=p.read_text();tree=ast.parse(s);changes=[]
 for n in ast.walk(tree):
  if isinstance(n,ast.Dict):
   entries={k.value:v for k,v in zip(n.keys,n.values) if isinstance(k,ast.Constant)}
   is_layer=isinstance(entries.get("attention"),ast.Constant) and entries["attention"].value in fam
   is_access="access_configs" in entries
   if is_layer or is_access:
    import copy
    new=ast.unparse(Migrate().visit(copy.deepcopy(n)))
    lines=s.splitlines(True);offset=[0]
    for line in lines:offset.append(offset[-1]+len(line))
    changes.append((offset[n.lineno-1]+n.col_offset,offset[n.end_lineno-1]+n.end_col_offset,new))
 # outer replacements subsume nested ones
 selected=[]
 for c in sorted(changes,key=lambda c:(c[0],-c[1])):
  if not selected or c[0]>=selected[-1][1]:selected.append(c)
 for a,b,v in reversed(selected):s=s[:a]+v+s[b:]
 p.write_text(s)
