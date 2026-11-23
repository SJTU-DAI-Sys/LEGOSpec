from pathlib import Path
import ast,json
root=Path('tests/environment_upgrade_20261123/upstream0521/sglang/srt')
checks={}
for file in ['distributed/parallel_state.py','layers/dp_attention.py','layers/logits_processor.py','models/gemma4_unified.py','model_executor/forward_batch_info.py']:
 tree=ast.parse((root/file).read_text());checks[file]=[]
 for n in ast.walk(tree):
  if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ['GroupCoordinator','initialize_dp_attention','LogitsProcessorOutput','init_new','Gemma4UnifiedForConditionalGeneration']:
   if isinstance(n,ast.FunctionDef):checks[file].append((n.name,ast.unparse(n.args)))
   else:checks[file].append((n.name, [t.target.id for t in n.body if isinstance(t,ast.AnnAssign) and isinstance(t.target,ast.Name)]))
print(json.dumps(checks,indent=2))
