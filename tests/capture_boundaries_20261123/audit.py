from pathlib import Path
import ast,json,tarfile
root=Path.cwd(); work=root/'tests/capture_boundaries_20261123'
modules={}; edges=[]; violations=[]
for p in (root/'legospec').rglob('*.py'):
 rel=p.relative_to(root);name='.'.join(rel.with_suffix('').parts);tree=ast.parse(p.read_text());modules[name]=(p,tree)
 for node in ast.walk(tree):
  if isinstance(node,ast.ImportFrom) and node.module and node.module.startswith('legospec'):
   edges.append(dict(source=str(rel),line=node.lineno,target=node.module))
  if isinstance(node,ast.Import):
   for alias in node.names:
    if alias.name.startswith('legospec'):
     edges.append(dict(source=str(rel),line=node.lineno,target=alias.name))
for e in edges:
 if e['source'].startswith('legospec/modeling/target/') and e['target'].startswith('legospec.training'):
  violations.append(e)
for p in (root/'legospec/modeling/target/adapters').glob('*.py'):
 for token in ['ModelRunner(', 'NativeCapture(', 'init_process_group(', 'initialize_collocated_parallel', 'create_capture_engine']:
  if token in p.read_text(): violations.append(dict(source=str(p.relative_to(root)),token=token))
for p in [root/'legospec/data/capture_engine.py',root/'legospec/data/hidden_cache.py']:
 for token in ['Gemma4Capture','sglang_gemma4_capture','dist.init_process_group(', 'nn.functional.rms_norm(', 'model.language_model.norm.weight']:
  if token in p.read_text():violations.append(dict(source=str(p.relative_to(root)),token=token))
old_references=[]
for folder in ['legospec','scripts','tests/unit']:
 for p in (root/folder).rglob('*.py'):
  if 'sglang_gemma4_capture' in p.read_text() or 'Gemma4Capture' in p.read_text():old_references.append(str(p.relative_to(root)))
changes=[]
with tarfile.open(work/'before.tgz') as backup:
 known={m.name:m for m in backup.getmembers() if m.isfile() and Path(m.name).suffix in {'.py','.md'}}
 for name,m in known.items():
  p=root/name
  if not p.exists():changes.append(dict(file=name,status='deleted'))
  elif p.read_bytes()!=backup.extractfile(m).read():changes.append(dict(file=name,status='changed'))
 for folder in ['legospec','tests/unit']:
  for p in (root/folder).rglob('*'):
   if p.is_file() and p.suffix in {'.py','.md'} and str(p.relative_to(root)) not in known:changes.append(dict(file=str(p.relative_to(root)),status='new'))
findings={
 'modeling_to_training':[e for e in edges if e['source'].startswith('legospec/modeling/') and e['target'].startswith('legospec.training')],
 'custom_backend_external_imports':[e for e in edges if 'custom_backend' in e['target'] and '/custom_backend/' not in e['source']],
}
report=dict(python_files=len(modules),changes=changes,boundary_violations=violations,deleted_file_references=old_references,other_findings=findings)
(work/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2));assert not violations and not old_references
