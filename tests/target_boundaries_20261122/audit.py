from pathlib import Path
import tarfile,ast,json
root=Path.cwd();work=root/'tests/target_boundaries_20261122'
changed=[];deleted=[]
with tarfile.open(work/'before.tgz') as backup:
 members={m.name:m for m in backup.getmembers() if m.isfile() and (m.name.endswith('.py') or m.name.endswith('.md'))}
 for name,member in members.items():
  p=root/name
  if not p.exists():deleted.append(name)
  elif p.read_bytes()!=backup.extractfile(member).read():changed.append(name)
 new=[str(p.relative_to(root)) for folder in ['legospec','tests/unit'] for p in (root/folder).rglob('*') if p.is_file() and p.suffix in ['.py','.md'] and str(p.relative_to(root)) not in members]
for p in (root/'legospec').rglob('*.py'):ast.parse(p.read_text(),filename=str(p))
violations=[]
for p in (root/'legospec/modeling/target').rglob('*.py'):
 if 'legospec.training' in p.read_text():violations.append(str(p))
for p in (root/'legospec/training/target').rglob('*.py'):
 for marker in ['model_runner','SGLangTP','gemma4','qwen35','target.model']:
  if marker in p.read_text():violations.append(f'{p}: {marker}')
report=dict(changed=changed,deleted=deleted,new=new,boundary_violations=violations)
(work/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2));assert not violations
