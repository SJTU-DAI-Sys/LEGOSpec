from pathlib import Path
import ast, hashlib, json, tarfile
R=Path(__file__).resolve().parents[2];W=Path(__file__).resolve().parent
removed=['PLAN_VERSION','ROUND_ORDER_VERSION','ANCHOR_SAMPLING_VERSION','LOSS_CONTRACT_VERSION','rank_local_optimizer_v2','legospec_training_v2','muon_adamw_v1','validate_session_resume','validate_training_identity','validate_resume_topology','training_data_identity','loss_contract_version','anchor_sampling_version','round_order_version','execution_contract','after_scheduler_type']
paths=[*(R/'legospec').rglob('*.py'),*(R/'scripts').rglob('*.py'),*(R/'tests/unit').glob('*.py')]
for p in paths:
 s=p.read_text();ast.parse(s)
 assert not any(name in s for name in removed),p
changed=[]
with tarfile.open(W/'before.tgz') as t:
 for item in t.getmembers():
  if not item.isfile() or '__pycache__' in item.name:continue
  p=R/item.name;before=t.extractfile(item).read()
  if p.exists() and p.read_bytes()!=before:
   changed.append({'file':item.name,'before_lines':len(before.splitlines()),'after_lines':len(p.read_bytes().splitlines()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
report={'checked_python_files':len(paths),'removed_markers':removed,'compatibility_adapters_added':False,'changes':changed}
(W/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(f'Audit passed: {len(paths)} active Python files; removed resume markers/checkers absent; {len(changed)} changed files recorded.')
