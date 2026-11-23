"""Reuse semantic checks and verify every restarted endpoint after resource-only tuning."""
from pathlib import Path
import json,time
from generate import fingerprint,verify_server,completion
from transformers import AutoTokenizer
j=Path(__file__).resolve().parent
c=json.loads((j/'config.json').read_text());oldc=json.loads((j/'config.before_concurrency160.json').read_text())
assert c==dict(oldc,concurrency_per_replica=160)
old=json.loads((j/'smoke.before_concurrency160.json').read_text());oldfp,oldhashes=fingerprint(oldc);fp,hashes=fingerprint(c)
assert old['status']=='passed' and old['fingerprint']==oldfp
assert old['model_metadata_sha256']==oldhashes==hashes
endpoints=[f'http://127.0.0.1:{c["base_port"]+i}' for i in range(c['replicas'])]
for _ in range(1440):
 try:
  for endpoint in endpoints:verify_server(c,endpoint)
  break
 except Exception:time.sleep(5)
else:raise TimeoutError('Server readiness timeout')
print('All three TP2 services ready; checking unchanged generation parameters',flush=True)
t=AutoTokenizer.from_pretrained(c['model'],local_files_only=True,trust_remote_code=False);proof=[]
for endpoint in endpoints:
 a,m=completion(c,t,endpoint,[dict(role='user',content='What is 2 + 3? Reply with only the integer.')],'concurrency160/math',0)
 assert a['content'].strip().strip(' .\n')=='5',a
 proof.append(dict(endpoint=endpoint,case='concurrency160_math',assistant=a,generation=m))
receipt=dict(old,fingerprint=fp,model_metadata_sha256=hashes,cases=old['cases']+proof,
 runtime_change=dict(previous_concurrency_per_replica=256,concurrency_per_replica=160,only_changed_config_key='concurrency_per_replica',prior_receipt=str(j/'smoke.before_concurrency160.json')))
p=Path(c['output_dir'])/'default/smoke.parts0-1.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
print('160 concurrency verification passed; receipt published',flush=True)
