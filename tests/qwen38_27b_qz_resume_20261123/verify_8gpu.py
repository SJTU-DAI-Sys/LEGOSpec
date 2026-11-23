"""Validate the added TP2 replica before resuming unchanged generation."""
from pathlib import Path
import json,time
from generate import fingerprint,http,smoke,verify_server
from transformers import AutoTokenizer
j=Path(__file__).resolve().parent
c=json.loads((j/'config.json').read_text())
oldc=json.loads((j/'config.6gpu.json').read_text())
expected=dict(oldc,replicas=4)
assert c==expected, 'Only replica count may change during this expansion'
old=json.loads((j/'smoke.6gpu.json').read_text())
oldfp,oldhashes=fingerprint(oldc)
fp,hashes=fingerprint(c)
assert old['status']=='passed' and old['fingerprint']==oldfp
assert old['model_metadata_sha256']==oldhashes==hashes
endpoints=[f'http://127.0.0.1:{c["base_port"]+i}' for i in range(c['replicas'])]
for _ in range(1440):
 try:
  for endpoint in endpoints:verify_server(c,endpoint)
  break
 except Exception:time.sleep(5)
else:raise TimeoutError('Four server readiness timeout')
print('All four TP2 services ready; checking new replica',flush=True)
t=AutoTokenizer.from_pretrained(c['model'],local_files_only=True,trust_remote_code=False)
proof=smoke(c,t,[endpoints[-1]],[])
receipt=dict(old,endpoints=endpoints,fingerprint=fp,model_metadata_sha256=hashes,
 cases=old['cases']+proof,topology_expansion=dict(previous_replicas=3,replicas=4,
 original_receipt=str(j/'smoke.6gpu.json'),reused_endpoints=endpoints[:-1],new_endpoint=endpoints[-1]))
p=Path(c['output_dir'])/'default/smoke.parts0-1.json'
tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
print('8-GPU smoke passed: unchanged model/template/client/sampling; new TP2 strict budget, math, code and multi-turn passed',flush=True)
