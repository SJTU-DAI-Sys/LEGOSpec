"""Observe a warmup then a five-minute production window without changing requests."""
from pathlib import Path
from datetime import datetime
import json,time,statistics
from measure_throughput import measure
J=Path(__file__).resolve().parent
state=J/'benchmark_concurrency160.state.json'
def save(stage,**kw):
 value=dict(stage=stage,time=datetime.now().isoformat(),**kw);state.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value),flush=True)
def healthy():
 s=json.loads(next((J/'logs').glob('launch.*.json')).read_text())
 if s['controller_pid']==int((J/'logs/controller.pid').read_text()) and s['status']=='failed':raise RuntimeError(s.get('error'))
def wait(seconds):
 end=time.monotonic()+seconds
 while time.monotonic()<end:healthy();time.sleep(min(5,max(0,end-time.monotonic())))
save('awaiting_generation')
for _ in range(720):
 healthy()
 if '"resumed_success"' in (J/'logs/latest.log').read_text():break
 time.sleep(5)
else:raise TimeoutError('Generator did not resume')
save('warmup',seconds=180);wait(180)
root=J/'results'
def committed_offset(p):
 size=p.stat().st_size
 with p.open('rb') as f:
  pos=size
  while pos:
   lower=max(0,pos-1048576);f.seek(lower);b=f.read(pos-lower);i=b.rfind(b'\n')
   if i>=0:return lower+i+1
   pos=lower
 return 0
positions={str(p):committed_offset(p) for p in root.glob('part[01].jsonl')}
start=time.monotonic();save('measuring',seconds=300,start_offsets=positions);wait(300)
rows=[]
for path,pos in positions.items():
 with open(path,'rb') as f:f.seek(pos);raw=f.read()
 rows.extend(json.loads(l) for l in raw.splitlines(keepends=True) if l.endswith(b'\n'))
elapsed=time.monotonic()-start
r=measure(300);r.update(measurement_seconds=elapsed,committed_conversations=len(rows),committed_conversations_per_second=len(rows)/elapsed,committed_conversations_per_hour=len(rows)/elapsed*3600,avg_completed_turns=statistics.mean(len(x['generation']['turns']) for x in rows) if rows else None,avg_completed_output_tokens=statistics.mean(sum(t['sampled_tokens'] for t in x['generation']['turns']) for x in rows) if rows else None)
b=json.loads((J/'throughput_256_before160.json').read_text());r['comparison']=dict(baseline_decode=b['decode_tokens_per_second'],decode_change_percent=(r['decode_tokens_per_second']/b['decode_tokens_per_second']-1)*100,baseline_conversations_per_second=b['progress_slope']['conversations_per_second'],baseline_retractions=b['retraction_events'])
(J/'throughput_160_after_restart.json').write_text(json.dumps(r,indent=2)+'\n');save('completed',result=r)
