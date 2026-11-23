"""Summarize current-host decode, cache pressure and committed progress."""
from pathlib import Path
from datetime import datetime,timedelta
import json,re,argparse
J=Path(__file__).resolve().parent

def measure(seconds=300):
 c=json.loads((J/'config.json').read_text());now=datetime.now();cut=now-timedelta(seconds=seconds);groups=[]
 for i in range(c['replicas']):
  p=max((J/'logs').glob(f'server.*.replica{i}.*.log'),key=lambda p:p.stat().st_mtime)
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-8000000));lines=f.read().decode(errors='replace').splitlines()
  points=[];retract=0;requests=0;prev=None
  for l in lines:
   m=re.match(r'\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)(?: TP(\d+))?',l)
   if not m:continue
   t=datetime.strptime(m[1],'%Y-%m-%d %H:%M:%S')
   if t>=cut and 'Retract requests' in l and m[2]=='0':
    retract+=1;requests+=int(re.search(r'#retracted_reqs: (\d+)',l)[1])
   if 'Decode batch' not in l:continue
   def val(pattern):
    a=re.search(pattern,l);return float(a[1]) if a else 0
   if t>=cut:
    points.append(dict(rate=val(r'gen throughput \(token/s\): ([\d.]+)'),duration=max(0,(t-max(prev or cut,cut)).total_seconds()),kv=val(r'full token usage: ([\d.]+)'),running=val(r'#running-req: (\d+)'),queue=val(r'#queue-req: (\d+)')))
   prev=t
  duration=sum(x['duration'] for x in points)
  avg=lambda key:sum(x[key]*x['duration'] for x in points)/duration if duration else 0
  groups.append(dict(replica=i,log=str(p),seconds=duration,decode_tokens_per_second=round(avg('rate'),2),avg_running=round(avg('running'),2),avg_queue=round(avg('queue'),2),kv_at_least_95_percent_fraction=round(sum(x['duration'] for x in points if x['kv']>=.95)/duration,3) if duration else 0,retraction_events=retract,retracted_requests=requests))
 progress=[]
 for l in (J/'logs/latest.log').read_text().splitlines():
  try:r=json.loads(l)
  except ValueError:continue
  if 'elapsed_seconds' in r and 'counts' in r:progress.append(r)
 slope=None
 if len(progress)>1:
  end=progress[-1];eligible=[r for r in progress if r['elapsed_seconds']>=end['elapsed_seconds']-seconds];start=eligible[0] if len(eligible)>1 else progress[-2]
  dt=end['elapsed_seconds']-start['elapsed_seconds']
  if dt>0:slope=dict(seconds=dt,success=end['counts'].get('success',0)-start['counts'].get('success',0),conversations_per_second=(end['counts'].get('success',0)-start['counts'].get('success',0))/dt)
 return dict(measured_at=now.isoformat(),window_seconds=seconds,concurrency_per_group=c['concurrency_per_replica'],replicas=c['replicas'],groups=groups,decode_tokens_per_second=sum(x['decode_tokens_per_second'] for x in groups),retraction_events=sum(x['retraction_events'] for x in groups),retracted_requests=sum(x['retracted_requests'] for x in groups),progress_slope=slope,latest_progress=progress[-1] if progress else None)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seconds',type=int,default=300);p.add_argument('--save');a=p.parse_args();r=measure(a.seconds);s=json.dumps(r,indent=2)+'\n'
 if a.save:(J/a.save).write_text(s)
 print(s)
