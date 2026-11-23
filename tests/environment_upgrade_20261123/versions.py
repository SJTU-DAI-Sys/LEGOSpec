from pathlib import Path
import json,urllib.request,concurrent.futures
work=Path('tests/environment_upgrade_20261123')
def request(url):return urllib.request.urlopen(url,timeout=25).read()
def check(v):
 try:
  j=json.loads(request(f'https://pypi.org/pypi/sglang/{v}/json'));req=[r for r in j['info']['requires_dist'] if r.startswith(('torch','transformers','sgl','tilelang','flashinfer')) and 'extra ==' not in r]
  path=f'https://raw.githubusercontent.com/sgl-project/sglang/v{v}/python/sglang/srt/distributed/bootstrap.py'
  try:
   b=request(path);native=True;(work/f'bootstrap-{v}.py').write_bytes(b)
  except Exception:native=False
  return dict(version=v,native=native,requires=req)
 except Exception as e:return dict(version=v,error=str(e))
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
 data=list(pool.map(check,['0.5.12','0.5.14','0.5.16','0.5.18','0.5.20','0.5.21','0.5.22','0.5.23']))
(work/'versions.json').write_text(json.dumps(data,indent=2));print(json.dumps(data,indent=2))
