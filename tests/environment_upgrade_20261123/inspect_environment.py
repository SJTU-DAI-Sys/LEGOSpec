from pathlib import Path
import sys,importlib.metadata as m,json,urllib.request,concurrent.futures,shutil
root=Path.cwd();work=root/'tests/environment_upgrade_20261123'
print('python',sys.executable,'prefix',sys.prefix)
for name in ['sglang','torch','transformers','triton','sgl-kernel','flash-linear-attention','tilelang','flash-attn','causal-conv1d']:
 try:
  d=m.distribution(name);print(name,d.version,str(d.locate_file('')))
 except m.PackageNotFoundError:print(name,'absent')
(work/'freeze_before.txt').write_text('\n'.join(sorted(f'{d.metadata["Name"]}=={d.version}' for d in m.distributions()))+'\n')
print('uv', [(str(p),p.exists()) for p in [Path.home()/'.local/bin/uv',Path('/usr/local/bin/uv')]])
print('disk',shutil.disk_usage(root))
urls={'sglang':'https://pypi.org/pypi/sglang/json','latest':'https://api.github.com/repos/sgl-project/sglang/releases/latest','main':'https://raw.githubusercontent.com/sgl-project/sglang/main/python/pyproject.toml'}
def get(item):
 key,url=item
 try:
  raw=urllib.request.urlopen(url,timeout=30).read().decode();(work/(key+'.json' if key!='main' else 'sglang_main.toml')).write_text(raw)
  if key=='sglang':
   j=json.loads(raw);return key,{'version':j['info']['version'],'requires':j['info']['requires_dist'],'recent_versions':list(j['releases'])[-18:]}
  if key=='latest':
   j=json.loads(raw);return key,{k:j[k] for k in ['tag_name','published_at','html_url','body']}
  return key,raw[:300]
 except Exception as e:return key,str(e)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 for result in pool.map(get,urls.items()):print(json.dumps(result))
