from pathlib import Path
import json,urllib.request,concurrent.futures,importlib.metadata as m
root=Path.cwd();work=root/'tests/environment_upgrade_20261123'
files=['python/sglang/srt/managers/overlap_utils.py','python/sglang/srt/model_executor/model_runner.py','python/sglang/srt/model_executor/forward_batch_info.py','python/sglang/srt/models/gemma4_unified.py','python/sglang/srt/runtime_context/__init__.py','python/sglang/srt/models/qwen3.py']
def fetch(path):
 try:
  b=urllib.request.urlopen('https://raw.githubusercontent.com/sgl-project/sglang/v0.5.16/'+path,timeout=25).read();out=work/'upstream'/path;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(b);return path,len(b)
 except Exception as e:return path,str(e)
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
 for r in pool.map(fetch,files):print(r)
for folder in (root.parent/'envs').iterdir():
 site=folder/'lib/python3.11/site-packages'
 if site.is_dir():
  print('env',folder.name,'torch',[p.name for p in site.glob('torch-*.dist-info')])
print('model_roots')
for base in [root.parent,Path('/share/dai-sys/wanghanzhen/models'),Path('/mnt/dai-sys/wanghanzhen/.cache/huggingface/hub')]:
 if base.is_dir():
  print(str(base),[p.name for p in base.iterdir() if any(x in p.name.lower() for x in ['model','cache','gemma','qwen'])])
