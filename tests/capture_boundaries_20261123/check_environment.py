from pathlib import Path
import tarfile,json,importlib,importlib.metadata
root=Path.cwd();work=root/'tests/capture_boundaries_20261123'
results={'sglang_version':importlib.metadata.version('sglang'),'transformers_version':importlib.metadata.version('transformers')}
with tarfile.open(work/'before.tgz') as archive:
 source=archive.extractfile('legospec/modeling/target/sglang_gemma4_capture.py').read().decode()
try:
 exec(compile(source,'before/sglang_gemma4_capture.py','exec'),{'__name__':'capture_before'})
 results['before']='import passed'
except ImportError as exc:results['before']=f'{type(exc).__name__}: {exc}'
try:
 importlib.import_module('legospec.modeling.target.sglang_backend.capture')
 results['after']='import passed'
except ImportError as exc:results['after']=f'{type(exc).__name__}: {exc}'
paths={}
for p in (root.parent/'envs').glob('*/lib/python3.11/site-packages/sglang/srt/distributed/bootstrap.py'):
 paths[str(p)]=True
results['other_native_bootstrap_files']=paths
(work/'environment.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
