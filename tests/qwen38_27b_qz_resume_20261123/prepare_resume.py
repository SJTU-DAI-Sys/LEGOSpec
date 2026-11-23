from pathlib import Path
import json,hashlib,shutil,sys,time
JOB=Path(__file__).resolve().parent
old=JOB/'imported/LEGOSpec/tests/qwen38_27b_prompts1411k_think2048_medium_20261121'
manifest=json.loads((JOB/'source_transfer_manifest.json').read_text())
results=[]
for entry in manifest['files']:
 p=JOB/'imported'/entry['path'];h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(8*1024*1024):h.update(b)
 assert p.stat().st_size==entry['bytes'] and h.hexdigest()==entry['sha256'],str(p)
 results.append(dict(path=str(p),bytes=p.stat().st_size,records=entry['records'],sha256=entry['sha256']))
 print('VERIFIED',entry['path'],entry['records'],flush=True)
for name in ['generate.py','launch.py','test_generation.py','set_concurrency.py','split_input.py']:
 if not (JOB/name).exists():shutil.copy2(old/name,JOB/name)
shutil.copytree(old/'official_reference',JOB/'official_reference',dirs_exist_ok=True)
config=json.loads((old/'config.json').read_text())
config['model']='/mnt/dai-sys/wanghanzhen/models/Qwen/Qwen3.8-27B'
config['input_file']=str(JOB/'imported/training_data/prompts/prompts_system_user_multi.jsonl')
config['input_files']=[str(JOB/f'imported/training_data/prompts/prompts_system_user_multi_2parts/part{p}.jsonl') for p in range(2)]
config['output_dir']=str(JOB/'imported/training_data/regen/qwen38_27b_prompts1411k_T1_medium_think2048_max5120_20261121')
config['cache_root']=str(JOB/'cache')
(JOB/'config.json').write_text(json.dumps(config,indent=2)+'\n')
sys.path.insert(0,str(JOB))
from generate import fingerprint
fp,hashes=fingerprint(config)
root=Path(config['output_dir'])/'default'
for p in range(2):
 pin=root/f'part{p}.identity.json';shutil.copy2(pin,JOB/f'part{p}.qz_identity.json')
 source=Path(config['input_files'][p]);st=source.stat()
 identity=dict(fingerprint=fp,input=str(source),input_bytes=st.st_size,input_mtime_ns=st.st_mtime_ns,part=p,input_shards=2)
 pin.write_text(json.dumps(identity,indent=2)+'\n')
# Preserve QZ-specific receipts as provenance; H800 must run its own smoke.
for path in root.glob('smoke.parts*.json'):
 shutil.copy2(path,JOB/(path.stem+'.qz.json'))
receipt=dict(created_at=time.time(),source='QZ',destination='H800_2',verified_files=results,successful_records=sum(e['records'] for e in manifest['files'] if e['path'].endswith(('/part0.jsonl','/part1.jsonl')) and '/regen/' in e['path']),runtime_fingerprint=fp,model_metadata=hashes,parameters={k:config[k] for k in ['temperature','top_p','top_k','min_p','presence_penalty','repetition_penalty','reasoning_effort','thinking_budget','max_new_tokens','concurrency_per_replica','tp_size','mem_fraction_static']})
(JOB/'migration_verified.json').write_text(json.dumps(receipt,indent=2)+'\n');print('RESUME_READY',receipt['successful_records'],flush=True)
