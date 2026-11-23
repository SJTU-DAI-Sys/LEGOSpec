from pathlib import Path
import os,fcntl,subprocess,json
root=Path(__file__).resolve().parents[2];work=Path(__file__).resolve().parent
locks=[]
for name in ['LEGOSpec_20261111','LEGOSpec_swa_20261112','LEGOSpec_public']:
 f=(root.parent/name/'tests/h8002_eval_dispatch.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
assert int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))<500
files=['test_attention_taxonomy.py','test_gdn_shared_block_attention.py','test_gdn_block_attention.py','test_layer_window_overrides.py','test_training_planning.py','test_composable_eagle3.py','test_e2e_tv_causal.py','test_mla_absorb.py','test_recipe_catalog.py','test_dsa_gather.py','test_component_boundaries.py','test_training_json_projection.py','test_checkpoint_resume.py','test_execution_modes.py','test_packed_protocol.py','test_anchor_opd.py']
with (work/'pytest.log').open('w') as log:
 result=subprocess.run([os.sys.executable,'-m','pytest','-q','--tb=short','--basetemp=tests/linear_ownership_20261122/tmp',*[f'tests/unit/{f}' for f in files]],cwd=root,env=dict(os.environ,PYTHONPATH='.',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT)
print((work/'pytest.log').read_text()[-8000:]);raise SystemExit(result.returncode)
