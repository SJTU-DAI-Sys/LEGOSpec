from pathlib import Path
import fcntl,os,subprocess,sys
root=Path(__file__).resolve().parents[2];work=Path(__file__).resolve().parent
locks=[]
for name in ["LEGOSpec_20261111","LEGOSpec_swa_20261112","LEGOSpec_public"]:
 handle=(root.parent/name/"tests/h8002_eval_dispatch.lock").open("a")
 fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(handle)
gpu="2"
assert int(subprocess.check_output(["nvidia-smi","-i",gpu,"--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True))<500
with (work/"regression.log").open("w") as log:
 result=subprocess.run([sys.executable,"-m","pytest","-q","-rs","--tb=short","--basetemp=tests/defer_opd_20261123/tmp","tests/unit"],cwd=root,env=dict(os.environ,PYTHONPATH=".",CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS="1"),stdout=log,stderr=subprocess.STDOUT)
print((work/"regression.log").read_text()[-5000:]);raise SystemExit(result.returncode)
