from pathlib import Path
import fcntl,os,subprocess,sys
root=Path(__file__).resolve().parents[2];work=Path(__file__).resolve().parent
locks=[]
for name in ["LEGOSpec_20261111","LEGOSpec_swa_20261112","LEGOSpec_public"]:
 f=(root.parent/name/"tests/h8002_eval_dispatch.lock").open("a");fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
for mode in sys.argv[1:]:
 with (work/(mode+".log")).open("w") as log:
  result=subprocess.run([sys.executable,"-m","torch.distributed.run","--nproc-per-node=2","--master-port=29782",str(work/"distributed_smoke.py"),mode],cwd=root,env=dict(os.environ,PYTHONPATH=".",CUDA_VISIBLE_DEVICES="2,3",OMP_NUM_THREADS="1"),stdout=log,stderr=subprocess.STDOUT)
 print(mode,result.returncode,flush=True)
 if result.returncode:raise SystemExit(result.returncode)
