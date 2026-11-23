"""Own four TP2 servers, validate them, then regenerate assigned input parts."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from generate import http

JOB=Path(__file__).resolve().parent

def launch_state(plan,status,controller_pid,server_pids,client_pid,**extra):
    return dict(plan,status=status,updated_at=time.time(),controller_pid=controller_pid,
        server_pids=server_pids,client_pid=client_pid,**extra)

def server_environment(config,hostname):
    # Build products and Ninja dependency databases must not be concurrently
    # rewritten by independent GPU hosts on the shared project filesystem.
    cache=str(Path(config.get('cache_root',JOB/'cache'))/hostname/'sglang')
    return dict(SGLANG_MAX_THINK_TOKENS=str(config['thinking_budget']),
        SGLANG_CACHE_DIR=cache,FLASHINFER_WORKSPACE_BASE=cache)

def server_command(config, replica):
    port=config['base_port']+replica
    return [sys.executable,'-m','sglang.launch_server','--model-path',config['model'],
        '--served-model-name','qwen38-27b-regen','--host','127.0.0.1','--port',str(port),
        '--tp-size',str(config['tp_size']),'--dtype','bfloat16',
        '--dist-timeout',str(config['dist_timeout']),
        '--reasoning-parser','qwen3','--enable-strict-thinking','--grammar-backend','xgrammar',
        '--attention-backend','triton','--linear-attn-backend','triton',
        '--mamba-ssm-dtype','float32','--kv-cache-dtype','bfloat16',
        '--context-length',str(config['context_length']),'--mem-fraction-static',str(config.get('mem_fraction_static',0.85)),
        '--chunked-prefill-size','4096','--max-running-requests',str(config['concurrency_per_replica']),
        '--max-mamba-cache-size',str(config['concurrency_per_replica']),
        '--disable-radix-cache','--disable-overlap-schedule',
        '--cuda-graph-backend-decode','disabled','--cuda-graph-backend-prefill','disabled']

def stop(process):
    if process is None:return
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL);process.wait()
    try:os.killpg(process.pid,signal.SIGKILL)
    except ProcessLookupError:pass

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--parts',default=os.environ.get('PARTS','0,1'))
    p.add_argument('--run-tag',default=os.environ.get('RUN_TAG','default'))
    p.add_argument('--smoke-only',action='store_true')
    p.add_argument('--reuse-smoke',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();config=json.loads((JOB/'config.json').read_text())
    hostname=socket.gethostname();logs=JOB/'logs';logs.mkdir(exist_ok=True)
    parts=[int(x) for x in a.parts.split(',')]
    assert parts and len(parts)==len(set(parts)) and all(0<=x<config['input_shards'] for x in parts)
    import re
    assert re.fullmatch(r'[A-Za-z0-9_.-]+',a.run_tag), 'Invalid RUN_TAG'
    plan=dict(host=hostname,parts=parts,run_tag=a.run_tag,
        server_env=server_environment(config,hostname),
        servers=[server_command(config,i) for i in range(config['replicas'])])
    if a.dry_run:
        print(json.dumps(plan,indent=2));return
    lock=open(f'/tmp/legospec_qwen38_regen_{os.getuid()}.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for replica in range(config['replicas']):
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            probe.bind(('127.0.0.1',config['base_port']+replica))
    import torch
    count=torch.cuda.device_count()
    required=config['replicas']*config['tp_size']
    assert count==required,f'Requires {required} visible GPUs, found {count}'
    memory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    used={int(row.split(',')[0]):int(row.split(',')[1]) for row in memory.splitlines()}
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    gpu_ids=visible.split(',') if visible else [str(i) for i in range(count)]
    assert len(gpu_ids)==required and all(x.isdigit() for x in gpu_ids),'Use numeric visible GPU IDs matching the replica count'
    assert all(used[int(i)]<1024 for i in gpu_ids),f'GPUs are busy; stop the intended old job/keeper first: {memory}'
    models=[];client=None;handles=[];interrupted=False
    def interrupt(number,frame):
        raise KeyboardInterrupt(f'Received signal {number}')
    signal.signal(signal.SIGTERM,interrupt)
    signal.signal(signal.SIGINT,interrupt)
    state_file=logs/f'launch.{hostname}.json'
    def save(status,**extra):
        state=launch_state(plan,status,os.getpid(),[p.pid for p in models],client.pid if client else None,**extra)
        temporary=state_file.with_suffix('.tmp')
        temporary.write_text(json.dumps(state,indent=2)+'\n')
        temporary.replace(state_file)
    try:
        for i,command in enumerate(plan['servers']):
            env=dict(os.environ,**plan['server_env'],CUDA_VISIBLE_DEVICES=','.join(gpu_ids[2*i:2*i+2]),
                TRITON_CACHE_DIR=str(Path(config.get('cache_root',JOB/'cache'))/hostname/f'gpu{i}'/'triton'))
            path=logs/f'server.{hostname}.replica{i}.{int(time.time())}.log'
            handle=path.open('ab');handles.append(handle)
            models.append(subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,
                stdout=handle,stderr=subprocess.STDOUT,start_new_session=True))
        save('starting_servers')
        deadline=time.monotonic()+config['startup_timeout']
        while True:
            if any(p.poll() is not None for p in models):
                raise RuntimeError('A server exited during startup; inspect this host server logs')
            ready=0
            for i in range(config['replicas']):
                try:
                    info=http(f'http://127.0.0.1:{config["base_port"]+i}/model_info',timeout=2)
                    ready+=info.get('model_path')==config['model']
                except Exception:pass
            if ready==config['replicas']:break
            if time.monotonic()>deadline:raise TimeoutError(f'Model server startup exceeded {config["startup_timeout"]} seconds')
            time.sleep(3)
        save('awaiting_verified_import')
        while not ((JOB/'migration_verified.json').exists() and (Path(config['output_dir'])/'default/smoke.parts0-1.json').exists()):
            if any(p.poll() is not None for p in models):
                raise RuntimeError('A server exited while waiting for the data transfer')
            time.sleep(5)
        save('smoke_then_generation')
        command=[sys.executable,'-u',str(JOB/'generate.py'),'--parts',a.parts,'--run-tag',a.run_tag,
            '--ports',','.join(str(config['base_port']+i) for i in range(config['replicas']))]
        if a.smoke_only:command.append('--smoke-only')
        if a.reuse_smoke:command.append('--reuse-smoke')
        client=subprocess.Popen(command,stdin=subprocess.DEVNULL,start_new_session=True)
        save('smoke_then_generation')
        while client.poll() is None:
            if any(p.poll() is not None for p in models):
                raise RuntimeError('A model server exited; stopping only this launch')
            time.sleep(2)
        assert client.returncode==0,f'Generator failed with {client.returncode}'
        save('completed')
    except BaseException as error:
        save('failed',error=repr(error));raise
    finally:
        stop(client)
        for process in models:stop(process)
        for handle in handles:handle.close()

if __name__=='__main__':main()
