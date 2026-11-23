"""Migrate runtime concurrency without discarding or duplicating saved answers."""
import argparse
import json
from pathlib import Path
import shutil
import time

from generate import fingerprint


def atomic_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def migrate(job,concurrency,fingerprint_fn=fingerprint,*,mem_fraction_static=None):
    job=Path(job)
    path=job/'config.json'
    old=json.loads(path.read_text())
    new=dict(old,concurrency_per_replica=concurrency)
    if mem_fraction_static is not None:
        assert 0 < mem_fraction_static < 1
        new['mem_fraction_static']=mem_fraction_static
    changed=sorted(k for k in old.keys()|new.keys() if old.get(k)!=new.get(k))
    if not changed:
        return dict(status='already_configured',concurrency_per_tp_group=concurrency)
    assert set(changed)<={'concurrency_per_replica','mem_fraction_static'}
    old_fp=fingerprint_fn(old)[0]
    new_fp=fingerprint_fn(new)[0]
    output=Path(old['output_dir'])/'default'
    updates=[]
    for part in range(old['input_shards']):
        pin=output/f'part{part}.identity.json'
        if not pin.exists():continue
        identity=json.loads(pin.read_text())
        if identity['fingerprint']!=old_fp:
            raise ValueError(f'{pin}: model, client, input or sampling fingerprint changed; refusing migration')
        source=Path(old['input_files'][part]);stat=source.stat()
        assert identity['input']==str(source)
        assert identity['input_bytes']==stat.st_size and identity['input_mtime_ns']==stat.st_mtime_ns
        data=output/f'part{part}.jsonl'
        with data.open('rb') as stream:
            records=sum(line.endswith(b'\n') for line in stream)
        updates.append((pin,dict(identity,fingerprint=new_fp),dict(part=part,accepted_records=records,bytes=data.stat().st_size)))
    # All pins are validated before making any mutation. Data JSONL files are
    # never rewritten; the normal resume reader recovers an interrupted tail.
    backup=job/f'config.before_concurrency{concurrency}.json'
    if backup.exists():raise ValueError(f'Backup already exists: {backup}')
    shutil.copyfile(path,backup)
    for pin,identity,_ in updates:
        shutil.copyfile(pin,pin.with_name(pin.stem+f'.before_concurrency{concurrency}.json'))
    atomic_json(path,new)
    for pin,identity,_ in updates:atomic_json(pin,identity)
    receipt=dict(status='migrated',time=time.time(),changed_keys=changed,
        old_concurrency_per_tp_group=old['concurrency_per_replica'],concurrency_per_tp_group=concurrency,
        old_mem_fraction_static=old.get('mem_fraction_static',0.85),mem_fraction_static=new.get('mem_fraction_static',0.85),
        per_node_concurrency=new['replicas']*concurrency,total_concurrency=2*new['replicas']*concurrency,
        old_fingerprint=old_fp,new_fingerprint=new_fp,preserved_outputs=[x[2] for x in updates])
    atomic_json(job/f'concurrency{concurrency}_migration.json',receipt)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--concurrency',type=int,required=True)
    parser.add_argument('--mem-fraction-static',type=float)
    args=parser.parse_args()
    assert args.concurrency>0
    job=Path(__file__).resolve().parent
    for state in (job/'logs').glob('launch.*.json'):
        current=json.loads(state.read_text())
        if current['status'] not in ('failed','completed'):
            raise RuntimeError(f'Stop the existing generation controller before migration: {state}')
    print(json.dumps(migrate(job,args.concurrency,mem_fraction_static=args.mem_fraction_static),ensure_ascii=False))
