"""Split whole conversations round-robin; publish both files atomically."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time


def split(source, destination):
    source, destination = Path(source), Path(destination)
    before = source.stat()
    identity = dict(path=str(source.resolve()), bytes=before.st_size, mtime_ns=before.st_mtime_ns)
    if destination.exists():
        manifest = json.loads((destination/'manifest.json').read_text())
        if manifest['source'] != identity:
            raise ValueError('Existing split belongs to a different source; refusing overwrite')
        for part in manifest['parts']:
            assert (destination/part['file']).stat().st_size == part['bytes']
        print(json.dumps(manifest,ensure_ascii=False),flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=destination.name+'.tmp.',dir=destination.parent))
    handles = [(work/f'part{i}.jsonl').open('wb') for i in range(2)]
    parts = [dict(file=f'part{i}.jsonl',records=0,user_turns=0,bytes=0) for i in range(2)]
    hashes = [hashlib.sha256(),hashlib.sha256()]
    original_hash = hashlib.sha256()
    started = time.monotonic()
    try:
        with source.open('rb') as stream:
            for index,line in enumerate(stream):
                row = json.loads(line)
                # The complete JSON record is copied verbatim, including every
                # system/user turn and all original IDs and metadata.
                part = index % 2
                handles[part].write(line)
                original_hash.update(line)
                hashes[part].update(line)
                parts[part]['records'] += 1
                parts[part]['bytes'] += len(line)
                parts[part]['user_turns'] += sum(m['role']=='user' for m in row['conversations'])
                if (index+1)%250000 == 0:
                    print(json.dumps(dict(records=index+1,seconds=round(time.monotonic()-started,1))),flush=True)
        after = source.stat()
        assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns), 'Source changed while splitting'
        for handle in handles:
            handle.flush();os.fsync(handle.fileno());handle.close()
        for part,digest in zip(parts,hashes):part['sha256']=digest.hexdigest()
        manifest = dict(source=identity,source_sha256=original_hash.hexdigest(),
            method='whole_record_round_robin_line_index_mod_2',parts=parts,
            total_records=sum(p['records'] for p in parts),
            total_user_turns=sum(p['user_turns'] for p in parts))
        (work/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        os.rename(work,destination)
        print(json.dumps(manifest,ensure_ascii=False),flush=True)
    except BaseException:
        for handle in handles:handle.close()
        shutil.rmtree(work)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    split(args.input,args.output_dir)
