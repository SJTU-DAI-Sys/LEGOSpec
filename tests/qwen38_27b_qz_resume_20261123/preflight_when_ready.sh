#!/usr/bin/env bash
set -euo pipefail
JOB=/mnt/dai-sys/wanghanzhen/projects/DAC27/LEGOSpec_public/tests/qwen38_27b_qz_resume_20261123
source "$JOB/activate.sh"
"$PYTHON_BIN" -u - <<'PY'
import sys,time,json
from pathlib import Path
j=Path('/mnt/dai-sys/wanghanzhen/projects/DAC27/LEGOSpec_public/tests/qwen38_27b_qz_resume_20261123');sys.path.insert(0,str(j))
from generate import http
endpoints=[f'http://127.0.0.1:{32281+i}' for i in range(json.loads((j/'config.json').read_text())['replicas'])]
for _ in range(720):
 try:
  for endpoint in endpoints:http(endpoint+'/model_info',timeout=2)
  print('All four endpoints ready',flush=True);break
 except Exception:time.sleep(5)
else:raise RuntimeError('Server readiness timeout')
PY
"$PYTHON_BIN" -u "$JOB/generate.py" --parts 0,1 --ports 32281,32282,32283 --smoke-only --run-tag h800_preflight
"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import json,shutil
j=Path('/mnt/dai-sys/wanghanzhen/projects/DAC27/LEGOSpec_public/tests/qwen38_27b_qz_resume_20261123');c=json.loads((j/'config.json').read_text());o=Path(c['output_dir']);shutil.copy2(o/'h800_preflight/smoke.parts0-1.json',o/'default/smoke.parts0-1.json')
print('H800 smoke receipt published')
PY
