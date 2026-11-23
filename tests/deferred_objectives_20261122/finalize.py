from pathlib import Path
import json, shutil, hashlib
R=Path(__file__).resolve().parents[2]
a=R/'tests/deferred_objectives_20261122'
p=Path('legospec/training/component_reduction.py')
d=R/'development/objectives/snapshots'/p
d.parent.mkdir(parents=True,exist_ok=True)
shutil.copy2(R/p,d)
m=json.loads((a/'manifest.json').read_text())
m.append(dict(source=str(p),archive=str(d.relative_to(R)),sha256=hashlib.sha256(d.read_bytes()).hexdigest()))
(a/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
s=(R/p).read_text().replace('metrics[branch + "/opd_survival/" + key]', 'metrics[branch + "/" + objective.recipe["type"] + "/" + key]')
(R/p).write_text(s)
# Archived mixed tests retain their historical imports and fixtures; active tests no longer need these imports.
p=R/'tests/unit/test_anchor_opd.py'
s=p.read_text().replace('import copy\n','').replace('from types import SimpleNamespace\n','')
p.write_text(s)
p=R/'tests/development/objectives/README.md'
p.write_text(p.read_text().replace('../../../development','../../../development'))
