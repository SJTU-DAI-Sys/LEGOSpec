from pathlib import Path
import json, hashlib, ast, difflib
R=Path(__file__).resolve().parents[2]
A=R/'tests/deferred_objectives_20261122'
manifest=json.loads((A/'manifest.json').read_text())
patch=[]
for item in manifest:
    archived=R/item['archive']; active=R/item['source']
    assert hashlib.sha256(archived.read_bytes()).hexdigest()==item['sha256'],archived
    before=archived.read_text(); after=active.read_text() if active.exists() else ''
    if active.suffix=='.py' and active.exists(): ast.parse(after)
    patch.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=item['archive'],tofile=item['source']))
(A/'changes.diff').write_text(''.join(patch))
assert not (R/'legospec/modeling/draft/composable/objective/survival.py').exists()
for p in (R/'legospec').rglob('*.py'):
    assert not any(x in p.read_text() for x in ('dpal_plus_plus','tv_survival','opd_survival','objective.survival','divide_own_overlap')),p
print(f'{len(manifest)} archived files verified; deferred names absent from active Python sources.')
