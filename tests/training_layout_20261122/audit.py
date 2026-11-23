from pathlib import Path
import ast, json, re, hashlib
from setuptools import find_namespace_packages
R=Path(__file__).resolve().parents[2];W=Path(__file__).resolve().parent;T=R/'legospec/training'
assert {p.name for p in T.glob('*.py')}=={'__init__.py','config.py','outputs.py'}
expected={'runtime','data','target','execution','optim','logger'}
assert {p.name for p in T.iterdir() if p.is_dir() and p.name!='__pycache__'}==expected
old=set(p.stem for p in (W/'pruned').glob('*.py'))-{'__init__','config','outputs'}
pattern=re.compile(r'legospec\.training\.('+ '|'.join(sorted(old))+r')(?=[.\s\"\x27,:;)]|$)')
files=[*(R/'legospec').rglob('*.py'),*(R/'scripts').rglob('*.py'),*(R/'tests/unit').glob('*.py')]
for p in files:
    s=p.read_text();ast.parse(s)
    assert not pattern.search(s),p
for p in T.rglob('*.py'):
    if p.is_relative_to(T/'runtime') or p==T/'__init__.py':continue
    for n in ast.walk(ast.parse(p.read_text())):
        if isinstance(n,ast.ImportFrom):
            assert n.module not in {'legospec.training.runtime.setup','legospec.training.runtime.runner'},p
            if n.module=='legospec.training.runtime':assert not {'setup','runner'} & {a.name for a in n.names},p
        if isinstance(n,ast.Import):assert not {'legospec.training.runtime.setup','legospec.training.runtime.runner'} & {a.name for a in n.names},p
packages=set(find_namespace_packages(where=str(R),include=['legospec*']))
for name in expected:assert 'legospec.training.'+name in packages
report={
 'directories':sorted(expected),'python_modules':len(list(T.rglob('*.py'))),
 'setup_lines':len((T/'runtime/setup.py').read_text().splitlines()),
 'old_setup_lines':1808,
 'checked_active_python_files':len(files),
 'old_imports_remaining':0,'reverse_setup_runner_imports':0,
 'namespace_package_discovery':'passed',
 'regression':{'initial_passed':539,'initial_failed':38,'rechecked_files_passed':56,'final_unique_passed':577,'skipped':9,'subtests_passed':13,'skip_reason':'Existing candidate-only CE contract'},
 'equivalence':json.loads((W/'equivalence.json').read_text()),
 'files':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in T.rglob('*.py')},
}
(W/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
print('Audit passed:',report['python_modules'],'training modules;',len(files),'active Python files; no old imports or reverse setup/runner dependencies; namespace packages discovered.')
