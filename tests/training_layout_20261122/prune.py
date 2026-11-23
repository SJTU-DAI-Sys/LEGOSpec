from pathlib import Path
import ast, shutil
R=Path(__file__).resolve().parents[2]
W=Path(__file__).resolve().parent
T=R/'legospec/training'
D=R/'development/objectives/training_snapshots_20261122'
# Preserve the remaining integration before removing dormant component-objective code.
for p in [*T.glob('*.py'),R/'legospec/modeling/draft/composable/objective/base.py']:
    dest=D/p.relative_to(R);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
def cut(s,pred,otherwise=False):
    lines=s.splitlines(True); ranges=[]
    for n in ast.walk(ast.parse(s)):
        if pred(n):
            first=min([n.lineno]+[d.lineno for d in getattr(n,'decorator_list',[])])-1
            value=''
            if otherwise and getattr(n,'orelse',[]):
                value=''.join(l[4:] if l.strip() else l for l in lines[n.orelse[0].lineno-1:n.orelse[-1].end_lineno])
            ranges.append((first,n.end_lineno,value))
    chosen=[]
    for a,b,v in sorted(ranges,key=lambda x:(x[0],-x[1])):
        if chosen and a<chosen[-1][1]:continue
        chosen.append((a,b,v))
    for a,b,v in reversed(chosen):lines[a:b]=[v]
    return ''.join(lines)
def assign_names(n):
    return {x.id if isinstance(x,ast.Name) else x.attr for t in (n.targets if isinstance(n,ast.Assign) else [n.target]) for x in [t] if isinstance(x,(ast.Name,ast.Attribute))}
def has(n,word):return word in ast.unparse(n)
def put(name,s):ast.parse(s);(T/name).write_text(s)
s=(T/'step.py').read_text()
s=cut(s,lambda n:isinstance(n,ast.If) and (has(n.test,'self.component_normalization') or has(n.test,'opd_accepted_lengths')))
s=cut(s,lambda n:isinstance(n,ast.Assign) and bool(assign_names(n)&{'component_normalization','component_terms','component_sum_metrics','accepted_chunks','replay','source_layout'}))
s=s.replace('from legospec.training.component_reduction import combine_component_terms\n','')
for line in ['        opd_accepted_lengths=None,\n','        opd_global_denominators=None,\n','        opd_reference_samples=None,\n','        metadata_only=False,\n','                opd_global_denominators=opd_global_denominators,\n','                opd_reference_samples=opd_reference_samples,\n','                metadata_only=metadata_only,\n','                replay,\n','                first_chunk,\n','                start == 0,\n']:
    s=s.replace(line,'')
s=s.replace('if not metadata_only and (','if (')
put('step.py',s)
s=(T/'anchor_opd.py').read_text()
s=cut(s,lambda n:isinstance(n,ast.If) and has(n.test,'metadata_only'))
for key in ['opd_global_denominators','opd_reference_samples','metadata_only']:
    s=s.replace(f'    {key}=None,\n','').replace(f'    {key}=False,\n','')
put('anchor_opd.py',s)
s=(T/'batch.py').read_text()
s=cut(s,lambda n:isinstance(n,ast.ClassDef) and n.name=='OPDReplayMetadata')
s=cut(s,lambda n:isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name) and n.target.id in {'opd_metadata','opd_component_denominators','opd_reference_samples','opd_inputs_prepared'})
put('batch.py',s)
s=(T/'windows.py').read_text()
s=cut(s,lambda n:isinstance(n,ast.FunctionDef) and n.name in {'prepare_anchor_opd_window','prepare_opd_window'})
s=s.replace('from legospec.training.component_reduction import global_component_counts\n','').replace('from legospec.training.batch import BlockLayout, OPDReplayMetadata, prepare_layout\n','')
put('windows.py',s)
(T/'component_reduction.py').unlink()
s=(T/'setup.py').read_text()
s=cut(s,lambda n:isinstance(n,ast.If) and (has(n.test,'needs_component_normalization') or has(n.test,'prepared.opd_metadata')))
# The previous broad predicate also matched packet_metadata's condition; restore that condition without its inactive clause.
original=(D/'legospec/training/setup.py').read_text()
oldfn=next(n for n in ast.parse(original).body if isinstance(n,ast.FunctionDef) and n.name=='packet_metadata')
newfn=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name=='packet_metadata')
orig='\n'.join(original.splitlines()[oldfn.lineno-1:oldfn.end_lineno])
a=orig.index('        and (\n            not any(');b=orig.index('\n    ):',a)
orig=orig[:a]+orig[b:]
lines=s.splitlines(True);lines[newfn.lineno-1:newfn.end_lineno]=[orig+'\n'];s=''.join(lines)
s=cut(s,lambda n:isinstance(n,ast.If) and has(n.test,'prepared.opd_inputs_prepared'),otherwise=True)
s=s.replace('                opd_global_denominators=data.opd_component_denominators,\n','').replace('                opd_reference_samples=data.opd_reference_samples,\n','')
s=s.replace('    component = {}\n','').replace('        **component,\n','')
a=s.index('    if getattr(session.args, "opd_data_mode", "teacher_forced") == "sampled_anchor":',s.index('def prepare_optimizer_window'))
b=s.index('    recipe = session.draft.config.composable_config',a)
s=s[:a]+'    if getattr(session.args, "opd_data_mode", "teacher_forced") == "sampled_anchor":\n        return None\n'+s[b:]
put('setup.py',s)
s=(T/'planning.py').read_text()
a=s.index('        self.metadata_counts = not any(');b=s.index('        self.orders, self.windows',a)
s=s[:a]+'        self.metadata_counts = True\n'+s[b:];put('planning.py',s)
s=(T/'outputs.py').read_text();a=s.index('    # Optional native terms');s=s[:a];put('outputs.py',s)
p=R/'legospec/modeling/draft/composable/objective/base.py';s=p.read_text();s=s.replace('    opd_accepted_lengths: Optional[torch.Tensor] = None\n','').replace('    opd_distribution_terms: Optional[tuple] = None\n','');p.write_text(s)
# Keep the cleaned source as the deterministic input to the mechanical relocation.
(W/'pruned').mkdir(exist_ok=True)
for p in T.glob('*.py'):shutil.copy2(p,W/'pruned'/p.name)
print('Dormant OPD integration archived and removed.')
