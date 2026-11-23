from pathlib import Path
from collections import defaultdict
import ast, symtable, json, re, shutil, tarfile
R=Path(__file__).resolve().parents[2];W=Path(__file__).resolve().parent
P='legospec.training.'
defmod={
'__init__':'__init__','config':'config','outputs':'outputs',
'setup':'runtime.setup','runner':'runtime.runner','state':'runtime.state','checkpoint':'runtime.checkpoint',
'batch':'data.batch','block':'data.batch','layout':'data.layout','planning':'data.planning','prefetch':'data.prefetch',
'anchor_opd':'execution.sampled','step':'execution.step','supervision':'execution.supervision','windows':'data.planning',
'optimizer':'optim.optimizer','muon':'optim.muon','fsdp_layout':'optim.fsdp_layout',
'metrics':'logger.metrics','block_metrics':'logger.metrics','logging':'logger.logging'}
special={}
def own(old,new,names):
    for n in names.split():special[P+old,n]=P+new
own('setup','data.loader','_build_processed_dataset _has_valid_anchor_supervision _prepare_dataloader build_train_dataloader build_hidden_cache_dataloader training_data_identity FixedCollator _build_target_loader build_feature_dataloader build_loader')
own('setup','data.batch','select_tp_rank_batch anchor_layout')
own('setup','target.features','required_hidden_layer_ids hidden_states_to_cuda project_cached_target_logits load_cached_target_data build_target_model build_target_and_components _select_mask_token_id resolve_tokenizer_and_components concatenate_target_hidden final_target_hidden teacher_target_hidden prepare_target_features')
own('setup','target.streaming','_broadcast_input _packet_spec _target_packet packet_metadata target_proxies packet_spec empty_packet produce_packet target_epoch_offset prepare_target_batches annotate_target_packet')
own('setup','runtime.checkpoint','topology_identity validate_resume_topology _load_common_state training_metadata checkpoint_metadata')
own('setup','runtime.state','phases')
own('setup','runtime.runner','enter_phase')
own('setup','logger.logging','detail_metrics log_metrics')
own('setup','execution.step','compute_step _step_values _anchor_values')
own('setup','execution.windows','prepare_optimizer_window release_batch')
own('runner','target.streaming','run_target run_target_phase')
own('optimizer','optim.scheduler','_enable_get_lr_call TwoStageScheduler DelayerScheduler WarmupScheduler WarmupDelayerScheduler CosineAnnealingLR CosineAnnealingWarmupLR')
sources={P+p.stem:p.read_text() for p in (W/'pruned').glob('*.py')}
# Import rewriting resolves each symbol's actual owner, not just its old file.
exports={}; blocks=defaultdict(list); import_candidates=defaultdict(list)
def bindings(n):
    if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):return [n.name]
    if isinstance(n,(ast.Assign,ast.AnnAssign)):
        return [x.id for t in (n.targets if isinstance(n,ast.Assign) else [n.target]) for x in ast.walk(t) if isinstance(x,ast.Name) and isinstance(x.ctx,ast.Store)]
    return []
def globals_used(text):
    table=symtable.symtable(text,'<relocate>','exec');names=set()
    def rec(t):
        names.update(s.get_name() for s in t.get_symbols() if s.is_referenced() and (s.is_global() or t.get_type()=='module'))
        for child in t.get_children():rec(child)
    rec(table)
    # Include postponed type annotations as dependencies.
    for n in ast.walk(ast.parse(text)):
        a=n.annotation if isinstance(n,(ast.AnnAssign,ast.arg)) else n.returns if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) else None
        if a:names.update(x.id for x in ast.walk(a) if isinstance(x,ast.Name))
    return names
for old,s in sources.items():
    for n in ast.parse(s).body:
        for name in bindings(n):exports[old,name]=special.get((old,name),P+defmod[old.rsplit('.',1)[-1]])
for old,s in sources.items():
    lines=s.splitlines(True); nodes=ast.parse(s).body
    imports=[n for n in nodes if isinstance(n,(ast.Import,ast.ImportFrom)) and not (isinstance(n,ast.ImportFrom) and n.module=='__future__')]
    for n in nodes:
        if isinstance(n,(ast.Import,ast.ImportFrom)) or (isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str)):continue
        names=bindings(n);dest=exports[old,names[0]] if names else P+defmod[old.rsplit('.',1)[-1]]
        start=min([n.lineno]+[d.lineno for d in getattr(n,'decorator_list',[])])-1
        body=''.join(lines[start:n.end_lineno]); used=globals_used(body)
        blocks[dest].append((old,body))
        for imp in imports:
            if isinstance(imp,ast.ImportFrom):
                aliases=[a for a in imp.names if (a.asname or a.name) in used]
                if aliases:import_candidates[dest].append(ast.unparse(ast.ImportFrom(module=imp.module,names=aliases,level=imp.level)))
            else:
                aliases=[a for a in imp.names if (a.asname or a.name.split('.')[0]) in used]
                if aliases:import_candidates[dest].append(ast.unparse(ast.Import(names=aliases)))
        for name in used:
            owner=exports.get((old,name))
            if owner and owner!=dest:import_candidates[dest].append(f'from {owner} import {name}')
# Explicit module descriptions are more accurate than the old combined module docstrings.
docs={
'runtime.setup':'Build draft models and assemble one training session; no batch execution.',
'runtime.runner':'Own training iteration, backward/update timing and resource teardown.',
'runtime.state':'Training session, phase definitions and resumable cursors.',
'runtime.checkpoint':'Export models and save/restore complete rank-local training state.',
'data.loader':'Build online JSON and cached-hidden training loaders and collators.',
'data.batch':'Sample anchors and prepare labels, masks and packed execution batches.',
'data.layout':'Occurrence and packed-layout contracts and index construction.',
'data.planning':'Plan sample order, balanced work and token-only supervision windows.',
'data.prefetch':'Bounded ordered CPU prefetch without committing training cursors.',
'execution.step':'One FSDP forward owns all trainable components and objective computation.',
'execution.sampled':'Sample, verify and replay current-policy anchor blocks.',
'execution.supervision':'Select and align frozen supervision rows for chunked objectives.',
'execution.windows':'Prepare global optimizer-window denominators and manage batch lifetime.',
'optim.optimizer':'FP32-master parameter updates, AdamW/Muon routing and gradient normalization.',
'optim.scheduler':'Learning-rate schedules used by the shared optimizer.',
'optim.muon':'FP32-sharded Muon with shape-batched all-to-all orthogonalization.',
'optim.fsdp_layout':'Isolate private FSDP parameter-layout access from optimizer mathematics.',
'logger.metrics':'Additive token accuracy and speculative acceptance diagnostics.',
'logger.logging':'Aggregate optimizer-window metrics and send logs to experiment trackers.',
'target.features':'Build frozen target components and obtain cached or online supervision features.',
'target.streaming':'Produce target features and stream training packets to draft ranks.',
'config':'Parse and validate the public training CLI.',
'outputs':'Additive training outputs; numerators retain their gradient graph.',
'__init__':'Shared LEGOSpec training entry.'}

def default_owner(old):return P+defmod[old[len(P):]] if old.startswith(P) and old[len(P):] in defmod else old
def rewrite(s,current):
    tree=ast.parse(s);lines=s.splitlines(True);offset=[0]
    for l in lines:offset.append(offset[-1]+len(l))
    edits=[]; extra={}
    def span(n):return offset[n.lineno-1]+len(lines[n.lineno-1].encode()[:n.col_offset].decode()),offset[n.end_lineno-1]+len(lines[n.end_lineno-1].encode()[:n.end_col_offset].decode())
    def change(n,value):a,b=span(n);edits.append((a,b,value))
    def module_ref(owner):
        alias='_'.join(owner.removeprefix(P).split('.'))
        extra[alias]=owner
        return alias
    def owner_attr(old,attr):return exports.get((old,attr),default_owner(old))
    def scan_aliases(node,inherited):
        aliases=dict(inherited)
        def find(n):
            if n is not node and isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef,ast.Lambda)):return
            if isinstance(n,ast.ImportFrom) and n.module=='legospec.training':
                for a in n.names:
                    if P+a.name in sources:aliases[a.asname or a.name]=P+a.name
            if isinstance(n,ast.Import):
                for a in n.names:
                    if a.name in sources and a.asname:aliases[a.asname]=a.name
            for child in ast.iter_child_nodes(n):find(child)
        find(node);return aliases
    def walk(n,aliases):
        if isinstance(n,(ast.Module,ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
            aliases=scan_aliases(n,aliases)
        if isinstance(n,ast.ImportFrom):
            if n.module in sources:
                grouped=defaultdict(list)
                for a in n.names:grouped[owner_attr(n.module,a.name)].append(a)
                text=[]
                for owner,names in grouped.items():
                    if owner!=current:text.append(ast.unparse(ast.ImportFrom(module=owner,names=names,level=0)))
                change(n,('\n'+' '*n.col_offset).join(text) or '# relocated import')
                return
            if n.module=='legospec.training':
                text=[]
                for a in n.names:
                    old=P+a.name
                    if old in sources:
                        new=default_owner(old);parent,name=new.rsplit('.',1)
                        text.append(f'from {parent} import {name} as {a.asname or a.name}')
                    else:text.append(ast.unparse(ast.ImportFrom(module=n.module,names=[a],level=0)))
                change(n,('\n'+' '*n.col_offset).join(text));return
        if isinstance(n,ast.Import):
            changed=False;names=[]
            for a in n.names:
                new=default_owner(a.name);changed|=new!=a.name;names.append(ast.alias(name=new,asname=a.asname))
            if changed:change(n,ast.unparse(ast.Import(names=names)))
            return
        if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id in aliases:
            owner=owner_attr(aliases[n.value.id],n.attr)
            if owner==current:change(n,n.attr)
            else:change(n,module_ref(owner)+'.'+n.attr)
            return
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in {'getattr','hasattr','setattr'} and len(n.args)>=2 and isinstance(n.args[0],ast.Name) and n.args[0].id in aliases and isinstance(n.args[1],ast.Constant) and isinstance(n.args[1].value,str):
            owner=owner_attr(aliases[n.args[0].id],n.args[1].value)
            change(n.args[0],module_ref(owner))
            for a in n.args[1:]:walk(a,aliases)
            for a in n.keywords:walk(a,aliases)
            return
        for child in ast.iter_child_nodes(n):walk(child,aliases)
    walk(tree,{})
    for a,b,v in sorted(edits,reverse=True):s=s[:a]+v+s[b:]
    # Also update literal monkeypatch targets and subprocess snippets.
    for (old,name),new in sorted(exports.items(),key=lambda x:len(x[0][0]+x[0][1]),reverse=True):
        s=s.replace(old+'.'+name,new+'.'+name)
    for old in sorted(sources,key=len,reverse=True):
        s=re.sub(re.escape(old)+r'(?=[\s\"\x27,:;)]|$)',default_owner(old),s)
    # Newly introduced imports are placed after future imports.
    if extra:
        t=ast.parse(s);insertion=0
        for n in t.body:
            if (isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str)) or (isinstance(n,ast.ImportFrom) and n.module=='__future__'):insertion=n.end_lineno
            else:break
        lines=s.splitlines(True)
        lines[insertion:insertion]=[f'import {owner} as {alias}\n' for alias,owner in sorted(extra.items())]
        s=''.join(lines)
    return s

for dest,parts in blocks.items():
    short=dest[len(P):]
    imps=list(dict.fromkeys(import_candidates[dest]))
    s='"""'+docs[short]+'"""\n\nfrom __future__ import annotations\n\n'+'\n'.join(imps)+'\n\n'+'\n\n'.join(body for _,body in parts)
    # Resolve symbol imports and module attributes after assembling merged modules.
    s=rewrite(s,dest)
    # Removing self imports leaves a `pass`; delete these no-op import placeholders.
    s=re.sub(r'^ *# relocated import\n','',s,flags=re.M)
    ast.parse(s)
    path=R/Path(*dest.split('.')).with_suffix('.py');path.parent.mkdir(parents=True,exist_ok=True);path.write_text(s)
# Entry module is stored as training/__init__.py, as in the original tree.
for old in sources:
    p=R/Path(*old.split('.')).with_suffix('.py')
    if old not in blocks and p.exists():p.unlink()
# Update active consumers only. Frozen snapshots and published/checkpoint exports are untouched.
consumers=[*list((R/'legospec').rglob('*.py')),*list((R/'scripts').rglob('*.py')),*list((R/'tests/unit').glob('*.py'))]
for p in consumers:
    if p.is_relative_to(R/'legospec/training'):continue
    original=W/'consumer_originals'/p.relative_to(R)
    if not original.exists():original.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,original)
    s=original.read_text()
    if 'legospec.training' not in s:continue
    new=rewrite(s,'.'.join(p.relative_to(R).with_suffix('').parts))
    ast.parse(new);p.write_text(new)
(W/'symbol_owners.json').write_text(json.dumps({old+'.'+name:new+'.'+name for (old,name),new in exports.items()},indent=2)+'\n')
print('Relocated',len(blocks),'modules; active consumer imports rewritten.')
