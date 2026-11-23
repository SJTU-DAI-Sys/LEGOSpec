from pathlib import Path
import ast, shutil, json, hashlib, textwrap
R=Path(__file__).resolve().parents[2]
D=R/'development/objectives'
T=R/'tests/development/objectives'
A=R/'tests/deferred_objectives_20261122'
C=Path('legospec/modeling/draft/composable')
manifest=[]
def archive(rel, dest):
    src=R/rel; dst=dest/rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src,dst)
    manifest.append({'source':str(rel),'archive':str(dst.relative_to(R)), 'sha256':hashlib.sha256(src.read_bytes()).hexdigest()})
prod=[C/'loss_config.py',C/'schema.py',C/'objective/__init__.py',C/'objective/dpal.py',C/'objective/survival.py',Path('legospec/training/anchor_opd.py'),Path('legospec/training/setup.py')]
for p in prod: archive(p,D/'snapshots')
tests=['test_dpal_plus_plus.py','test_dpal_renyi.py','test_vocab_reuse.py','test_composable_eagle3.py','test_anchor_opd.py']
for name in tests: archive(Path('tests/unit')/name,T/'snapshots')
# Keep implementation source in its original relative path for future restoration.
(R/C/'objective/survival.py').unlink()
(R/'tests/unit/test_dpal_plus_plus.py').unlink()
for p in list((R/'configs').glob('*.json'))+list((R/'scripts/config').glob('*.yaml')):
    if any(k in p.read_text() for k in ('dpal_plus_plus','opd_survival','tv_survival')):
        rel=p.relative_to(R); archive(rel,D); p.unlink()

def edit(rel, fn):
    p=R/rel; old=p.read_text(); new=fn(old)
    assert new != old, rel
    ast.parse(new) if p.suffix=='.py' else None
    p.write_text(new)
def replace(s,a,b):
    assert a in s, a
    return s.replace(a,b)
def remove_nodes(s,predicate,unwrap_else=False):
    lines=s.splitlines(keepends=True); edits=[]
    for n in ast.walk(ast.parse(s)):
        if predicate(n):
            start=min([n.lineno]+[d.lineno for d in getattr(n,'decorator_list',[])])-1
            value=''
            if unwrap_else and getattr(n,'orelse',[]):
                value=''.join(l[4:] if l.strip() else l for l in lines[n.orelse[0].lineno-1:n.orelse[-1].end_lineno])
            edits.append((start,n.end_lineno,value))
    for start,end,value in sorted(edits,reverse=True): lines[start:end]=[value]
    return ''.join(lines)
def string_in(n,words):
    return any(isinstance(x,ast.Constant) and x.value in words for x in ast.walk(n))

def loss(s):
    s=s.replace('("dpal_renyi", "dpal_plus_plus")','("dpal_renyi",)').replace('{"dpace", "tv_survival", "opd_survival", "e2e_tv"}','{"dpace", "e2e_tv"}').replace('("opd_survival", "e2e_tv")','("e2e_tv",)')
    return remove_nodes(s,lambda n:isinstance(n,ast.If) and string_in(n.test,('dpal_plus_plus','opd_survival')))
edit(C/'loss_config.py',loss)
def schema(s):
    a=s.index('    "dpal_plus_plus":'); b=s.index('    "selector_losses":',a)
    s=s[:a]+s[b:]
    return remove_nodes(s,lambda n:isinstance(n,ast.If) and string_in(n.test,('tv_survival','opd_survival')))
edit(C/'schema.py',schema)
edit(C/'objective/__init__.py',lambda s:''.join(l for l in s.splitlines(keepends=True) if not any(k in l for k in ('objective.survival','"tv_survival"','"opd_survival"','"dpal_plus_plus"'))))
def dpal(s):
    s=replace(s,', *, divide_own_overlap=False','')
    s=replace(s,'return credit / smooth if divide_own_overlap else credit','return credit')
    s=replace(s,'        self.divide_own_overlap = self.recipe["type"] == "dpal_plus_plus"\n','')
    return replace(s,'credit = dpal_credit(\n            overlap, valid, self.rho, divide_own_overlap=self.divide_own_overlap\n        )','credit = dpal_credit(overlap, valid, self.rho)')
edit(C/'objective/dpal.py',dpal)
def setup(s):
    s=replace(s,'objective_types & {"dpal_renyi", "dpal_plus_plus"}','"dpal_renyi" in objective_types')
    s=replace(s,'dpal_credit="detached_overlap_prefix_suffix_divided_by_own_overlap"\n            if "dpal_plus_plus" in objective_types\n            else "detached_overlap_prefix_suffix",','dpal_credit="detached_overlap_prefix_suffix",')
    return replace(s,'opd_survival or e2e_tv','e2e_tv')
edit(Path('legospec/training/setup.py'),setup)
def anchor(s):
    s=''.join(l for l in s.splitlines(keepends=True) if not any(k in l for k in ('import combine_component_terms','import distribution_terms','_compiled_statistics = None','component_loss = getattr(')))
    s=remove_nodes(s,lambda n:isinstance(n,ast.FunctionDef) and n.name=='full_batch_statistics')
    s=remove_nodes(s,lambda n:isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='component_loss',True)
    # Metadata collection only existed for the deferred component-normalized objective.
    tree=ast.parse(s); n=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='metadata_only')
    lines=s.splitlines(keepends=True)
    lines[n.lineno-1:n.end_lineno]=['    if metadata_only:\n        raise ValueError("E2E-TV does not require a sampled OPD metadata pass")\n']
    s=''.join(lines)
    s=remove_nodes(s,lambda n:isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='statistics' for t in n.targets))
    s=s.replace('                opd_accepted_lengths=record.accepted_lengths,\n','').replace('                opd_distribution_terms=statistics,\n','')
    s=replace(s,'native = (\n            terms.component_terms\n            if component_loss\n            else {"loss": terms.loss_terms, **terms.ratio_metrics}\n        )','native = {"loss": terms.loss_terms, **terms.ratio_metrics}')
    return replace(s,'opd_survival or e2e_tv','e2e_tv')
edit(Path('legospec/training/anchor_opd.py'),anchor)
# Remove deferred cases from mixed suites; full original suites remain archived.
def mixed(s):
    for k in ('tv_survival','opd_survival','dpal_plus_plus'):
        s=s.replace('"'+k+'", ','').replace(', "'+k+'"','')
        s=''.join(l for l in s.splitlines(keepends=True) if l.strip()!='"'+k+'",')
    return s
edit(Path('tests/unit/test_vocab_reuse.py'),mixed)
def renyi(s):
    s=remove_nodes(s,lambda n:isinstance(n,ast.FunctionDef) and n.name=='test_dpal_with_opd_sibling_keeps_component_and_metric_denominators')
    s=s.replace('train.component_normalization = kind == "opd_survival"','train.component_normalization = False')
    return mixed(s)
edit(Path('tests/unit/test_dpal_renyi.py'),renyi)
def eagle(s):
    s=remove_nodes(s,lambda n:isinstance(n,ast.If) and string_in(n.test,('opd_survival',)),True)
    s=s.replace('    metadata = []\n','').replace('            # Supply explicit component counts for the OPD objective.\n','')
    s=s.replace('for kind in ("soft_ce", "opd_survival"):','for kind in ("soft_ce",):')
    return mixed(s)
edit(Path('tests/unit/test_composable_eagle3.py'),eagle)
def anchor_tests(s):
    deferred={'test_parallel_rnn_sampling_replay_and_freezing','test_cached_context_and_online_context_match','test_accumulation_collects_target_once_and_spills','test_generator_resume_determinism','test_distribution_statistics_match_raw_loss_and_gradients'}
    s=remove_nodes(s,lambda n:isinstance(n,ast.FunctionDef) and n.name in deferred)
    s=s.replace('from legospec.training.windows import prepare_anchor_opd_window\n','')
    return replace(s,'"type": "opd_survival"','"type": "e2e_tv"')
edit(Path('tests/unit/test_anchor_opd.py'),anchor_tests)
# Exclude archived tests from recursive test discovery, even for `pytest tests`.
edit(Path('pyproject.toml'),lambda s:replace(s,'norecursedirs = ["archive",','norecursedirs = ["development", "archive",'))
edit(C/'objective/DESIGN.md',lambda s:s.replace('survival 保留各自公式；','').replace('`survival.py`, ','')+'\n待开发的 dpal_plus_plus、tv_survival、opd_survival 位于仓库根目录 development/objectives，未注册到正常训练入口。\n')
edit(Path('docs/COMPONENT_ORGANIZATION.md'),lambda s:replace(s,'token、block、survival、composition、DPAL','token、block、composition、DPAL')+'\n暂缓使用的实验目标及其恢复快照保存在 development/objectives/，不属于 legospec 安装包。\n')
D.mkdir(parents=True,exist_ok=True)
(D/'README.md').write_text('''# 待开发的训练目标

2026-11-22：暂存 `dpal_plus_plus`、`tv_survival`、`opd_survival`。
这些目标未注册到当前 schema / loss registry，现有训练入口拒绝选择它们。
此目录在 `legospec` 包外，不随 Python 包安装。

- D-PAL++：在 D-PAL 动态梯度 credit 上除去当前位置自身的平滑 overlap；rho 控制平滑，local_loss 选择局部损失，reduction 选择归一化。
- TV-survival：逐位置 TV 加连续前缀存活辅助项，survival_weight 控制后者。
- OPD-survival：接受位置 forward KL、拒绝位置 reverse KL，加前缀存活项；beta_rej、rejection_decay 控制拒绝项，survival_weight 控制存活项。离线接受长度由固定标签上的贪心匹配给出；采样模式使用 p/q 验证。

## 保留内容

- `snapshots/legospec/modeling/draft/composable/objective/survival.py`：TV/OPD-survival 原始实现。
- `snapshots/legospec/modeling/draft/composable/objective/dpal.py`：含 D-PAL++ 分支的原始实现，同时含共享的普通 D-PAL。
- `snapshots/` 其他文件：移出前的 schema、注册、配置归一化及训练集成参考。
- `configs/`、`scripts/config/`：原始 JSON 配方和 launcher。路径、权重引用按原样保留，其中旧 TV launcher 引用的 JSON 在移出前已不存在。这些是开发资料，不是当前可运行入口。
- 对应测试快照位于仓库 `tests/development/objectives/`。

## 恢复开发

快照不是独立可运行插件。恢复时应将相关公式与注册、schema、训练分支合并到届时的代码，并恢复专属测试；不要整文件覆盖持续演进的共享模块。
普通 `dpal_renyi`、`dpace`、`e2e_tv` 仍在主代码中。采样训练共用的 E2E-TV 路径保留。
历史 checkpoint、模型发布目录、训练源码快照未迁移。旧目标的完整 resume 仍应使用原始训练环境、配置及 rank topology。
''')
(T/'README.md').write_text('''# 待开发目标测试快照

保存移出前的完整测试文件，包含混合测试中的活动目标用例作为上下文。
这些快照依赖已暂存的注册和训练分支，当前不运行，pytest 默认递归排除 development。
主测试目录保留现行目标及共享采样功能的回归；恢复步骤见 ../../../development/objectives/README.md。
''')
(A/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Archived',len(manifest),'files; active sources parsed successfully.')
