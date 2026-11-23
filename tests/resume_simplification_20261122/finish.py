from pathlib import Path
R=Path(__file__).resolve().parents[2]
p=R/'legospec/training/optim/fsdp_layout.py';s=p.read_text();a=s.index('        state = asdict(self)');b=s.index('\n\n\ndef local_layouts',a);s=s[:a]+'        return asdict(self)\n'+s[b:];p.write_text(s)
p=R/'legospec/training/optim/optimizer.py';s=p.read_text().replace('            # Legacy lr/weight_decay are deliberately irrelevant in hybrid mode.\n','');p.write_text(s)
p=R/'legospec/training/data/planning.py';s=p.read_text().replace('if state.get("pending") is not None:', 'if state["pending"] is not None:');p.write_text(s)
p=R/'tests/unit/test_checkpoint_resume.py';s=p.read_text().replace('import torch\n','import torch\nimport pytest\n');s=s.replace('def test_full_optimizer_rng_and_cursor_resume(tmp_path):','@pytest.mark.parametrize("muon_config", [None, {}], ids=["adamw", "muon_adamw"])\ndef test_full_optimizer_rng_and_cursor_resume(tmp_path, muon_config):');s=s.replace('total_steps=10, warmup_ratio=0.2)', 'total_steps=10, warmup_ratio=0.2, muon_config=muon_config)').replace('resumed_model, lr=0.01, total_steps=10, warmup_ratio=0.2\n','resumed_model, lr=0.01, total_steps=10, warmup_ratio=0.2, muon_config=muon_config\n');a=s.index('    for key, values in expected["optimizer_state_dict"]["state"].items():');s=s[:a]+'''    expected_opts = expected["optimizer_state_dict"]
    actual_opts = actual["optimizer_state_dict"]
    pairs = (
        [(actual_opts[name], expected_opts[name]) for name in ("adamw", "muon")]
        if muon_config is not None else [(actual_opts, expected_opts)]
    )
    for actual_opt, expected_opt in pairs:
        assert actual_opt["param_groups"] == expected_opt["param_groups"]
        for key, values in expected_opt["state"].items():
            for name, value in values.items():
                torch.testing.assert_close(
                    actual_opt["state"][key][name], value, rtol=0, atol=0,
                )


def test_each_rank_restores_its_own_tensors_and_rng(tmp_path):
    for rank in (0, 1):
        torch.manual_seed(42 + rank)
        state = {
            "cursor": asdict(RunState(global_step=7, optimizer_step=1)),
            "optimizer_state_dict": {"moment": torch.randn(3)},
            "scheduler_state_dict": {"last_epoch": 1},
            "fp32_master_params": [torch.randn(3)],
            "rng_state": capture_rng_state(),
        }
        save_distributed_training_state(str(tmp_path), state, rank=rank)
        loaded = load_distributed_training_state(str(tmp_path), rank=rank)
        assert loaded["cursor"] == state["cursor"]
        assert loaded["scheduler_state_dict"] == state["scheduler_state_dict"]
        torch.testing.assert_close(loaded["optimizer_state_dict"], state["optimizer_state_dict"], rtol=0, atol=0)
        torch.testing.assert_close(loaded["fp32_master_params"], state["fp32_master_params"], rtol=0, atol=0)
        expected = torch.rand(4)
        restore_rng_state(loaded["rng_state"])
        torch.testing.assert_close(torch.rand(4), expected, rtol=0, atol=0)
    common = torch.load(tmp_path / "training_state.pt", weights_only=False)
    assert not {"optimizer_state_dict", "fp32_master_params", "rng_state"} & common.keys()
''';p.write_text(s)
# Persist the user's explicit default for later development, without a compatibility mechanism.
p=R/'AGENTS.md';p.write_text(p.read_text()+'''\n## 开发迭代与训练恢复（用户明确要求，2026-11-22）

开发更新默认只维护当前实现。除非用户明确要求兼容旧版，不添加旧版本分支、适配层、版本标记或兼容性预检查，也不为保留历史行为而保留代码。
同一次训练恢复按原配置和 rank topology 直接加载实际状态；不比较版本号、配置身份、规划签名或源码指纹。
保留 optimizer、scheduler、FP32 master、每 rank RNG、训练游标和 pending plan 的完整保存/恢复。无效输入由实际读取或执行操作正常报错。
''')
p=R/'docs/VERSION_CLEANUP.md';p.write_text('''# 开发迭代与恢复约定

2026-11-22 用户最新要求：只维护当前实现，除非明确要求，不考虑旧版兼容。

已删除训练恢复的 PLAN_VERSION、ROUND_ORDER_VERSION、ANCHOR_SAMPLING_VERSION、LOSS_CONTRACT_VERSION，以及 checkpoint/optimizer 格式版本和执行契约标签。
已删除配置身份、数据指纹、rank topology、参数签名、loss contract、scheduler 类型和 pending plan 签名的恢复比较。不新增旧格式适配或自动迁移。

恢复直接加载当前 rank 的状态文件、optimizer/scheduler、FP32 master、RNG、cursor 和 pending plan。checkpoint 仍先写各 rank 状态，再发布公共元数据；模型权重和训练 YAML 继续保存。
shuffle 与 anchor 的种子只依赖实际 seed、epoch、phase、window 和样本信息，不再包含版本前缀。不承诺跨代码更新复现以前的采样序列。

training_mode=legacy|optimized 仍是已明确保留的执行模式。模型结构选择、外部依赖版本要求、缓存存储编码不属于这次训练恢复清理范围。

本次只修改当前源码、测试和说明；不迁移已有 checkpoint、cache、发布权重或历史运行快照。
验证记录：tests/resume_simplification_20261122/。
''')
for rel in ['docs/COMPONENT_ORGANIZATION.md','legospec/training/DESIGN.md','legospec/training/runtime/DESIGN.md']:
 p=R/rel;s=p.read_text().replace('checkpoint 自己构造恢复身份，不反调 setup','checkpoint 直接读写恢复状态，不反调 setup').replace('checkpoint 管导出、完整保存/恢复及身份校验','checkpoint 管导出与完整保存/恢复').replace('checkpoint 身份构造归 checkpoint；不反调 setup','checkpoint 直接加载实际状态；不反调 setup').replace('checkpoint 独立拥有保存、恢复和训练身份','checkpoint 独立拥有保存和恢复')
 s+='\n当前开发不维护旧版兼容。恢复直接加载完整状态，不保存或比较版本号、配置身份、规划签名。\n';p.write_text(s)
p=R/'docs/SIMPLIFICATION_PLAN.md';p.write_text(p.read_text()+'\n2026-11-22 按用户要求删除训练恢复版本标签与身份比较，只保留当前实现的直接状态保存/恢复；不提供旧格式兼容。\n')
print('Updated current-format restore tests and recorded the development rule.')
