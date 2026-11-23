from pathlib import Path
root=Path.cwd();target=root/'legospec/modeling/target'
p=target/'adapters/qwen35.py';s=p.read_text();s=s.replace('class Qwen35TargetAdapter(TargetModelAdapter):','''class Qwen35TargetAdapter(TargetModelAdapter):
    def __init__(self, config=None):
        super().__init__(config)
        self._capture_restores = []
        self._patched_model = None

    def remove_capture(self):
        super().remove_capture()
        for owner, name, original in reversed(self._capture_restores):
            setattr(owner, name, original)
        self._capture_restores.clear()
        if self._patched_model is not None:
            del self._patched_model._training_target_communicator_capture
            self._patched_model = None
''');s=s.replace('    @staticmethod\n    def _set_communicator_capture(\n        model:', '    def _set_communicator_capture(\n        self, model:');s=s.replace('            model._training_target_communicator_capture = state','            model._training_target_communicator_capture = state\n            self._patched_model = model');s=s.replace('                communicator.prepare_attn = prepare_attn_and_capture','                self._capture_restores.append((communicator, "prepare_attn", original_prepare_attn))\n                communicator.prepare_attn = prepare_attn_and_capture');s=s.replace('                processor.aux_hidden_states_provider = take_hidden_states','                self._capture_restores.append((processor, "aux_hidden_states_provider", processor.aux_hidden_states_provider))\n                processor.aux_hidden_states_provider = take_hidden_states');p.write_text(s)
# Native capture forward lifecycle is backend code, not a model subclass.
p=target/'sglang.py';s=p.read_text().replace('        # Default to all layers until the caller selects the required subset.\n','').replace('Gemma prefill expects matching','Native prefill expects matching').replace('Gemma 4 native teacher currently requires collocated training','Native SGLang capture currently requires collocated training').replace('Gemma 4 native teacher requires the trainer contiguous TP mesh','Native SGLang capture requires the trainer contiguous TP mesh');p.write_text(s)
# Explain current ownership and the fixed feature contract.
(target/'DESIGN.md').write_text('''# Target 模型边界

`training_target_model.py` 定义输出、backend 接口和构造入口；不依赖 training。
`hf.py` / `sglang.py` 负责 forward、设备、TP、批次还原、共享组件和 anchor scorer 生命周期。
`adapters/base.py` 提供默认模型结构与 capture 语义；`gemma.py` 保留原生 embedding 缩放、softcap、native capture 初始化要求；`qwen35.py` 管固定环境的 Qwen3.5/3.6 communicator capture patch。
只有模型确有差异才增加 adapter；不复制完整 backend。默认 adapter 被普通 HF/SGLang 和离线组件加载复用，Gemma/Qwen adapter 被统一构造入口选择。

`components.py` 负责共享 embedding/head、词表 TP 通信与独立权重加载。模型 adapter 定位组件、提供数值语义；是否分片与如何通信由组件实现负责。
Gemma native packed capture 算法继续在 `sglang_gemma4_capture.py`，离线采集和 adapter 共用；没有独立 Gemma training target 类。
anchor scorer 的独立数学保留在原文件，训练端只调用 `build_anchor_scorer()`。

所有训练输出 `hidden_states` 是 0-based decoder 层编号到 tensor 的字典。最后一层 context 是 final norm 前 residual；`final_hidden_state` 单独保存 final norm 后 teacher 特征，不能互相替代。
HF/SGLang backend 负责从原生输出转换；缓存 reader 返回相同含义。训练端不猜 tuple offset、不兜底填补缺失 teacher。
packed capture 保留 token offsets 与 bulk aux/last 传输，训练按 batch-first 输出，不以 padding 扩大离线采集的显存/访存。

capture 层在构造时传入。SGLang adapter hooks 仅安装在当前模型实例，重设时移除；Qwen communicator patch 显式保存并恢复原方法，创建 anchor scorer 前释放普通 capture hooks。
训练需求与资源预算在 `training/target/setup.py`，选择/拼接在 `features.py`，producer/packet 调度在 `streaming.py`。监督 logits 的按需投影仍归 objective/supervision。

不提供旧路径转发、版本号或旧格式兼容。回归记录：`tests/target_boundaries_20261122/`。
''')
p=root/'legospec/training/DESIGN.md';s=p.read_text().replace('- `target/`：features 管冻结 target、共享 embedding/head 和监督特征；streaming 管分离部署的生产循环及训练包。底层 target 实现仍在 modeling/target，传输实现在 distributed。','- `target/`：setup 管训练需要的 capture 层、容量预算、在线/缓存选择、tokenizer/mask 和组件装配；features 只管指定层的选择/拼接及设备搬运；streaming 管分离部署的生产循环及训练包。模型语义、backend 差异及共享组件实现归 modeling/target，传输实现在 distributed。');p.write_text(s)
p=root/'docs/COMPONENT_ORGANIZATION.md';s=p.read_text()+'\nTarget 的 backend 与模型 adapter 分离：HF/SGLang 管执行与资源；默认/Gemma/Qwen3.5 adapter 管模型语义与 capture 差异。训练 target/setup 只提出训练需求，features 只组装特征，streaming 只调度生产/传输。最后一层 residual 和 normalized teacher 输出分开，详见 modeling/target/DESIGN.md。\n';p.write_text(s)
p=root/'docs/SIMPLIFICATION_PLAN.md';p.write_text(p.read_text()+'\n2026-11-22 target 边界重组：拆出 training/target/setup；HF/SGLang 复用模型 adapter，移除 Gemma 独立 training backend；统一层字典与显式 final-norm teacher，移除训练端模型内部访问和重复 cached logits 路径。\n')
