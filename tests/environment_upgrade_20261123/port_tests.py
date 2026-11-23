from pathlib import Path
p=Path("tests/unit/test_native_capture.py");s=p.read_text()
s=s.replace("self.model = nn.Linear(4, 4)","self.model = nn.Linear(4, 4)\n            self.model.logits_processor = Processor()\n            self.model.set_eagle3_layers_to_capture = lambda ids: setattr(self.model, \"capture_ids\", ids)")
s=s.replace("self.server_args.return_aux_hidden_state_layer_ids","self.model.capture_ids")
s=s.replace("return SimpleNamespace(\n                logits_output=", "self.model.logits_processor(None, values + 100)\n            return SimpleNamespace(\n                logits_output=")
s=s.replace("    class Runner:","    class Processor(nn.Module):\n        def forward(self, ids, hidden):\n            return hidden\n\n    class Runner:")
s=s.replace("hidden_size=4,\n        torch_dtype=", "hidden_size=4,\n        adapter=GemmaTargetAdapter(),\n        torch_dtype=")
s=s.replace("args.return_aux_hidden_state_layer_ids == [1, 3]","engine.runner.model.capture_ids == [1, 3]").replace("args.enable_return_aux_hidden_states","args.enable_return_hidden_states")
s=s.replace("                    last_hidden_states=values + 100,\n","")
p.write_text(s)
p=Path("tests/unit/test_sglang_gemma4_training.py");s=p.read_text().replace("    def __init__(self):\n        class NativeScaledEmbedding","    def __init__(self):\n        self.layer_ids = [1, 3]\n        class NativeScaledEmbedding")
a=s.index("def test_capture_policy_cannot_change");b=s.index("\ndef test_tp2_",a)
s=s[:a]+'''def test_capture_policy_updates_native_model_and_engine():
    t = target()
    with patch.object(t.adapter, "install_capture") as install:
        t.set_capture_layers([0])
    install.assert_called_once_with(t.model_runner.model, [0])
    assert t.engine.layer_ids == t.capture_layer_ids == [0]

'''+s[b:];p.write_text(s)
p=Path("tests/unit/test_sglang_training_prefill.py");s=p.read_text();a=s.index("    backend =");b=s.index("\n    kept =",a)
s=s[:a]+'''    def capture(records):
        tokens = [record["input_ids"].tolist() for record in records]
        assert tokens == [[2, 3], [4, 0, 5], [1, 4, 1, 2]]
        causal = torch.cat([torch.tensor(row).float().cumsum(0)[:, None] for row in tokens])
        return SimpleNamespace(hidden_states={0:causal, 2:causal+10},
            final_hidden_state=causal+20, token_offsets=[0,2,5,9])
    engine = SimpleNamespace(layer_ids=[0,2], generate_capture_data=capture)
    backend = SGLangTrainingTargetModel(SimpleNamespace(), adapter=None, engine=engine)
    head = lambda hidden: torch.cat((hidden, -hidden), dim=-1)
    with patch.object(backend, "build_shared_components", return_value=SimpleNamespace(lm_head=head)):
        output = backend.generate_training_data(ids, attention, loss, return_logits)
'''+s[b:]
s=s.replace("logits = torch.cat((expected, -expected), dim=-1)","logits = torch.cat((expected + 20, -expected - 20), dim=-1)")
p.write_text(s)
p=Path("tests/unit/test_target_adapters.py");s=p.read_text()
s=s.replace("        self.model = nn.Module()","        self.model = CaptureBackbone()")
a=s.index("    def forward(self, hidden, residual):\n        first =");b=s.index("\ndef test_sglang_capture",a)
s=s[:a]+'''    def forward(self, hidden, residual):
        self.model.ids = self.ids
        normalized, aux = self.model(hidden, residual)
        return self.logits_processor(None, normalized, None, None, aux_hidden_states=aux)


class CaptureBackbone(nn.Module):
    def forward(self, hidden, residual):
        first = hidden.clone()
        normalized, _ = self.norm(hidden, residual)
        aux = [first] if 0 in self.ids else []
        return (normalized, aux) if aux else normalized

'''+s[b:]
a=s.index("def test_qwen_communicator");b=s.index("\ndef test_adapter_selection",a)
s=s[:a]+'''def test_qwen35_uses_native_capture_selector():
    model = nn.Module()
    model.set_dflash_layers_to_capture = lambda ids: setattr(model, "capture_ids", ids)
    adapter = Qwen35TargetAdapter()
    adapter.install_capture(model, [0, 1])
    assert model.capture_ids == [0, 1]
    assert adapter._capture_handles == []

'''+s[b:]
s=s.replace('    shim = {\n        "legospec.modeling.target.sglang_backend.utils": SimpleNamespace(\n            LogitsProcessorForEAGLE3=Processor\n        )\n    }\n','')
s=s.replace('    with patch.dict("sys.modules", shim):\n','    if True:\n')
p.write_text(s)
p=Path("legospec/modeling/target/sglang_anchor_opd.py");s=p.read_text()
a=s.index("        self._capture_ids, self._captured");b=s.index("\n    def _clear_pools",a)
s=s[:a]+'''        self._captured = {}
'''+s[b:]
s=s.replace("        self._capture_ids.clear()\n","")
s=s.replace("        req.fill_ids = ids","        req.init_next_round_input()\n        req.extend_end = len(ids)").replace("        req.extend_input_len = len(ids) - len(req.prefix_indices)\n","")
s=s.replace("        from legospec.modeling.target.sglang_backend.utils import (\n            LogitsProcessorForEAGLE3,\n        )","        from sglang.srt.managers.overlap_utils import resolve_forward_inputs")
s=s.replace("required = sum(req.extend_input_len for req in reqs)","required = sum(req.extend_end - len(req.prefix_indices) for req in reqs)")
a=s.index("        for module in self.model.modules():");b=s.index("        tree = RadixCache",a)
s=s[:a]+'''        layer_ids = sorted(capture_layer_ids)
        self.target.adapter.remove_capture()
        if layer_ids:
            self.target.adapter.install_capture(self.model, layer_ids)
        else:
            self.model.capture_aux_hidden_states = False
            self.model.model.layers_to_capture = []
'''+s[b:]
a=s.index("        forward = ForwardBatch.init_new");b=s.index("\n    @torch.no_grad()\n    def prefill_anchors",a)
s=s[:a]+'''        resolve_forward_inputs(batch, self.target.engine.future_map)
        forward = ForwardBatch.init_new(
            batch, self.runner, capture_hidden_mode=CaptureHiddenMode.FULL,
            return_hidden_states_before_norm=False,
        )
        teacher = {}
        def capture_teacher(module, args):
            teacher["hidden"] = args[1]
        handle = self.model.logits_processor.register_forward_pre_hook(capture_teacher)
        try:
            output = self.runner.forward(forward).logits_output
            hidden = teacher["hidden"]
            if hidden.shape[0] != required:
                raise RuntimeError("SGLang must return every EXTEND token hidden state")
            captured = {}
            if layer_ids:
                aux = output.hidden_states.reshape(required, len(layer_ids), hidden.shape[-1])
                captured = {layer:aux[:, i] for i, layer in enumerate(layer_ids)}
            return hidden, captured, batch
        finally:
            handle.remove()
            self.target.adapter.remove_capture()
            self.model.capture_aux_hidden_states = False
            self.model.model.layers_to_capture = []
'''+s[b:];p.write_text(s)
p=Path("pyproject.toml");s=p.read_text().replace('sys_platform == linux',"sys_platform == 'linux'").replace('preview = true','preview = true\nprerelease = "allow"');p.write_text(s)
