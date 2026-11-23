from pathlib import Path
root=Path.cwd();unit=root/'tests/unit'
p=unit/'test_target_feature_contract.py';s=p.read_text().replace('import legospec.training.target.features as target_features','import legospec.training.target.features as target_features\nimport legospec.training.target.setup as target_setup');s=s.replace('target_features.AutoTokenizer','target_setup.AutoTokenizer').replace('target_features.resolve_tokenizer_and_components','target_setup.resolve_tokenizer_and_components').replace('                target_features,\n                "load_text_model_config"','                target_setup,\n                "load_text_model_config"')
a=s.index('    def test_context_selection_for_layer_only');b=s.index('    def test_target_packet',a)
s=s[:a]+'''    def test_context_selection_uses_absolute_layer_ids_and_draft_order(self):
        states = {i: torch.full((1, 2, 4), i, dtype=torch.float32) for i in [2, 7, 11]}
        got = target_features.concatenate_target_hidden(states, [11, 2])
        torch.testing.assert_close(got, torch.cat([states[11], states[2]], -1))

'''+s[b:]
a=s.index('    def test_cached_teacher_feature_and_tensor_cuda_helper');b=s.index('\n\nif __name__',a)
s=s[:a]+'''    def test_cached_teacher_feature_is_not_inferred_from_context(self):
        raw, teacher = torch.randn(2, 7, 16), torch.randn(2, 7, 16)
        draft = SimpleNamespace(target_layer_ids=[2])
        data = {"hidden_states": {2: raw}, "final_hidden_state": teacher}
        with patch.object(torch.Tensor, "cuda", lambda tensor, **kwargs: tensor):
            context, got = target_features.prepare_target_features(data, None, draft, True)
            torch.testing.assert_close(context, raw)
            self.assertIs(got, teacher)
            del data["final_hidden_state"]
            with self.assertRaisesRegex(ValueError, "post-final-norm"):
                target_features.prepare_target_features(data, None, draft, True)
            _, absent = target_features.prepare_target_features(data, None, draft, False)
            self.assertIsNone(absent)
'''+s[b:];p.write_text(s)
p=unit/'test_sglang_gemma4_training.py';s=p.read_text().replace('import legospec.training.target.features as target_features','import legospec.training.target.setup as target_features').replace('from legospec.modeling.target.sglang_gemma4_training import (\n    Gemma4SGLangTrainingTargetModel,\n)','from legospec.modeling.target.sglang import SGLangTrainingTargetModel\nfrom legospec.modeling.target.adapters.gemma import GemmaTargetAdapter')
a=s.index('def target():');b=s.index('\n\n@pytest.mark.parametrize',a)
s=s[:a]+'''def target():
    engine = CaptureFixture()
    adapter = GemmaTargetAdapter(SimpleNamespace(hidden_size=4, vocab_size=32, final_logit_softcapping=3.0))
    target = SGLangTrainingTargetModel(engine.runner, adapter=adapter, engine=engine)
    target.capture_layer_ids = [1, 3]
    return target
'''+s[b:]
a=s.index('def test_factory_routes');b=s.index('\n\ndef test_native_capture_ids',a)
s=s[:a]+'''def test_factory_routes_gemma_through_common_backend():
    sentinel = object()
    config = SimpleNamespace(model_type="gemma4_unified_text", num_hidden_layers=48)
    with (
        patch("legospec.modeling.config_utils.load_text_model_config", return_value=config),
        patch.object(SGLangTrainingTargetModel, "from_pretrained", return_value=sentinel) as load,
    ):
        got = get_training_target_model("fixture", backend="sglang", capture_layer_ids=[1, 47])
    assert got is sentinel and load.call_args.kwargs["capture_layer_ids"] == [1, 47]
    assert isinstance(load.call_args.kwargs["adapter"], GemmaTargetAdapter)
    assert load.call_args.kwargs["attention_backend"] == "triton"
'''+s[b:]
s=s.replace('assert captured == [[1, 5, 47]]','assert captured == []');p.write_text(s)
p=unit/'test_sglang_training_prefill.py';s=p.read_text().replace('            causal,\n            [row + 10 for row in causal],','            [torch.cat((row, row + 10), dim=-1) for row in causal],\n            [row + 20 for row in causal],');s=s.replace('    torch.testing.assert_close(output.input_ids, ids)','    torch.testing.assert_close(output.final_hidden_state, (expected + 20).masked_fill(~kept[..., None], 0))\n    torch.testing.assert_close(output.input_ids, ids)');p.write_text(s)
p=unit/'test_hidden_feature_cache.py';s=p.read_text().replace('Round-trip both context/teacher streams and retain legacy cache loading.','Round-trip explicit decoder residual and normalized teacher streams.')
# Exercise current cache semantics: missing residuals must not be synthesized from teacher.
a=s.index('    def test_dataset_collator_and_teacher_routing');s=s[:a]+s[a:].replace('for separate in [False, True]:','for separate in [True]:',1)
s=s.replace('            expected = p["hidden_states"].get(35, p["final_hidden_state"])','            expected = p["hidden_states"][35]')
p.write_text(s)
# Input device conversion belongs to the execution backend.
p=root/'legospec/modeling/target/hf.py';s=p.read_text().replace('        backbone = self.adapter.hf_backbone(self.model)','        device = next(self.model.parameters()).device\n        input_ids, attention_mask, loss_mask = (tensor.to(device, non_blocking=True) for tensor in (input_ids, attention_mask, loss_mask))\n        backbone = self.adapter.hf_backbone(self.model)');p.write_text(s)
p=root/'legospec/training/execution/step.py';s=p.read_text();a=s.index('            target_inputs = (input_ids, attention_mask, loss_mask)');b=s.index('            target_output =',a);s=s[:a]+'''            target_inputs = (data["input_ids"], data["attention_mask"], data["loss_mask"])
'''+s[b:];p.write_text(s)
p=root/'legospec/training/target/features.py';s=p.read_text().replace('    context, teacher = select_target_features(','    hidden_states = hidden_states_to_cuda({layer: hidden_states[layer] for layer in draft.target_layer_ids})\n    context, teacher = select_target_features(').replace('return hidden_states_to_cuda(context),','return context,');p.write_text(s)
p=root/'legospec/modeling/target/training_target_model.py';s=p.read_text().replace('    adapter = get_model_adapter(config)','    if capture_layer_ids is None:\n        capture_layer_ids = list(range(config.num_hidden_layers))\n    else:\n        capture_layer_ids = sorted(capture_layer_ids)\n    adapter = get_model_adapter(config)');p.write_text(s)
# Anchor scorer owns its own capture hooks. Drop training capture before handing off.
p=root/'legospec/modeling/target/sglang.py';s=p.read_text().replace('        return SGLangAnchorScorer(self)','        self.adapter.remove_capture()\n        return SGLangAnchorScorer(self)').replace('super().set_capture_layers(list(layer_ids))','super().set_capture_layers(sorted(layer_ids))');p.write_text(s)
