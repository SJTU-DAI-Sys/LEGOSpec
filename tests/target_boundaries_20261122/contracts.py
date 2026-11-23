from pathlib import Path
root=Path.cwd()
p=root/'legospec/modeling/target/adapters/base.py';s=p.read_text().replace('    def configure_backend(self, backend, kwargs):','    def draft_config_values(self):\n        return self.config.to_dict()\n\n    def configure_backend(self, backend, kwargs):');p.write_text(s)
p=root/'legospec/modeling/target/adapters/gemma.py';s=p.read_text().replace('    def configure_backend(self, backend, kwargs):','''    def draft_config_values(self):
        values = self.config.to_dict()
        values.pop("per_layer_config", None)
        values["rope_parameters"] = values["rope_parameters"]["full_attention"]
        values["hidden_act"] = values.pop("hidden_activation")
        return values

    def configure_backend(self, backend, kwargs):''');p.write_text(s)
p=root/'legospec/training/runtime/setup.py';s=p.read_text().replace('    values = target.to_dict()','    from legospec.modeling.target.adapters.base import get_model_adapter\n\n    values = get_model_adapter(target).draft_config_values()');a=s.index('    if values.get("model_type") == "gemma4_unified_text":');b=s.index('    for key in (',a);s=s[:a]+'    reference_attention = {key: values[key] for key in attention_keys}\n'+s[b:];p.write_text(s)
# Do not reinterpret a normalized teacher tensor as a missing decoder residual.
p=root/'legospec/data/hidden_cache.py';s=p.read_text();s=s.replace('    if final_layer_id not in stored_layer_ids:\n        missing_layer_ids = sorted(set(missing_layer_ids) | {final_layer_id})','    if manifest.get("recompute_gemma4_final_norm") and final_layer_id not in stored_layer_ids:\n        missing_layer_ids = sorted(set(missing_layer_ids) | {final_layer_id})');s=s.replace('        needed_layer_ids = sorted(set(required_layer_ids) | {final_layer_id})','        needed_layer_ids = sorted(set(required_layer_ids) | set(missing_layer_ids))')
# Keep reader's selected residuals limited to actual requirements; recomputation needs raw final.
s=s.replace('        for layer_id in sorted(self.required_layer_ids | {self.final_layer_id}):','        layer_ids = self.required_layer_ids | ({self.final_layer_id} if self._gemma4_final_norm is not None else set())\n        for layer_id in sorted(layer_ids):');p.write_text(s)
p=root/'legospec/data/capture_engine.py';s=p.read_text().replace('None if recompute_final else getattr(output, "final_hidden_state", None)','None if recompute_final else output.final_hidden_state')
s=s.replace('                if normalized_final is None and (not recompute_final):\n                    aux_ids.remove(self.args.final_norm_layer_id)\n','')
s=s.replace('''                    cpu[
                        "teacher_final"
                        if normalized_final is not None
                        else self.args.final_norm_layer_id
                    ] = buffers["last"]''','''                    cpu["teacher_final"] = buffers["last"]''')
s=s.replace('''                normalized_final = hidden.pop("teacher_final", None)
                if normalized_final is None:
                    normalized_final = hidden.pop(self.args.final_norm_layer_id)
                payload["final_hidden_state"] = normalized_final''','''                payload["final_hidden_state"] = hidden.pop("teacher_final")''');p.write_text(s)
p=root/'tests/unit/test_hidden_feature_cache.py';s=p.read_text().replace('def test_legacy_and_new_raw_tar_roundtrip','def test_partial_and_full_context_raw_tar_roundtrip').replace('''                    if separate:
                        cpu["teacher_final"] = p["final_hidden_state"][None]
                    else:
                        cpu[35] = p["final_hidden_state"][None]''','''                    cpu["teacher_final"] = p["final_hidden_state"][None]''');s=s.replace('for separate in [True]:','for separate in [False, True]:');s=s.replace('            dataset.required_layer_ids = {1, 35}','            dataset.required_layer_ids = {1, 35} if separate else {1}');s=s.replace('            expected = p["hidden_states"][35]\n            self.assertTrue(torch.equal(item["hidden_states"][35], expected))','            context_layer = 35 if separate else 1\n            expected = p["hidden_states"][context_layer]\n            self.assertTrue(torch.equal(item["hidden_states"][context_layer], expected))');s=s.replace('                target_layer_ids=[35], config=SimpleNamespace(num_target_layers=36)','                target_layer_ids=[context_layer], config=SimpleNamespace(num_target_layers=36)');p.write_text(s)
