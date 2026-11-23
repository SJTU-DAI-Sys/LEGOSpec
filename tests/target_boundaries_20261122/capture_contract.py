from pathlib import Path
root=Path.cwd()
p=root/'legospec/data/capture_engine.py';s=p.read_text();a=s.index('        args.separate_final =');b=s.index('        header = {',a);s=s[:a]+'''        args.separate_final = not args.recompute_gemma4_final_norm
        contract = "decoder_outputs_pre_final_norm_v1"
'''+s[b:];p.write_text(s)
p=root/'tests/unit/test_capture_pipeline_cuda.py';s=p.read_text().replace('"separate_final", [False, True]','"include_final_context", [False, True]').replace('def test_async_capture_preserves_reused_slot_contents(packed, separate_final):','def test_async_capture_preserves_reused_slot_contents(packed, include_final_context):');s=s.replace('    args = SimpleNamespace(','    layer_ids = [0, 1] if include_final_context else [0]\n    args = SimpleNamespace(',1).replace('capture_layer_ids=[0, 1]','capture_layer_ids=layer_ids').replace('separate_final=separate_final','separate_final=True').replace('            for layer in [0, 1]','            for layer in layer_ids')
s=s.replace('''        last = (
            torch.full(shape, 10 * iteration + 7, device="cuda", dtype=torch.bfloat16)
            if separate_final
            else None
        )''','''        last = torch.full(shape, 10 * iteration + 7, device="cuda", dtype=torch.bfloat16)''')
a=s.index('            output.packed_aux =');b=s.index('            output.token_offsets',a);s=s[:a]+'''            output.packed_aux = torch.stack([hidden[layer] for layer in layer_ids], 1)
            output.packed_last = last
'''+s[b:];s=s.replace('expected + (7 if separate_final else 1)','expected + 7').replace('== separate_final','== include_final_context');p.write_text(s)
