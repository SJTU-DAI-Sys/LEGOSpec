from pathlib import Path
import ast,shutil,json
dev=Path("development/opd");dev.mkdir(parents=True,exist_ok=True)
archive=Path("tests/development/opd");archive.mkdir(parents=True,exist_ok=True)
integrations=[
"legospec/modeling/target/training_target_model.py","legospec/modeling/target/hf.py","legospec/modeling/target/sglang.py",
"legospec/modeling/draft/composable/loss_config.py","legospec/training/config.py",
"legospec/training/runtime/setup.py","legospec/training/runtime/checkpoint.py",
"legospec/training/target/setup.py","legospec/training/execution/step.py",
"legospec/training/execution/windows.py","legospec/training/logger/logging.py",
]
for name in integrations:
 dest=dev/"integration_snapshots"/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,dest)
tests=["test_target_feature_contract.py","test_e2e_tv_causal.py","test_composable_eagle3.py","test_packed_training.py"]
for name in tests:shutil.copy2(Path("tests/unit")/name,archive/name)
moves={
"legospec/modeling/target/anchor_opd.py":"target/anchor_opd.py",
"legospec/modeling/target/sglang_anchor_opd.py":"target/sglang_anchor_opd.py",
"legospec/training/execution/sampled.py":"sampled.py",
"scripts/config/anchor_comparison_sft_rnn.yaml":"configs/anchor_comparison_sft_rnn.yaml",
}
for name,target in moves.items():
 dest=dev/target;dest.parent.mkdir(parents=True,exist_ok=True);shutil.move(name,dest)
shutil.move("tests/unit/test_anchor_opd.py",archive/"test_anchor_opd.py")

def delete_nodes(path,predicate):
 p=Path(path);s=p.read_text();lines=s.splitlines(True);ranges=[]
 for node in ast.walk(ast.parse(s)):
  if predicate(node):
   start=min([node.lineno]+[d.lineno for d in getattr(node,"decorator_list",[])])-1
   ranges.append((start,node.end_lineno))
 remove=set(i for a,b in ranges for i in range(a,b))
 p.write_text("".join(line for i,line in enumerate(lines) if i not in remove))
for name in integrations:
 delete_nodes(name,lambda n: isinstance(n,ast.If) and "opd" in ast.unparse(n.test).lower())
for name in integrations[:3]:
 delete_nodes(name,lambda n: isinstance(n,ast.FunctionDef) and n.name=="build_anchor_scorer")
delete_nodes("legospec/training/execution/step.py",lambda n:isinstance(n,ast.FunctionDef) and n.name=="_anchor_values")
delete_nodes("legospec/training/config.py",lambda n:isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and any(isinstance(arg,ast.Constant) and isinstance(arg.value,str) and arg.value.startswith("--opd-") for arg in n.value.args))
p=Path("legospec/training/target/setup.py");s=p.read_text()
s=s.replace('    offline = (\n        bool(args.train_hidden_states_path)\n        and getattr(args, "opd_data_mode", "teacher_forced") != "sampled_anchor"\n    )','    offline = bool(args.train_hidden_states_path)');p.write_text(s)
p=Path("legospec/training/runtime/setup.py");s=p.read_text().replace("    SAMPLED_ANCHOR_OBJECTIVES,\n","").replace('        opd_data_mode=getattr(session.args, "opd_data_mode", "teacher_forced"),\n',"");p.write_text(s)
p=Path("legospec/modeling/draft/composable/loss_config.py");s=p.read_text().replace('SAMPLED_ANCHOR_OBJECTIVES = frozenset(("e2e_tv",))\n',"");p.write_text(s)
p=Path("legospec/training/execution/step.py");s=p.read_text()
for line in ['        opd_data_mode="teacher_forced",\n','        self.opd_data_mode = opd_data_mode\n','        opd_scorer=None,\n','        sampled_opd_batch=None,\n']:s=s.replace(line,"")
# Reject unknown external options; do not silently swallow a removed mode.
s=s.replace("        self.draft_model = draft_model",'''        if kwargs:
            raise TypeError("Unexpected TrainingStep arguments: " + ", ".join(kwargs))
        self.draft_model = draft_model''')
p.write_text(s)
delete_nodes("tests/unit/test_target_feature_contract.py",lambda n:isinstance(n,ast.FunctionDef) and "sampled_anchor" in n.name)
p=Path("tests/unit/test_target_feature_contract.py");p.write_text(p.read_text().replace("from legospec.modeling.target.anchor_opd import Qwen3AnchorScorer\n",""))
delete_nodes("tests/unit/test_packed_training.py",lambda n:isinstance(n,ast.FunctionDef) and "sampled_opd" in n.name)
p=Path("tests/unit/test_composable_eagle3.py");s=p.read_text();a=s.index('    with pytest.raises(ValueError, match="sampled_anchor"):');b=s.index("\n\ndef target_model",a)
s=s[:a]+s[b:];s=s.replace("test_padding_empty_short_sequences_and_unsupported_opd","test_padding_empty_short_sequences");p.write_text(s)
p=Path("tests/unit/test_disaggregated_streaming_windows.py");p.write_text(p.read_text().replace('args=SimpleNamespace(opd_data_mode="teacher_forced")','args=SimpleNamespace()'))
delete_nodes("tests/unit/test_e2e_tv_causal.py",lambda n:isinstance(n,ast.FunctionDef) and n.name=="test_sampled_opd_e2e_uses_one_target_rollout")
p=Path("tests/unit/test_e2e_tv_causal.py");s=p.read_text()
for line in ["import legospec.training.execution.sampled as execution_sampled\n","from legospec.training.execution import sampled as anchor_opd\n","from test_anchor_opd import fixture\n","from unittest.mock import Mock\n","from legospec.training.execution.step import TrainingStep\n"]:s=s.replace(line,"")
a=s.index('def model_with_mode(');b=s.index("\ndef inputs_for",a)
s=s[:a]+'''def model_with_mode(mode, device="cpu", backend="sdpa", conv=True):
    recipe = {
        "context": {"type": "concat_projection", "target_layer_ids": [0]},
        "backbone": {
            "type": "layer_stack",
            "layers": [{"attention": "gqa", "access": "dense"}],
        },
        "attention_configs": {"gqa": {}},
        "proposal": {"type": "rnn", "rank": 4},
        "objective": {"type": "e2e_tv"},
    }
    if mode is not None:
        recipe["backbone"]["query_attention"] = mode
    if conv:
        recipe["backbone"]["layers"][0]["local_operator"] = {
            "type": "grouped_conv", "kernel_size": 2, "group_size": 4,
        }
    config = LegoSpecConfig(
        hidden_size=16, intermediate_size=24, num_hidden_layers=1,
        num_target_layers=2, vocab_size=23, num_attention_heads=2,
        num_key_value_heads=1, head_dim=16, block_size=4, attention_dropout=0.0,
        composable_config=recipe, execution_config={"attention_backend": backend},
    )
    return LegoSpecModel(config).to(device)

'''+s[b:]
s=s.replace('"""Numerical E2E-TV, sampled replay, and causal query visibility."""','"""Numerical E2E-TV and causal query visibility."""')
s=s.replace('    wrapper, _, _ = model_with_mode("causal_parallel")\n    causal = wrapper.draft_model.eval()\n    legacy, _, _ = model_with_mode(None)\n    full = legacy.draft_model.eval()','    causal = model_with_mode("causal_parallel").eval()\n    full = model_with_mode(None).eval()')
s=s.replace('model_with_mode("causal_parallel")[0].draft_model.eval()','model_with_mode("causal_parallel").eval()')
s=s.replace('model_with_mode("causal_parallel", "cuda")[0].draft_model.eval()','model_with_mode("causal_parallel", "cuda").eval()')
s=s.replace('model_with_mode("causal_parallel", "cuda", "flex_attention")[\n        0\n    ].draft_model.eval()','model_with_mode("causal_parallel", "cuda", "flex_attention").eval()')
p.write_text(s)
Path("tests/defer_opd_20261123/moves.json").write_text(json.dumps(moves,indent=2))
