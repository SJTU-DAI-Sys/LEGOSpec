from pathlib import Path
import shutil,ast
r=Path("legospec/modeling/draft/composable")
for p in list(r.iterdir()):
 if p.name=="__pycache__":shutil.rmtree(p);continue
 dest=r.parent/p.name
 if dest.exists() and p.name=="DESIGN.md":
  dest.write_text(dest.read_text()+"\n"+p.read_text());p.unlink();continue
 if dest.exists():raise RuntimeError(dest)
 p.rename(dest)
r.rmdir()
r=Path("legospec/modeling/draft")
for name in ["dense","latent"]:
 p=r/f"backbone/attention/{name}.py";p.write_text("from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm\n"+p.read_text())
p=Path("legospec/training/data/batch.py");s=p.read_text();a=s.index("def position_weights");b=s.index("\ndef cpu_copy",a);weights=s[a:b];s=s[:a]+s[b:]
a=s.index("def apply_exponential_position_decay");b=s.index("\ndef aligned_width",a);decay=s[a:b];s=s[:a]+s[b:];s=s.replace('    "apply_exponential_position_decay",\n','');p.write_text(s)
p=r/"objective/base.py";s=p.read_text().replace("from legospec.modeling.position_decay import apply_exponential_position_decay\n","");p.write_text(s+"\n\n"+weights+"\n"+decay)
p=Path("tests/unit/test_composable_eagle3.py");s=p.read_text().replace("from legospec.training.data.batch import position_weights, prepare_layout","from legospec.training.data.batch import prepare_layout\nfrom legospec.modeling.draft.objective.base import position_weights");p.write_text(s)
hooks={
 "context/fusion.py":("GatedFusionContext",'    def reset_draft_parameters(self):\n        self.reset_fusion_prior()\n'),
 "backbone/convolution.py":("GroupedConv",'    def reset_draft_parameters(self):\n        nn.init.zeros_(self.kernel_projection.weight)\n'),
 "backbone/attention/linear/block.py":("GatedBlockAttention",'    def reset_draft_parameters(self):\n        self.reset_gate_parameters()\n'),
 "proposal/markov.py":("AcceptRatePredictor",'    def reset_draft_parameters(self):\n        nn.init.zeros_(self.markov_w2.weight)\n'),
 "proposal/selector.py":("CandidateSelector",'    def reset_draft_parameters(self):\n        nn.init.zeros_(self.successor_codebook)\n'),
}
for file,(cls,hook) in hooks.items():
 p=r/file;s=p.read_text();node=next(n for n in ast.parse(s).body if isinstance(n,ast.ClassDef) and n.name==cls)
 lines=s.splitlines(True);lines[node.end_lineno:node.end_lineno]=["\n"+hook+"\n"];p.write_text("".join(lines))
p=r/"modeling.py";s=p.read_text();a=s.index("        # post_init");b=s.index("\n    def forward",a)
s=s[:a]+'''        for module in self.modules():
            reset = getattr(module, "reset_draft_parameters", None)
            if reset is not None:
                reset()
'''+s[b:];p.write_text(s)
