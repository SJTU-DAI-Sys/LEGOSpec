from pathlib import Path
import ast
r=Path("legospec/modeling/draft/composable")
# State owns cache mutation; contracts only hold messages.
p=r/"contracts.py";s=p.read_text();a=s.index("@dataclass\nclass LayerState");b=s.index("@dataclass\nclass DraftInputs")
state=s[a:b];a2=s.index("@dataclass(frozen=True)\nclass IndexerKeyState");b2=s.index("@dataclass(frozen=True)\nclass SelectionView")
state+=s[a2:b2]
(r/"backbone/state.py").write_text('"""Request-local cache types and rollback operations."""\nfrom dataclasses import dataclass, field\nfrom typing import Optional\nimport torch\nfrom .attention.linear.cache import GDNBackboneState\n\n'+state)
s=s[b:a2]+s[b2:];s='"""Messages exchanged by draft components."""\nfrom dataclasses import dataclass\nfrom typing import Optional\nimport torch\nfrom .backbone.state import DraftState\n\n'+s
p.write_text(s)
# Objective owns additive results and position weights.
out=Path("legospec/training/outputs.py").read_text()
(r/"objective/outputs.py").write_text(out.replace("Additive training outputs","Additive objective outputs").replace("TrainingOutput","ObjectiveOutput"))
for p in list(Path("legospec").rglob("*.py"))+list(Path("tests/unit").glob("*.py")):
 s=p.read_text()
 s=s.replace("from legospec.training.outputs import","from legospec.modeling.draft.composable.objective.outputs import").replace("TrainingOutput","ObjectiveOutput")
 p.write_text(s)
Path("legospec/training/outputs.py").unlink()
p=r/"objective/base.py";s=p.read_text().replace("from legospec.training.data.batch import position_weights","from legospec.modeling.position_decay import apply_exponential_position_decay")
# discover actual owner below if necessary
s=s.replace('        return position_weights(\n            supervision.prediction_mask, self.recipe.get("loss_decay_gamma")\n        )','''        mask = supervision.prediction_mask
        return apply_exponential_position_decay(
            mask.float(), self.recipe.get("loss_decay_gamma"),
            torch.arange(mask.size(-1), device=mask.device), first_prediction_index=0,
        )''');p.write_text(s)
# Module factory abstraction has only one provider: use native norm and MLP directly.
for p in r.rglob("*.py"):
 s=p.read_text()
 s=s.replace("from legospec.modeling.draft.composable.backbone.kernels import DEFAULT_KERNELS","from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP, Qwen3RMSNorm")
 s=s.replace("from legospec.modeling.draft.composable.backbone.kernels import BackboneKernels","")
 s=s.replace("kernels: BackboneKernels","kernels")
 s=s.replace("        kernels = DEFAULT_KERNELS\n","").replace("            kernels = DEFAULT_KERNELS\n","")
 s=s.replace("kernels.make_rms_norm(","Qwen3RMSNorm(").replace("kernels.make_mlp(","Qwen3MLP(")
 # keep signature cleanup for later
 s=s.replace("                kernels,\n","                None,\n")
 p.write_text(s)
(r/"backbone/kernels.py").unlink()
# Independent linear module. Outer layer keeps normalization, residual and external convolution.
p=r/"backbone/attention/linear/recurrence.py";s=p.read_text();a=s.index("class _LinearAttention:")
s=s[:a]+'''class LinearAttention(nn.Module):
    """GDN/KDA history read and intrinsic gated block read with shared projections."""
    def __init__(self, config, recipe):
        super().__init__()
        self.block_size = config.block_size
        linear = recipe["gdn_config"]
        self.mixer = GDNContextMemoryMixer(config, linear)
        self.position_embedding = nn.Embedding(config.block_size, config.hidden_size)
        self.block_attention = GatedBlockAttention(
            config.hidden_size, config.block_size,
            dict(share_gdn_projections=True, num_heads=linear["linear_num_value_heads"],
                 head_dim=linear["linear_key_head_dim"], gate_init=linear.get("gate_init", 0.1)),
            gdn_config=linear,
        )

    def add_positions(self, hidden):
        batch, length, width = hidden.shape
        return (hidden.reshape(batch, -1, self.block_size, width)
                + self.position_embedding.weight).reshape(batch, length, width)

    def forward(self, hidden, context, inputs, layout, previous, query_context=None):
        history, query, state = self.mixer(
            hidden, context, layout, self.block_size, previous,
            inputs=inputs, query_context=query_context,
        )
        block = self.block_attention(
            hidden, query=query, key_value=self.mixer.in_proj_kv(hidden))
        return self.mixer.out_proj((history + block.reshape_as(history)).flatten(-2)), state
'''
p.write_text(s)
p=r/"backbone/model.py";s=p.read_text().replace("    _LinearAttention,","    LinearAttention,").replace("class Layer(_LinearAttention, nn.Module):","class Layer(nn.Module):")
s=s.replace("            self._init_linear_attention(config, recipe)","            self.attention = LinearAttention(config, recipe)")
s=s.replace("self.self_attn","self.attention").replace("self._linear_positions(","self.attention.add_positions(")
s=s.replace('            draft_hidden_states, state = self._linear_forward(\n                draft_hidden_states, context_hidden_states, inputs, layout, previous\n            )','''            coefficients = None
            if self.attention_conv is not None:
                draft_hidden_states, coefficients = self.attention_conv.prepare(draft_hidden_states)
            query_context = self.input_norm(context_hidden_states) if self.attention.mixer.q_mode == "confirmed_history" else None
            draft_hidden_states, state = self.attention(
                draft_hidden_states, context_hidden_states, inputs, layout, previous, query_context)
            if self.attention_conv is not None:
                draft_hidden_states = self.attention_conv.finish(draft_hidden_states, coefficients)''')
p.write_text(s)
# Update actual consumers to owned modules/state.
for p in list(Path("legospec").rglob("*.py"))+list(Path("tests/unit").rglob("*.py")):
 s=p.read_text()
 s=s.replace("from legospec.modeling.draft.composable.contracts import (","from legospec.modeling.draft.composable.contracts import (")
 # moved types imported from contracts: separate automatically
 tree=ast.parse(s);lines=s.splitlines(True);edits=[]
 for n in ast.walk(tree):
  if isinstance(n,ast.ImportFrom) and n.module=="legospec.modeling.draft.composable.contracts":
   moved=[x.name for x in n.names if x.name in {"LayerState","MLALayerState","DraftState","IndexerKeyState"}]
   if moved:
    kept=[x.name for x in n.names if x.name not in moved]
    replacement="from legospec.modeling.draft.composable.backbone.state import "+", ".join(moved)+"\n"
    if kept:replacement+="from legospec.modeling.draft.composable.contracts import "+", ".join(kept)+"\n"
    edits.append((n.lineno-1,n.end_lineno,replacement))
 for a,b,v in sorted(edits,reverse=True):lines[a:b]=[v]
 s="".join(lines)
 # explicit layer consumers
 for old,new in [("layer.mixer","layer.attention.mixer"),("layer.block_attention","layer.attention.block_attention"),
                 ("layer.position_embedding","layer.attention.position_embedding"),("layer.self_attn","layer.attention"),
                 ("layers[0].mixer","layers[0].attention.mixer"),("layers[0].block_attention","layers[0].attention.block_attention"),
                 ("layers[0].position_embedding","layers[0].attention.position_embedding"),
                 ("layers[0].self_attn","layers[0].attention")]:
  s=s.replace(old,new)
 p.write_text(s)
# Move stepwise to execution owner.
(r/"stepwise.py").rename(r/"backbone/stepwise.py")
# position_weights other consumers will use objective helper until removed below
# Collapse single implementation directory and rewrite all active consumers.
for folder in ["legospec","scripts","tests/unit"]:
 for p in Path(folder).rglob("*.py"):
  s=p.read_text().replace("legospec.modeling.draft.composable.stepwise","legospec.modeling.draft.composable.backbone.stepwise")
  s=s.replace("legospec.modeling.draft.composable","legospec.modeling.draft")
  p.write_text(s)
for p in list(r.iterdir()):
 dest=r.parent/p.name
 if dest.exists() and dest.name=="__init__.py":dest.unlink()
 p.rename(dest)
r.rmdir()
