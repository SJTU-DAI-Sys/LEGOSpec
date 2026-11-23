from pathlib import Path
import json,re,shutil
root=Path('/mnt/dai-sys/wanghanzhen/projects/DAC27/LEGOSpec_public');work=root/'tests/linear_ownership_20261122'
base='legospec/modeling/draft/composable/backbone/'
changed=[]
def write(rel,s):
 p=root/rel;b=work/'backup'/rel
 if p.exists() and not b.exists():b.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,b)
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s)
 if rel not in changed:changed.append(rel)
# Shared softmax lifecycle is named base.py as requested; no compatibility shim.
old=base+'attention/softmax.py';new=base+'attention/base.py'
write(new,(root/old).read_text());write(old,(root/old).read_text());(root/old).unlink();changed.remove(old)
for tree in ['legospec','tests/unit','scripts','docs']:
 for p in (root/tree).rglob('*'):
  if p.suffix not in ('.py','.md'):continue
  s=p.read_text();t=s.replace('backbone.attention.softmax','backbone.attention.base').replace('softmax.py','base.py')
  if t!=s:write(str(p.relative_to(root)),t)
rel=base+'attention/linear/recurrence.py';s=(root/rel).read_text()
s=s.replace('from legospec.modeling.draft.composable.backbone.attention.linear.cache import (','from legospec.modeling.draft.composable.backbone.attention.linear.block import GatedBlockAttention\n\nfrom legospec.modeling.draft.composable.backbone.attention.linear.cache import (')
s=s.replace('        block_attention=None,\n','')
a=s.index('        if block_attention is not None:\n');b=s.index('        state = None\n',a)
s=s[:a]+s[b:]
s=s.replace('        return result, state','        return out, q.reshape(n, length, self.hk, self.dk), state')
s+='''

class _LinearAttention:
    """Own linear construction and fusion on Layer's existing parameter namespace.

    This behavior has no separate module registration: keeping mixer/block/position
    names and traversal order preserves saved weights and optimizer resume layouts.
    Legacy graph selection happens once here, never in Layer.forward.
    """

    def _init_linear_attention(self, config, recipe):
        self.position_embedding = nn.Embedding(config.block_size, config.hidden_size)
        linear = recipe["gdn_config"]
        self.mixer = GDNContextMemoryMixer(config, linear)
        if linear.get("intrinsic_block_attention"):
            block = dict(share_gdn_projections=True,
                         num_heads=linear["linear_num_value_heads"],
                         head_dim=linear["linear_key_head_dim"], gate_init=linear["gate_init"])
        else:
            # Original string recipes alone may omit the block or use separate QKVO.
            block = recipe.get("block_attention")
        self.block_attention = (GatedBlockAttention(config.hidden_size, config.block_size,
            block, gdn_config=linear) if block is not None else None)
        self._linear_forward = (self._shared_linear_attention
            if self.block_attention is not None and self.block_attention.share_gdn_projections
            else self._legacy_linear_attention)

    def _linear_positions(self, hidden):
        batch, length, width = hidden.shape
        return (hidden.reshape(batch, -1, self.block_size, width)
                + self.position_embedding.weight).reshape(batch, length, width)

    def _read_linear_attention(self, hidden, context, inputs, layout, previous):
        coefficients = None
        if self.attention_conv is not None:
            hidden, coefficients = self.attention_conv.prepare(hidden)
        query_context = self.input_norm(context) if self.mixer.q_mode == "confirmed_history" else None
        history, query, state = self.mixer(hidden, context, layout, self.block_size,
            previous, inputs=inputs, query_context=query_context)
        return hidden, history, query, state, coefficients

    def _shared_linear_attention(self, hidden, context, inputs, layout, previous):
        # Fixed current graph: history read + per-head gated block read, then one O.
        hidden, history, query, state, coefficients = self._read_linear_attention(
            hidden, context, inputs, layout, previous)
        block = self.block_attention(hidden, query=query,
                                     key_value=self.mixer.in_proj_kv(hidden))
        output = self.mixer.out_proj((history + block.reshape_as(history)).flatten(-2))
        if self.attention_conv is not None:
            output = self.attention_conv.finish(output, coefficients)
        return output, state

    def _legacy_linear_attention(self, hidden, context, inputs, layout, previous):
        # Preserve old independent-branch ordering: finish external conv first,
        # then add the block computed from the original normalized input.
        block_input = hidden
        _, history, _, state, coefficients = self._read_linear_attention(
            hidden, context, inputs, layout, previous)
        output = self.mixer.out_proj(history.flatten(-2))
        if self.attention_conv is not None:
            output = self.attention_conv.finish(output, coefficients)
        if self.block_attention is not None:
            output = output + self.block_attention(block_input)
        return output, state
'''
write(rel,s)
rel=base+'model.py';s=(root/rel).read_text()
s=s.replace('from legospec.modeling.draft.composable.backbone.attention.linear.block import (\n    GatedBlockAttention,\n)\n','').replace('    GDNContextMemoryMixer,','    _LinearAttention,').replace('class Layer(nn.Module):','class Layer(_LinearAttention, nn.Module):')
a=s.index('        if self.recurrent:\n');b=s.index('        else:\n',a)
s=s[:a]+'        if self.recurrent:\n            self._init_linear_attention(config, recipe)\n'+s[b:]
a=s.index('        if self.recurrent:\n',s.index('    def forward('));b=s.index('        else:\n',a)
s=s[:a]+'        if self.recurrent:\n            draft_hidden_states = self._linear_positions(draft_hidden_states)\n'+s[b:]
a=s.index('        block_input = draft_hidden_states\n');b=s.index('        draft_hidden_states = residual + draft_hidden_states\n',a)
s=s[:a]+'''        if self.recurrent:
            draft_hidden_states, state = self._linear_forward(
                draft_hidden_states, context_hidden_states, inputs, layout, previous)
        else:
            coefficients = None
            if self.attention_conv is not None:
                draft_hidden_states, coefficients = self.attention_conv.prepare(draft_hidden_states)
            draft_hidden_states = self.self_attn(
                draft_hidden_states, context_hidden_states, embeddings, inputs)
            if isinstance(draft_hidden_states, tuple):
                draft_hidden_states, teacher = draft_hidden_states
            if self.attention_conv is not None:
                draft_hidden_states = self.attention_conv.finish(draft_hidden_states, coefficients)
'''+s[b:]
write(rel,s)
(work/'files.json').write_text(json.dumps({'updated':changed,'removed':[old]},indent=2))
print('Updated',len(changed),'files; preserved module registration names/order')
