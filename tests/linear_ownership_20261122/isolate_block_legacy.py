from pathlib import Path
import ast,shutil,json
r=Path('/mnt/dai-sys/wanghanzhen/projects/DAC27/LEGOSpec_public');w=r/'tests/linear_ownership_20261122'
rel='legospec/modeling/draft/composable/backbone/attention/linear/block.py';p=r/rel;s=p.read_text()
b=w/'backup'/rel;b.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,b)
a='''            self.out_proj = nn.Linear(self.width, hidden_size, bias=False)
        self.gate_proj'''
z='''            self.out_proj = nn.Linear(self.width, hidden_size, bias=False)
            self.forward = self._legacy_forward
        self.gate_proj'''
assert a in s;s=s.replace(a,z);s=s[:s.index('    def forward(')]+'''    def forward(self, hidden, *, query=None, key_value=None):
        """Current shared-QKVO path; the legacy graph is selected at construction."""
        n, length, hidden_size = hidden.shape
        blocks = hidden.reshape(-1, self.block_size, hidden_size)
        if query is None or key_value is None:
            raise ValueError("Shared block_attention requires projected Q and K/V")
        key_dim = self.num_key_heads * self.head_dim
        q = query.reshape(-1, self.block_size, self.num_key_heads, self.head_dim).transpose(1, 2)
        k, v = key_value.split((key_dim, self.width), dim=-1)
        k = k.reshape(-1, self.block_size, self.num_key_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(-1, self.block_size, self.num_heads, self.value_head_dim).transpose(1, 2)
        repeat = self.num_heads // self.num_key_heads
        if repeat != 1:
            q = q.repeat_interleave(repeat, dim=1)
            k = k.repeat_interleave(repeat, dim=1)
        return self._attend(blocks, q, k, v).flatten(-2).reshape(n, length, self.width)

    def _legacy_forward(self, hidden, *, query=None, key_value=None):
        """Original independently projected Q/K/V/O, for old string recipes only."""
        n, length, hidden_size = hidden.shape
        blocks = hidden.reshape(-1, self.block_size, hidden_size)
        qkv = self.qkv_proj(blocks).reshape(blocks.size(0), self.block_size, 3, self.num_heads, self.head_dim)
        q, k, v = (part.transpose(1, 2) for part in qkv.unbind(dim=2))
        output = self._attend(blocks, q, k, v)
        return self.out_proj(output.flatten(-2)).reshape(n, length, hidden_size)

    def _attend(self, blocks, q, k, v):
        # Folding anchors into the batch prevents cross-block attention.
        output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        output = output.transpose(1, 2)
        gates = F.linear(blocks.float(), self.gate_proj.weight.float(), self.gate_proj.bias.float()).sigmoid().to(output.dtype)
        return output * gates[..., None]
'''
ast.parse(s);p.write_text(s)
m=json.loads((w/'files.json').read_text());m['updated'].append(rel);(w/'files.json').write_text(json.dumps(m,indent=2))
print('Selected independent QKVO once at construction; shared forward has no legacy flag branches')
