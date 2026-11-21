"""Gated bidirectional attention confined to each independent draft block.

There is no historical KV cache. The shared variant consumes GDN's projected,
query-convolved Q and block-local K/V, and owns only its per-head gate.
"""

import math

from torch import nn
from torch.nn import functional as F


class GatedBlockAttention(nn.Module):
    def __init__(self, hidden_size, block_size, recipe, gdn_config=None):
        super().__init__()
        self.block_size = block_size
        self.num_heads = recipe["num_heads"]
        self.head_dim = recipe["head_dim"]
        self.gate_init = recipe["gate_init"]
        self.share_gdn_projections = recipe.get("share_gdn_projections", False)
        self.num_key_heads = self.num_heads
        self.value_head_dim = self.head_dim
        if self.share_gdn_projections:
            if gdn_config is None:
                raise ValueError("Shared block_attention requires a GDN configuration")
            self.num_key_heads = gdn_config["linear_num_key_heads"]
            self.value_head_dim = gdn_config["linear_value_head_dim"]
        self.width = self.num_heads * self.value_head_dim
        if not self.share_gdn_projections:
            self.qkv_proj = nn.Linear(hidden_size, 3 * self.width, bias=False)
            self.out_proj = nn.Linear(self.width, hidden_size, bias=False)
        self.gate_proj = nn.Linear(hidden_size, self.num_heads, bias=True)
        self.reset_gate_parameters()

    def reset_gate_parameters(self):
        # Restore this prior after the enclosing Transformers model's post_init.
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, math.log(self.gate_init / (1 - self.gate_init)))

    def forward(self, hidden, *, query=None, key_value=None):
        n, length, hidden_size = hidden.shape
        blocks = hidden.reshape(-1, self.block_size, hidden_size)
        if self.share_gdn_projections:
            if query is None or key_value is None:
                raise ValueError("Shared block_attention requires projected Q and K/V")
            key_dim = self.num_key_heads * self.head_dim
            q = query.reshape(-1, self.block_size, self.num_key_heads, self.head_dim).transpose(1, 2)
            k, v = key_value.split((key_dim, self.width), dim=-1)
            k = k.reshape(-1, self.block_size, self.num_key_heads, self.head_dim).transpose(1, 2)
            v = v.reshape(-1, self.block_size, self.num_heads, self.value_head_dim).transpose(1, 2)
            # Grouped value attention: each Q/K head serves consecutive V heads.
            repeat = self.num_heads // self.num_key_heads
            if repeat != 1:
                q = q.repeat_interleave(repeat, dim=1)
                k = k.repeat_interleave(repeat, dim=1)
        else:
            qkv = self.qkv_proj(blocks).reshape(
                blocks.size(0), self.block_size, 3, self.num_heads, self.head_dim
            )
            q, k, v = (part.transpose(1, 2) for part in qkv.unbind(dim=2))
        # Folding anchors into the batch makes cross-block access impossible.
        output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        output = output.transpose(1, 2)
        gates = F.linear(
            blocks.float(), self.gate_proj.weight.float(), self.gate_proj.bias.float()
        ).sigmoid().to(output.dtype)
        output = output * gates[..., None]
        if self.share_gdn_projections:
            return output.flatten(-2).reshape(n, length, self.width)
        return self.out_proj(output.flatten(-2)).reshape(n, length, hidden_size)
