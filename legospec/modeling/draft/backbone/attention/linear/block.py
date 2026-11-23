"""Gated bidirectional attention confined to each independent draft block.

This branch consumes GDN/KDA's projected,
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
        self.num_key_heads = gdn_config["linear_num_key_heads"]
        self.value_head_dim = gdn_config["linear_value_head_dim"]
        self.width = self.num_heads * self.value_head_dim
        self.gate_proj = nn.Linear(hidden_size, self.num_heads, bias=True)
        self.reset_gate_parameters()

    def reset_gate_parameters(self):
        # Restore this prior after the enclosing Transformers model's post_init.
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(
            self.gate_proj.bias, math.log(self.gate_init / (1 - self.gate_init))
        )

    def forward(self, hidden, *, query=None, key_value=None):
        """Shared Q/K/V block read; output projection belongs to LinearAttention."""
        n, length, hidden_size = hidden.shape
        blocks = hidden.reshape(-1, self.block_size, hidden_size)
        if query is None or key_value is None:
            raise ValueError("Shared block_attention requires projected Q and K/V")
        key_dim = self.num_key_heads * self.head_dim
        q = query.reshape(
            -1, self.block_size, self.num_key_heads, self.head_dim
        ).transpose(1, 2)
        k, v = key_value.split((key_dim, self.width), dim=-1)
        k = k.reshape(-1, self.block_size, self.num_key_heads, self.head_dim).transpose(
            1, 2
        )
        v = v.reshape(
            -1, self.block_size, self.num_heads, self.value_head_dim
        ).transpose(1, 2)
        repeat = self.num_heads // self.num_key_heads
        if repeat != 1:
            q = q.repeat_interleave(repeat, dim=1)
            k = k.repeat_interleave(repeat, dim=1)
        return self._attend(blocks, q, k, v).flatten(-2).reshape(n, length, self.width)

    def _attend(self, blocks, q, k, v):
        # Folding anchors into the batch prevents cross-block attention.
        output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        output = output.transpose(1, 2)
        gates = (
            F.linear(
                blocks.float(),
                self.gate_proj.weight.float(),
                self.gate_proj.bias.float(),
            )
            .sigmoid()
            .to(output.dtype)
        )
        return output * gates[..., None]

    def reset_draft_parameters(self):
        self.reset_gate_parameters()
