"""Shared convolution primitives; independent of draft algorithm implementations."""

import torch
import torch.nn.functional as F
from torch import nn


class GroupedConv(nn.Module):
    """Grouped dynamic depthwise convolution within each proposal block.

    One projection of the sublayer input produces the dynamic kernel deltas
    for both the input and output side.  ``base_kernel`` starts as an identity,
    which makes enabling the module a stable extension of a DFlash backbone.
    """

    def __init__(
        self,
        hidden_size: int,
        block_size: int,
        taps: int,
        group_size: int,
    ) -> None:
        super().__init__()
        if taps < 1:
            raise ValueError("DFlash2 conv_kernel_size must be >= 1")
        if taps > block_size:
            raise ValueError(
                "DFlash2 conv_kernel_size must not exceed block_size, got "
                f"conv_kernel_size={taps}, block_size={block_size}"
            )
        if group_size < 1 or hidden_size % group_size:
            raise ValueError(
                f"DFlash2 conv_group_size={group_size} must divide "
                f"hidden_size={hidden_size}"
            )

        self.block_size = int(block_size)
        self.taps = int(taps)
        self.group_size = int(group_size)
        self.num_groups = int(hidden_size) // self.group_size

        # [input/output side, tap, channel], matching SGLang's loader contract.
        base_kernel = torch.zeros(2, self.taps, int(hidden_size))
        base_kernel[:, 0] = 1.0
        self.base_kernel = nn.Parameter(base_kernel)
        self.kernel_projection = nn.Linear(
            int(hidden_size),
            2 * self.taps * self.num_groups,
            bias=False,
        )
        nn.init.zeros_(self.kernel_projection.weight)

    def _convolve(
        self,
        hidden_states: torch.Tensor,
        delta: torch.Tensor,
        *,
        side: int,
    ) -> torch.Tensor:
        bsz, seq_len, hidden_size = hidden_states.shape
        if seq_len % self.block_size:
            raise ValueError(
                "DFlash2 convolution sequence length must be divisible by "
                f"block_size={self.block_size}, got {seq_len}"
            )

        num_blocks = seq_len // self.block_size
        blocks = hidden_states.reshape(
            bsz,
            num_blocks,
            self.block_size,
            self.num_groups,
            self.group_size,
        )
        dynamic = delta.reshape(
            bsz,
            num_blocks,
            self.block_size,
            self.taps,
            self.num_groups,
        )
        base = self.base_kernel[side].reshape(
            1,
            1,
            1,
            self.taps,
            self.num_groups,
            self.group_size,
        )
        coefficients = base + dynamic.unsqueeze(-1)

        output = coefficients[:, :, :, 0] * blocks
        for tap in range(1, self.taps):
            shifted = F.pad(
                blocks[:, :, : self.block_size - tap],
                (0, 0, 0, 0, tap, 0),
            )
            output = output + coefficients[:, :, :, tap] * shifted
        return output.reshape(bsz, seq_len, hidden_size)

    def prepare(self, hidden_states: torch.Tensor):
        coefficients = self.kernel_projection(hidden_states).reshape(
            *hidden_states.shape[:-1],
            2,
            self.taps,
            self.num_groups,
        )
        return (
            self._convolve(hidden_states, coefficients[..., 0, :, :], side=0),
            coefficients[..., 1, :, :],
        )

    def finish(
        self,
        hidden_states: torch.Tensor,
        coefficients: torch.Tensor,
    ) -> torch.Tensor:
        return self._convolve(hidden_states, coefficients, side=1)

    def reset_draft_parameters(self):
        nn.init.zeros_(self.kernel_projection.weight)


def build_local_operator(config, recipe):
    if recipe.get("type", "none") == "none":
        return None
    return GroupedConv(
        config.hidden_size,
        config.block_size,
        int(recipe.get("kernel_size", 2)),
        int(recipe.get("group_size", 16)),
    )
