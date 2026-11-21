"""Full-hidden transport with explicit context padding validity."""

from dataclasses import dataclass

import torch

from litespec.distributed.transport import (
    DFlashFamilyBatchPacket,
    DFlashFamilyPacketSpec,
)


@dataclass(frozen=True)
class ComposablePacketSpec(DFlashFamilyPacketSpec):
    window_num_anchors: int = 0

    @property
    def schema_id(self):
        return 30 + int(self.include_target_last_hidden) + 2 * int(self.window_num_anchors > 0)


@dataclass
class ComposableBatchPacket(DFlashFamilyBatchPacket):
    context_valid_mask: torch.Tensor
    anchor_positions: torch.Tensor | None = None
    block_keep_mask: torch.Tensor | None = None
    window_counts: torch.Tensor | None = None

    @classmethod
    def empty(cls, spec, *, device="cuda"):
        packet = DFlashFamilyBatchPacket.empty(spec, device=device)
        result = cls(
            packet.input_ids,
            packet.loss_mask,
            packet.target_context_hidden,
            packet.target_last_hidden,
            torch.empty(
                spec.batch_size, spec.max_length, dtype=torch.bool, device=device
            ),
        )
        if spec.window_num_anchors:
            result.anchor_positions = torch.empty(
                spec.batch_size, spec.window_num_anchors, dtype=torch.long, device=device)
            result.block_keep_mask = torch.empty_like(result.anchor_positions, dtype=torch.bool)
            result.window_counts = torch.empty(spec.batch_size, 3, dtype=torch.float64, device=device)
        return result
