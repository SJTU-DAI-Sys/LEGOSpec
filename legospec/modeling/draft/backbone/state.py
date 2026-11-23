"""Request-local cache types and rollback operations."""

from dataclasses import dataclass, field
from typing import Optional

import torch

from .attention.linear.cache import GDNBackboneState


@dataclass
class LayerState:
    key_states: torch.Tensor
    value_states: torch.Tensor
    positions: torch.Tensor
    valid: torch.Tensor
    document_ids: torch.Tensor


@dataclass
class MLALayerState:
    """Confirmed context only; no head-expanded K/V tensors."""

    kv_latent: torch.Tensor
    key_rope: torch.Tensor
    positions: torch.Tensor
    valid: torch.Tensor
    document_ids: torch.Tensor


@dataclass
class DraftState:
    layers: dict[int, LayerState | MLALayerState] = field(default_factory=dict)
    backbone_state: Optional["GDNBackboneState"] = None
    indexer_state: Optional["IndexerKeyState"] = None

    def rollback(self, next_position: int | torch.Tensor):
        if self.backbone_state is not None:
            self.backbone_state.validate_rollback(next_position)
        if self.indexer_state is not None:
            state = self.indexer_state
            limits = torch.as_tensor(next_position, device=state.positions.device)
            if limits.ndim == 1:
                limits = limits[:, None]
            valid = state.valid & (state.positions < limits)
            columns = valid.any(0)
            self.indexer_state = IndexerKeyState(
                state.keys[:, columns],
                state.positions[:, columns],
                state.document_ids[:, columns],
                valid[:, columns],
                state.fingerprint,
            )
        for index, draft_state in list(self.layers.items()):
            limits = torch.as_tensor(next_position, device=draft_state.positions.device)
            if limits.ndim == 1:
                limits = limits[:, None]
            keep = draft_state.positions < limits
            # Rectangular cache, including inactive padding, supports unequal batch lengths.
            valid = draft_state.valid & keep
            columns = valid.any(dim=0)
            if isinstance(draft_state, MLALayerState):
                self.layers[index] = MLALayerState(
                    draft_state.kv_latent[:, columns],
                    draft_state.key_rope[:, columns],
                    draft_state.positions[:, columns],
                    valid[:, columns],
                    draft_state.document_ids[:, columns],
                )
                continue
            self.layers[index] = LayerState(
                draft_state.key_states[:, :, columns],
                draft_state.value_states[:, :, columns],
                draft_state.positions[:, columns],
                valid[:, columns],
                draft_state.document_ids[:, columns],
            )
        return self


@dataclass(frozen=True)
class IndexerKeyState:
    keys: torch.Tensor
    positions: torch.Tensor
    document_ids: torch.Tensor
    valid: torch.Tensor
    fingerprint: tuple
