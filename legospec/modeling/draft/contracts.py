"""Messages exchanged by draft components."""

from dataclasses import dataclass
from typing import Optional

import torch

from .backbone.state import DraftState


@dataclass
class DraftInputs:
    """Raw selected target features [N,S,T*D] or [N,S,T,D], embeddings [N,A*Q,D].

    anchor_positions and context_position_ids are absolute; persistent state
    accepts only newly confirmed context. previous_token_ids has shape [N,A,B].
    """

    target_hidden_states: torch.Tensor
    draft_embeddings: torch.Tensor
    anchor_positions: torch.Tensor
    block_keep_mask: torch.Tensor
    context_position_ids: Optional[torch.Tensor] = None
    context_valid_mask: Optional[torch.Tensor] = None
    document_ids: Optional[torch.Tensor] = None
    anchor_document_ids: Optional[torch.Tensor] = None
    previous_token_ids: Optional[torch.Tensor] = None
    anchor_token_ids: Optional[torch.Tensor] = None
    draft_state: Optional[DraftState] = None
    mode: str = "teacher_forcing"
    temperature: float = 0.0
    prediction_mask: Optional[torch.Tensor] = None
    collect_indexer_teacher: bool = False
    selection: Optional["SelectionView"] = None
    # Execution metadata only: does not change the module recipe/weights.
    packed_plan: Optional[object] = None
    projection_rows: Optional[object] = None


@dataclass(frozen=True)
class SelectionView:
    indices: torch.Tensor
    selected_positions: torch.Tensor
    selected_valid: torch.Tensor
    scores: torch.Tensor
    context_positions: torch.Tensor
    context_documents: torch.Tensor
    context_valid: torch.Tensor
    anchor_valid: torch.Tensor
    chunk_size: int = 16


@dataclass(frozen=True)
class IndexerAux:
    numerator: torch.Tensor
    denominator: torch.Tensor


@dataclass
class ContextOutput:
    context_hidden_states: list[torch.Tensor]
    inputs: DraftInputs


@dataclass
class BackboneOutput:
    draft_hidden_states: torch.Tensor
    draft_state: Optional[DraftState] = None
    indexer_aux: Optional[IndexerAux] = None


@dataclass
class ProposalOutput:
    draft_hidden_states: torch.Tensor
    draft_logits: Optional[torch.Tensor] = None
    base_logits: Optional[torch.Tensor] = None
    draft_token_ids: Optional[torch.Tensor] = None
    draft_probabilities: Optional[torch.Tensor] = None
    candidate_ids: Optional[torch.Tensor] = None
    candidate_logits: Optional[torch.Tensor] = None
    backbone_hidden_states: Optional[torch.Tensor] = None
    draft_state: Optional[DraftState] = None
    indexer_aux: Optional[IndexerAux] = None


@dataclass
class SharedOps:
    embed_tokens: torch.nn.Module
    lm_head: torch.nn.Module
