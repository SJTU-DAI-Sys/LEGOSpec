"""Functional hidden-feedback schedule shared by training and decoding."""

from dataclasses import dataclass

import torch
from torch.utils.checkpoint import checkpoint

from legospec.modeling.draft.backbone.attention.backends.flex import (
    create_flex_block_mask,
)


@dataclass
class HiddenFeedbackState:
    """Request-local draft state; target KV is owned by the verifier."""

    key: torch.Tensor | None = None
    value: torch.Tensor | None = None
    hidden: torch.Tensor | None = None

    @property
    def length(self):
        return 0 if self.key is None else self.key.shape[-2]

    def crop(self, length):
        if not 0 <= length <= self.length:
            raise ValueError("Invalid EAGLE3 cache crop")
        if self.key is not None:
            self.key = self.key[..., :length, :]
            self.value = self.value[..., :length, :]
        self.hidden = None


def training_mask(valid, length, step, *, block=False, backend="TRITON"):
    """First chunk: causal prefix; later chunks: same-anchor diagonal only."""
    lengths = valid.long().sum(-1) - step

    def mask_mod(b, h, q, k):
        prefix = (k < length) & (k <= q) & (k < lengths[b])
        branch = (k >= length) & (k % length == q)
        return (q < lengths[b]) & (prefix | branch)

    if block:
        return create_flex_block_mask(
            mask_mod,
            B=valid.shape[0],
            H=1,
            Q_LEN=length,
            KV_LEN=length * (step + 1),
            device=str(valid.device),
            backend=backend,
        )
    q = torch.arange(length, device=valid.device)[None, :, None]
    k = torch.arange(length * (step + 1), device=valid.device)[None, None, :]
    return mask_mod(
        torch.arange(valid.shape[0], device=valid.device)[:, None, None], 0, q, k
    )[:, None]


def rollout_hidden_feedback(
    backbone, hidden, input_ids, valid, embed_tokens, anchor_positions
):
    """All L queries run at every depth; only output gathering samples anchors."""
    length = input_ids.shape[1]
    key, value = backbone.empty_kv(hidden)
    selected = []
    positions = torch.arange(length, device=hidden.device)[None].expand(
        hidden.shape[0], -1
    )
    ids = input_ids
    for step in range((backbone.block_size - 1)):
        ids = torch.cat((ids[:, 1:], torch.zeros_like(ids[:, :1])), 1)
        use_flex = backbone.layer.attention.backend == "flex_attention"
        mask = training_mask(
            valid,
            length,
            step,
            block=use_flex,
            backend=backbone.layer.attention.flex_backend,
        )

        # Freeze loop bindings: checkpoint recomputation occurs after this loop.
        def operation(h, e, p, k, v, mask=mask, use_flex=use_flex):
            return backbone.layer(h, e, p, k, v, mask, use_flex)

        args = (hidden, embed_tokens(ids), positions + step, key, value)
        if (
            backbone.gradient_checkpointing
            and backbone.training
            and torch.is_grad_enabled()
        ):
            hidden, key, value = checkpoint(operation, *args, use_reentrant=False)
        else:
            hidden, key, value = operation(*args)
        indices = (anchor_positions - 1).clamp_min(0)
        selected.append(
            hidden.gather(1, indices[..., None].expand(-1, -1, hidden.shape[-1]))
        )
    return torch.stack(selected, 2)


def extend_hidden_feedback(backbone, hidden, token_embeddings, state):
    """Pure decoder step, then publish a new request-local cache state."""
    if hidden.shape[1] == 0 or hidden.shape != token_embeddings.shape:
        raise ValueError(
            "EAGLE3 extension requires matching nonempty hidden and embedding tensors"
        )
    previous = state.length
    key, value = (
        backbone.empty_kv(hidden) if state.key is None else (state.key, state.value)
    )
    positions = torch.arange(
        previous, previous + hidden.shape[1], device=hidden.device
    )[None]
    mask = (
        torch.arange(previous + hidden.shape[1], device=hidden.device)[None, :]
        <= positions[0, :, None]
    )[None, None]
    output, key, value = backbone.layer(
        hidden, token_embeddings, positions, key, value, mask
    )
    state.key, state.value, state.hidden = key, value, output[:, -1:]
    return state.hidden
