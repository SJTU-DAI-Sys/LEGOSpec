"""Confirmed-context GDN state, separate from attention and proposal caches."""

from dataclasses import dataclass, field

import torch


@dataclass
class GDNLayerState:
    recurrent_state: torch.Tensor
    kv_conv_state: torch.Tensor | None
    context_conv_kernel_size: int
    q_conv_state: torch.Tensor | None = None


@dataclass
class GDNBackboneState:
    layers: dict[int, GDNLayerState] = field(default_factory=dict)
    next_positions: torch.Tensor | None = None
    document_ids: torch.Tensor | None = None

    def validate_rollback(self, next_position):
        if self.next_positions is None:
            return
        limits = torch.as_tensor(next_position, device=self.next_positions.device)
        if limits.ndim > 1:
            limits = limits.squeeze(-1)
        if (limits < self.next_positions).any():
            raise ValueError(
                "GDN cannot crop confirmed recurrent history; restore a complete "
                "state snapshot or rebuild the requested prefix"
            )


def prepare_cache(layers, attention_indices, inputs):
    """Validate the complete commit before publishing any layer's new state."""
    if inputs.draft_state is None:
        return None, None
    if set(inputs.draft_state.layers) - attention_indices:
        raise ValueError("GDN cannot reuse an attention KV cache")
    old = inputs.draft_state.backbone_state
    n, a = inputs.anchor_positions.shape
    if a != 1 or not inputs.block_keep_mask.all():
        raise ValueError("Cached GDN inference requires one active block per sample")
    if old is not None and not isinstance(old, GDNBackboneState):
        raise ValueError("Unsupported backbone state type")
    device = inputs.draft_embeddings.device
    docs = (
        torch.zeros(n, dtype=torch.long, device=device)
        if inputs.anchor_document_ids is None
        else inputs.anchor_document_ids[:, 0].to(device)
    )
    if old is None:
        start = torch.zeros(n, dtype=torch.long, device=device)
        reset = torch.ones(n, dtype=torch.bool, device=device)
    else:
        if old.next_positions.shape != (n,) or old.document_ids.shape != (n,):
            raise ValueError("GDN cached batch size changed")
        reset = docs != old.document_ids.to(device)
        start = torch.where(reset, 0, old.next_positions.to(device))
    s = inputs.target_hidden_states.size(1)
    if inputs.context_position_ids is None and s and (start > 0).any():
        raise ValueError("GDN continuation requires absolute context_position_ids")
    positions = (
        torch.arange(s, device=device).expand(n, -1)
        if inputs.context_position_ids is None
        else inputs.context_position_ids.detach().to(device)
    )
    valid = (
        torch.ones(n, s, dtype=torch.bool, device=device)
        if inputs.context_valid_mask is None
        else inputs.context_valid_mask.detach().to(device).bool()
    )
    context_docs = (
        torch.zeros(n, s, dtype=torch.long, device=device)
        if inputs.document_ids is None
        else inputs.document_ids.detach().to(device)
    )
    end = inputs.anchor_positions[:, 0].to(device)
    for sample in range(n):
        selected = positions[sample, valid[sample]]
        expected = torch.arange(selected.numel(), device=device) + start[sample]
        if end[sample] - start[sample] != selected.numel() or not torch.equal(
            selected, expected
        ):
            raise ValueError(
                "GDN commit has duplicate, missing, future or out-of-order context positions"
            )
        if (context_docs[sample, valid[sample]] != int(docs[sample])).any():
            raise ValueError("Cached GDN context must belong to the active document")
    previous = {}
    for index, layer in enumerate(layers):
        if not layer.recurrent:
            continue
        mixer = layer.attention.mixer
        shape = (n, mixer.hv, mixer.dk, mixer.dv)
        history = mixer.context_kernel - 1
        tail_shape = (n, mixer.key_dim + mixer.value_dim, history)
        state = None if old is None else old.layers.get(index)
        if old is not None and state is None:
            raise ValueError("Incomplete GDN layer state")
        if state is not None:
            if (
                state.context_conv_kernel_size != mixer.context_kernel
                or state.recurrent_state.shape != shape
            ):
                raise ValueError("GDN state layout does not match the model recipe")
            if history and (
                state.kv_conv_state is None or state.kv_conv_state.shape != tail_shape
            ):
                raise ValueError("Invalid GDN convolution state layout")
            if not history and state.kv_conv_state is not None:
                raise ValueError("kernel=1 must not have historical convolution state")
            matrix = state.recurrent_state.to(device=device, dtype=torch.float32)
            matrix = matrix.masked_fill(reset[:, None, None, None], 0)
            tail = (
                None
                if not history
                else state.kv_conv_state.to(device).masked_fill(reset[:, None, None], 0)
            )
        else:
            matrix = torch.zeros(shape, device=device, dtype=torch.float32)
            tail = (
                None if not history else inputs.draft_embeddings.new_zeros(tail_shape)
            )
        q_tail = None
        if mixer.q_mode == "confirmed_history":
            q_shape = (n, mixer.key_dim, mixer.q_kernel - 1)
            if state is not None:
                if state.q_conv_state is None or state.q_conv_state.shape != q_shape:
                    raise ValueError("Invalid GDN query convolution state layout")
                q_tail = state.q_conv_state.to(device).masked_fill(
                    reset[:, None, None], 0
                )
            else:
                q_tail = inputs.draft_embeddings.new_zeros(q_shape)
        previous[index] = GDNLayerState(matrix, tail, mixer.context_kernel, q_tail)
    return previous, GDNBackboneState(
        next_positions=end.detach().clone(), document_ids=docs.detach().clone()
    )
