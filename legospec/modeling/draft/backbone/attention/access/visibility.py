"""Visibility (mask) computation for composable attention."""

import torch


def build_visibility(
    inputs,
    key_positions,
    key_valid,
    key_documents,
    query_length,
    window_size=None,
    *,
    causal_queries=False,
):
    batch_size, num_anchors = inputs.anchor_positions.shape
    block_size = query_length // num_anchors
    anchor = inputs.anchor_positions.repeat_interleave(block_size, dim=1)
    valid = inputs.block_keep_mask.repeat_interleave(block_size, dim=1)
    documents = inputs.anchor_document_ids
    if documents is None:
        documents = torch.zeros_like(inputs.anchor_positions)
    query_documents = documents.repeat_interleave(block_size, dim=1)
    context = key_positions[:, None] < anchor[:, :, None]
    if window_size is not None:
        context &= key_positions[:, None] >= anchor[:, :, None] - window_size
    context &= key_valid[:, None] & (
        key_documents[:, None] == query_documents[:, :, None]
    )
    slots = torch.arange(query_length, device=anchor.device)
    block_ids = slots // block_size
    own_block = block_ids[:, None] == block_ids[None, :]
    if causal_queries:
        own_block &= slots[:, None] >= slots[None, :]
    current = own_block[None].expand(batch_size, -1, -1)
    current = current & valid[:, None, :]
    return torch.cat((context, current), dim=-1)[:, None] & valid[:, None, :, None]


def dense_visibility(
    inputs, positions, valid, documents, query_length, *, causal_queries=False
):
    return build_visibility(
        inputs, positions, valid, documents, query_length, causal_queries=causal_queries
    )


"""DSA selects history; ordinary within-block visibility is retained."""


def dsa_visibility(inputs, positions, valid, documents, query_length, **kwargs):
    visibility = build_visibility(inputs, positions, valid, documents, query_length)
    selection = inputs.selection
    if selection is None:
        return visibility
    import torch

    n, a, _ = selection.indices.shape
    counts = torch.zeros(
        n, a, positions.size(1), dtype=torch.long, device=positions.device
    )
    counts.scatter_add_(-1, selection.indices, selection.selected_valid.long())
    keep = (counts > 0).repeat_interleave(query_length // a, 1)
    visibility[..., : positions.size(1)] &= keep[:, None]
    return visibility


"""Flex visibility from O(context + blocks) metadata, without a Q×KV mask."""


def indexed_visibility(
    inputs,
    positions,
    valid,
    documents,
    query_length,
    window_size=None,
    *,
    causal_queries=False,
):
    block_size = query_length // inputs.anchor_positions.size(1)
    anchors, keep = inputs.anchor_positions, inputs.block_keep_mask
    owners = inputs.anchor_document_ids
    if owners is None:
        owners = torch.zeros_like(anchors)
    context_length = positions.size(1)
    key_length = context_length + query_length

    def mask_mod(b, h, q, k):
        # Flex visits the padded tile perimeter as well as real tokens.
        qi = q.clamp(max=query_length - 1)
        ci = k.clamp(max=context_length - 1)
        kb = (k - context_length).clamp(min=0, max=query_length - 1) // block_size
        qb = qi // block_size
        live = keep[b, qb]
        context = (k < context_length) & valid[b, ci]
        context = (
            context
            & (documents[b, ci] == owners[b, qb])
            & (positions[b, ci] < anchors[b, qb])
        )
        if window_size is not None:
            context = context & (positions[b, ci] >= anchors[b, qb] - window_size)
        current = (k >= context_length) & (kb == qb) & keep[b, kb]
        if causal_queries:
            current = current & (k - context_length <= qi)
        # Unused queries need a finite key for stable SDPA/Flex backward.
        visible = (live & (context | current)) | (~live & (k == 0))
        return visible & (q < query_length) & (k < key_length)

    return mask_mod, keep.repeat_interleave(block_size, dim=1)[:, None]


"""W historical positions plus the WHOLE bidirectional current block."""


def sliding_window_visibility(
    inputs, positions, valid, documents, query_length, window_size
):
    return build_visibility(
        inputs, positions, valid, documents, query_length, window_size
    )


ACCESS = {
    "dense": dense_visibility,
    "dsa": dsa_visibility,
    "sliding_window": sliding_window_visibility,
}


def build_access(config):
    return ACCESS[config.get("access", "dense")]
