"""Flex visibility from O(context + blocks) metadata, without a Q×KV mask."""
import torch


def indexed_visibility(inputs, positions, valid, documents, query_length,
                       window_size=None, *, causal_queries=False):
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
        kb = ((k - context_length).clamp(min=0, max=query_length - 1) // block_size)
        qb = qi // block_size
        live = keep[b, qb]
        context = (k < context_length) & valid[b, ci]
        context = context & (documents[b, ci] == owners[b, qb]) & (positions[b, ci] < anchors[b, qb])
        if window_size is not None:
            context = context & (positions[b, ci] >= anchors[b, qb] - window_size)
        current = (k >= context_length) & (kb == qb) & keep[b, kb]
        if causal_queries:
            current = current & (k - context_length <= qi)
        # Unused queries need a finite key for stable SDPA/Flex backward.
        visible = (live & (context | current)) | (~live & (k == 0))
        return visible & (q < query_length) & (k < key_length)

    return mask_mod, keep.repeat_interleave(block_size, dim=1)[:, None]
