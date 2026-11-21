from .base import build_visibility


def dense_visibility(inputs, positions, valid, documents, query_length, *, causal_queries=False):
    return build_visibility(inputs, positions, valid, documents, query_length, causal_queries=causal_queries)
