from legospec.modeling.draft.backbone.attention.dense import GQAAttention
from legospec.modeling.draft.backbone.attention.latent import MLAAttention

from .linear.recurrence import LinearAttention

ATTENTIONS = {
    "mha": GQAAttention,
    "gqa": GQAAttention,
    "mqa": GQAAttention,
    "mla": MLAAttention,
}


def build_attention(config, layer_idx, recipe, separate_context=False):
    if recipe["attention"]["representation"] in ("gdn", "kda"):
        return LinearAttention(config, recipe)
    return ATTENTIONS[recipe["attention"]["representation"]](
        config, layer_idx, recipe["attention"], separate_context
    )
