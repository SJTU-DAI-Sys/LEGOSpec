from legospec.modeling.draft.context.fusion import (
    ConcatProjectionContext,
    GatedFusionContext,
)

CONTEXTS = {
    "concat_projection": ConcatProjectionContext,
    "gated_fusion": GatedFusionContext,
}


def build_context(config):
    return CONTEXTS[config.composable_config["context"]["type"]](config)
