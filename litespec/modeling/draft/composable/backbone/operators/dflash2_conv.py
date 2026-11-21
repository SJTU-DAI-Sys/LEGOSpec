"""Use the existing source-matched dynamic convolution without formula changes."""

from ....convolution import DFlashGroupedConv


def build_dflash2_conv(config, recipe):
    return DFlashGroupedConv(
        config.hidden_size,
        config.block_size,
        int(recipe.get("kernel_size", 2)),
        int(recipe.get("group_size", 16)),
    )
