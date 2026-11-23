"""Serializable shared attention recipe; runtime expansion is never saved."""

from copy import deepcopy

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from legospec.modeling.draft.runtime_config import runtime_recipe
from legospec.modeling.draft.schema import (
    execution_options,
    normalize_recipe,
)


class LegoSpecConfig(Qwen3Config):
    model_type = "legospec"

    def __init__(
        self,
        composable_config=None,
        block_size=8,
        num_target_layers=None,
        target_attention=None,
        execution_config=None,
        **kwargs,
    ):
        if composable_config is None:
            depth = kwargs.get("num_hidden_layers", 5)
            composable_config = {
                "context": {"type": "concat_projection"},
                "backbone": {
                    "type": "layer_stack",
                    "layers": [
                        {
                            "attention": {
                                "family": "dense",
                                "variant": "gqa",
                                "access": "full",
                            }
                        }
                        for _ in range(depth)
                    ],
                },
                "proposal": {"type": "parallel"},
                "objective": {"type": "hard_ce"},
                "attention_configs": {"gqa": {}},
                "access_configs": {},
            }
        layers = composable_config.get("backbone", {}).get("layers", [])
        if "num_hidden_layers" not in kwargs:
            kwargs["num_hidden_layers"] = len(layers)
        super().__init__(**kwargs)
        self.block_size = block_size
        self.num_target_layers = (
            num_target_layers
            if num_target_layers is not None
            else self.num_hidden_layers
        )
        self.target_attention = deepcopy(
            target_attention
            if target_attention is not None
            else {
                "num_attention_heads": self.num_attention_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "head_dim": self.head_dim,
            }
        )
        self.execution_config = execution_options(execution_config or {})
        self.composable_config = deepcopy(composable_config)
        self.architectures = ["LegoSpecModel"]
        if self.composable_config["context"].get("target_layer_ids") is None:
            count = (
                5
                if self.composable_config["backbone"].get("schedule") == "stepwise"
                else self.num_hidden_layers
            )
            if self.num_target_layers < 4:
                selected = list(range(self.num_target_layers))
            elif count == 1:
                selected = [self.num_target_layers // 2]
            else:
                selected = sorted(
                    {
                        round(1 + i * (self.num_target_layers - 4) / (count - 1))
                        for i in range(count)
                    }
                )
            self.composable_config["context"]["target_layer_ids"] = selected
        self.validate_recipe()

    def validate_recipe(self):
        self.execution_config = execution_options(self.execution_config)
        self.composable_config = normalize_recipe(self)

    @property
    def indexer_config(self):
        return self.composable_config["access_configs"].get("dsa")

    def runtime_config(self):
        result = deepcopy(self)
        result.composable_config = runtime_recipe(self)
        if self.indexer_config is not None:
            first = result.composable_config["backbone"]["layers"][0]["attention"]
            result.indexer_source_width = first["num_attention_heads"] * (
                first["qk_nope_head_dim"] + first["qk_rope_head_dim"]
                if first["representation"] == "mla"
                else first["head_dim"]
            )
        return result
