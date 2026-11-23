"""Strict public recipe and its private, expanded execution view.

This module is inference-only and is included in standalone exports.
"""

import math

from legospec.modeling.draft.loss_config import (
    BLOCK_OBJECTIVES,
    TOKEN_OBJECTIVES,
)

ATTENTION_TYPES = ("mha", "gqa", "mqa", "mla", "gdn", "kda")
ACCESS_TYPES = ("dense", "sliding_window", "dsa", "linear")
PROPOSAL_TYPES = (
    "parallel",
    "markov",
    "rnn",
    "sequential_rnn",
    "selector",
    "residual_selector",
    "topk_selector",
)
MLA_DEFAULTS = dict(
    num_attention_heads=16,
    num_key_value_heads=16,
    q_lora_rank=None,
    kv_lora_rank=512,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    v_head_dim=128,
    rope_interleave=True,
)
GDN_DEFAULTS = dict(
    linear_num_key_heads=16,
    linear_num_value_heads=32,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    query_conv_kernel_size=2,
    context_conv_kernel_size=4,
    query_conv_mode="block",
)
DSA_DEFAULTS = dict(
    num_query_heads=4,
    head_dim=128,
    rope_head_dim=64,
    index_topk=1024,
    anchor_chunk_size=16,
    stage="sparse",
    enabled=True,
    loss_weight=1.0,
)
EXECUTION_DEFAULTS = dict(
    attention_backend="sdpa", flex_attention_backend="TRITON", gdn_backend="auto"
)

# The field manual and coverage tests use these exact sets.
FIELDS = {
    "recipe": {
        "context",
        "backbone",
        "proposal",
        "objective",
        "mask_token_id",
        "attention_configs",
        "access_configs",
    },
    "context": {"type", "target_layer_ids", "fusion", "normalize"},
    "backbone": {"schedule", "input_fusion", "type", "layers", "query_attention"},
    "layer": {
        "attention",
        "access",
        "local_operator",
        "window_size",
        "block_attention",
    },
    "block_attention": {"gate_init"},
    "local_operator": {"type", "kernel_size", "group_size"},
    "mha": {"num_attention_heads", "num_key_value_heads", "head_dim"},
    "gqa": {"num_attention_heads", "num_key_value_heads", "head_dim"},
    "mqa": {"num_attention_heads", "num_key_value_heads", "head_dim"},
    "mla": set(MLA_DEFAULTS),
    "gdn": set(GDN_DEFAULTS),
    "kda": set(GDN_DEFAULTS),
    "dense": set(),
    "linear": set(),
    "sliding_window": {"window_size"},
    "dsa": set(DSA_DEFAULTS) | {"rope_theta"},
    "execution": set(EXECUTION_DEFAULTS),
    "target_attention": {"num_attention_heads", "num_key_value_heads", "head_dim"},
    "parallel": {"type"},
    "markov": {"type", "rank"},
    "rnn": {"type", "rank"},
    "sequential_rnn": {"type", "rank"},
    "selector": {"type", "rank", "top_k", "selector_mode", "stop_gradient"},
    "component": {"type", "weight"},
    "branch": {"weight"},
    "objective": {"type", "loss_decay_gamma"},
    "weighted_sum": {"type", "loss_decay_gamma", "components"},
    "dpace": {"type", "loss_decay_gamma", "alpha", "reduction"},
    "dpal_renyi": {"type", "loss_decay_gamma", "rho", "reduction", "local_loss"},
    "selector_losses": {"type", "loss_decay_gamma", "base", "final"},
}
FIELDS["residual_selector"] = set(FIELDS["selector"])
FIELDS["topk_selector"] = set(FIELDS["selector"])


def fields(value, allowed, path):
    if not isinstance(value, dict):
        raise ValueError(path + " must be an object")
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(path + ": unknown field(s): " + ", ".join(sorted(unknown)))


def integer(value, path, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(path + " must be an integer >= " + str(minimum))


def number(value, path, minimum=0, strict=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or (value <= minimum if strict else value < minimum)
    ):
        raise ValueError(
            path
            + " must be a finite number "
            + ("> " if strict else ">= ")
            + str(minimum)
        )


def boolean(value, path):
    if not isinstance(value, bool):
        raise ValueError(path + " must be boolean")


def execution_options(value):
    fields(value, FIELDS["execution"], "execution_config")
    result = {**EXECUTION_DEFAULTS, **value}
    for key, options in {
        "attention_backend": ("eager", "sdpa", "flex_attention"),
        "flex_attention_backend": ("TRITON", "FLASH"),
        "gdn_backend": ("auto", "fla", "reference"),
    }.items():
        if result[key] not in options:
            raise ValueError(
                "execution_config." + key + " must be one of " + str(options)
            )
    return result


def check_objective(value, path="objective", branch=False, component=False):
    kind = value.get("type") if isinstance(value, dict) else None
    if not isinstance(kind, str) or kind not in TOKEN_OBJECTIVES | BLOCK_OBJECTIVES | {
        "weighted_sum",
        "selector_losses",
    }:
        raise ValueError(path + ": unsupported objective")
    allowed = set(FIELDS.get(kind, FIELDS["objective"]))
    if branch or component:
        allowed.discard("loss_decay_gamma")
        allowed.update(FIELDS["branch"])
    if component:
        allowed = FIELDS["component"]
    fields(value, allowed, path)
    if "weight" in value:
        number(value["weight"], path + ".weight")
    if kind == "selector_losses":
        if branch or component:
            raise ValueError("selector_losses is an outer objective only")
        for side in ("base", "final"):
            if side in value:
                check_objective(value[side], path + "." + side, branch=True)
    if kind == "weighted_sum":
        if not isinstance(value.get("components"), list) or not value["components"]:
            raise ValueError(path + ".components must be a nonempty array")
        for index, item in enumerate(value["components"]):
            if not isinstance(item, dict) or item.get("type") not in TOKEN_OBJECTIVES:
                raise ValueError("weighted_sum supports token objectives only")
            check_objective(
                item, path + ".components[" + str(index) + "]", component=True
            )
            if "weight" not in item:
                raise ValueError("weighted_sum component requires weight")
    if kind == "dpace":
        alpha = value.get("alpha", 0.5)
        number(alpha, path + ".alpha")
        if (
            alpha > 1
            or value.get("reduction", "complete_block_mean") != "complete_block_mean"
        ):
            raise ValueError(
                "dpace requires alpha in [0,1] and reduction=complete_block_mean"
            )


FAMILIES = {
    "dense": ("mha", "gqa", "mqa"),
    "latent": ("mla",),
    "linear": ("gdn", "kda"),
}
ACCESS = {"full": "dense", "swa": "sliding_window", "dsa": "dsa"}


def normalize_recipe(config):
    from .runtime_config import _execution_recipe

    original = config.composable_config
    normalized = _execution_recipe(config).composable_config
    variants = [
        layer["attention"]["variant"] for layer in original["backbone"]["layers"]
    ]
    for layer, variant in zip(normalized["backbone"]["layers"], variants):
        family = next(f for f, vs in FAMILIES.items() if variant in vs)
        access = layer.pop("access")
        branch = layer.pop("block_attention", None)
        layer["attention"] = {"family": family, "variant": variant}
        if family == "linear":
            normalized["attention_configs"][variant]["gate_init"] = branch["gate_init"]
        else:
            layer["attention"]["access"] = next(
                k for k, v in ACCESS.items() if v == access
            )
            if "window_size" in layer:
                layer["attention"]["window_size"] = layer.pop("window_size")
    normalized["access_configs"] = {
        next(k for k, v in ACCESS.items() if v == key): value
        for key, value in normalized["access_configs"].items()
        if key != "linear"
    }
    return normalized
