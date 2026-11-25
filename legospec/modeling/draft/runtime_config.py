"""Expand validated public recipes into executable layer parameters."""

import math
from copy import deepcopy

from .loss_config import SELECTOR_TYPES, normalize_training_recipe, selector_mode
from .schema import (
    ACCESS,
    ACCESS_TYPES,
    ATTENTION_TYPES,
    DSA_DEFAULTS,
    FAMILIES,
    FIELDS,
    GDN_DEFAULTS,
    MLA_DEFAULTS,
    PROPOSAL_TYPES,
    boolean,
    check_objective,
    fields,
    integer,
    number,
)


def _validate_expanded(config):
    r = deepcopy(config.composable_config)
    fields(r, FIELDS["recipe"], "composable_config")
    for required in ("context", "backbone", "attention_configs"):
        if required not in r:
            raise ValueError("composable_config requires " + required)
    stepwise = r["backbone"].get("schedule") == "stepwise"
    if stepwise:
        if r["backbone"].get("input_fusion") != "concat_hidden":
            raise ValueError("stepwise requires concat_hidden input fusion")
        if config.attention_dropout:
            raise ValueError("EAGLE3 currently requires attention_dropout=0")
        r.setdefault("proposal", {"type": "parallel"})
        if r["proposal"] != {"type": "parallel"}:
            raise ValueError("stepwise currently supports parallel projection only")
        r.setdefault(
            "objective", {"type": "soft_ce", "loss_decay_gamma": -1 / math.log(0.8)}
        )
        r["objective"].setdefault("loss_decay_gamma", -1 / math.log(0.8))
    elif "proposal" not in r or "objective" not in r:
        raise ValueError("composable_config requires proposal and objective")
    integer(config.block_size, "block_size", 2)
    integer(config.num_hidden_layers, "num_hidden_layers")
    integer(config.num_target_layers, "num_target_layers")
    fields(r["context"], FIELDS["context"], "context")
    if r["context"].get("type") not in ("concat_projection", "gated_fusion"):
        raise ValueError("context.type must be dflash, dflare or stepwise")
    if "normalize" in r["context"]:
        boolean(r["context"]["normalize"], "context.normalize")
    if r["backbone"].get("schedule", "parallel") not in ("parallel", "stepwise"):
        raise ValueError("Unsupported backbone schedule")
    if r["backbone"].get("input_fusion", "identity") not in (
        "identity",
        "concat_hidden",
    ):
        raise ValueError("Unsupported input fusion")
    if not stepwise and r["backbone"].get("input_fusion", "identity") != "identity":
        raise ValueError("parallel schedule requires identity input fusion")
    if "fusion" in r["context"]:
        if r["context"]["type"] != "gated_fusion" or r["context"]["fusion"] not in (
            "scalar_softmax",
            "channel_softmax_shared_projection",
        ):
            raise ValueError(
                "context.fusion requires dflare with scalar_softmax or channel_softmax_shared_projection"
            )
    ids = r["context"].get("target_layer_ids")
    if ids is not None:
        if not isinstance(ids, list) or not ids:
            raise ValueError("target_layer_ids must be a nonempty array or null")
        for value in ids:
            integer(value, "target_layer_ids[]", 0)
        if ids != sorted(set(ids)) or ids[-1] >= config.num_target_layers:
            raise ValueError(
                "target_layer_ids must be sorted, unique and within the target"
            )
    if "mask_token_id" in r:
        integer(r["mask_token_id"], "mask_token_id", 0)
        if r["mask_token_id"] >= config.vocab_size:
            raise ValueError("mask_token_id is outside vocab_size")
    b = r["backbone"]
    fields(b, FIELDS["backbone"], "backbone")
    if (
        b.get("type") not in ("layer_stack",)
        or not isinstance(b.get("layers"), list)
        or not b["layers"]
    ):
        raise ValueError("backbone.type must be layer_stack with nonempty layers")
    if len(b["layers"]) != config.num_hidden_layers:
        raise ValueError("layers length must equal num_hidden_layers")
    query_attention = b.get("query_attention", "full_parallel")
    if query_attention not in ("full_parallel", "causal_parallel"):
        raise ValueError(
            "backbone.query_attention must be full_parallel or causal_parallel"
        )
    if (
        query_attention == "causal_parallel"
        and r["context"]["type"] != "concat_projection"
    ):
        raise ValueError("causal_parallel requires DFlash context")
    used, accesses = set(), set()
    for layer in b["layers"]:
        fields(layer, FIELDS["layer"], "layers[]")
        kind, access = layer.get("attention"), layer.get("access")
        if query_attention == "causal_parallel" and (kind, access) != ("gqa", "dense"):
            raise ValueError("causal_parallel requires dense GQA in every layer")
        if kind not in ATTENTION_TYPES or access not in ACCESS_TYPES:
            raise ValueError("each layer requires an attention type and access")
        if "window_size" in layer:
            if access != "sliding_window":
                raise ValueError("layers[].window_size requires sliding_window access")
            integer(layer["window_size"], "layers[].window_size")
        if (kind in ("gdn", "kda")) != (access == "linear"):
            raise ValueError(
                "invalid attention/access combination: " + kind + "/" + access
            )
        if "block_attention" in layer:
            if kind not in ("gdn", "kda"):
                raise ValueError("block_attention requires a GDN layer")
            block_attention = layer["block_attention"]
            fields(block_attention, FIELDS["block_attention"], "block_attention")
            gate_init = block_attention.setdefault("gate_init", 0.1)
            number(gate_init, "block_attention.gate_init", strict=True)
            if gate_init >= 1:
                raise ValueError(
                    "block_attention.gate_init must be strictly between 0 and 1"
                )
        used.add(kind)
        accesses.add(access)
        op = layer.get("local_operator", {"type": "none"})
        fields(op, FIELDS["local_operator"], "local_operator")
        if op.get("type") not in ("none", "grouped_conv"):
            raise ValueError("unsupported local_operator.type")
        if op["type"] == "none" and set(op) != {"type"}:
            raise ValueError("local_operator=none has no parameters")
        if op["type"] == "grouped_conv":
            integer(op.setdefault("kernel_size", 2), "local_operator.kernel_size")
            integer(op.setdefault("group_size", 16), "local_operator.group_size")
            if (
                op["kernel_size"] > config.block_size
                or config.hidden_size % op["group_size"]
            ):
                raise ValueError(
                    "local convolution requires kernel_size <= block_size and group_size dividing hidden_size"
                )
        if "local_operator" in layer:
            layer["local_operator"] = op
        if stepwise and (
            len(b["layers"]) != 1
            or kind != "gqa"
            or access != "dense"
            or op["type"] != "none"
        ):
            raise ValueError(
                "EAGLE3 requires exactly one dense GQA layer without local operators"
            )
    profiles = r["attention_configs"]
    fields(profiles, ATTENTION_TYPES, "attention_configs")
    if set(profiles) != used:
        raise ValueError(
            "attention_configs must contain exactly the attention types used by layers"
        )
    target = config.target_attention
    fields(target, FIELDS["target_attention"], "target_attention")
    for key in FIELDS["target_attention"]:
        integer(target.get(key), "target_attention." + key)
    if target["num_attention_heads"] % target["num_key_value_heads"]:
        raise ValueError("target Q heads must be divisible by target KV heads")
    for kind, profile in profiles.items():
        fields(profile, FIELDS[kind], "attention_configs." + kind)
        if kind in ("mha", "gqa", "mqa"):
            q = target["num_attention_heads"]
            defaults = dict(
                num_attention_heads=q,
                num_key_value_heads=(
                    q
                    if kind == "mha"
                    else 1
                    if kind == "mqa"
                    else target["num_key_value_heads"]
                ),
                head_dim=target["head_dim"],
            )
        else:
            defaults = MLA_DEFAULTS if kind == "mla" else GDN_DEFAULTS
        profile = profiles[kind] = {**defaults, **profile}
        for key, value in profile.items():
            if key == "q_lora_rank" and value is None:
                continue
            if key == "rope_interleave":
                boolean(value, kind + "." + key)
            elif key == "query_conv_mode":
                if value not in ("block", "confirmed_history"):
                    raise ValueError(
                        "query_conv_mode must be block or confirmed_history"
                    )
            else:
                integer(value, kind + "." + key)
        if kind in ("mha", "gqa", "mqa"):
            if (
                profile["num_attention_heads"] % profile["num_key_value_heads"]
                or profile["head_dim"] % 2
            ):
                raise ValueError("MHA/GQA requires KV dividing Q and even head_dim")
            if (
                kind == "mha"
                and profile["num_attention_heads"] != profile["num_key_value_heads"]
            ):
                raise ValueError(
                    "MHA requires equal Q/K/V head counts; use gqa for grouped KV"
                )
        if kind == "mqa" and profile["num_key_value_heads"] != 1:
            raise ValueError("MQA requires exactly one KV head")
        if (
            kind == "kda"
            and profile["linear_key_head_dim"] > 256
            and config.execution_config["gdn_backend"] != "reference"
        ):
            raise ValueError("KDA FLA supports key head dimension <=256")
        if kind == "mla" and (
            profile["num_attention_heads"] != profile["num_key_value_heads"]
            or profile["qk_rope_head_dim"] % 2
        ):
            raise ValueError(
                "expanded MLA requires equal head counts and even RoPE dimension"
            )
        if kind in ("gdn", "kda"):
            if profile["linear_num_value_heads"] % profile["linear_num_key_heads"]:
                raise ValueError("GDN V heads must be a multiple of Q/K heads")
            if (
                profile["query_conv_mode"] == "confirmed_history"
                and profile["query_conv_kernel_size"] != 4
            ):
                raise ValueError("confirmed_history requires Q conv4")
    ac = r.setdefault("access_configs", {})
    fields(ac, ACCESS_TYPES, "access_configs")
    if set(ac) - accesses:
        raise ValueError("access_configs contains an unused access type")
    for access in sorted(accesses):
        profile = ac.setdefault(access, {})
        fields(profile, FIELDS[access], "access_configs." + access)
        if access == "sliding_window":
            integer(profile.setdefault("window_size", 512), "window_size")
        if access == "dsa":
            profile = ac[access] = {**DSA_DEFAULTS, **profile}
            for key in (
                "num_query_heads",
                "head_dim",
                "index_topk",
                "anchor_chunk_size",
            ):
                integer(profile[key], "dsa." + key)
            integer(profile["rope_head_dim"], "dsa.rope_head_dim", 0)
            if (
                profile["rope_head_dim"] % 2
                or profile["rope_head_dim"] > profile["head_dim"]
            ):
                raise ValueError("DSA rope_head_dim must be even and <= head_dim")
            if "rope_theta" in profile:
                number(profile["rope_theta"], "dsa.rope_theta", strict=True)
            number(profile["loss_weight"], "dsa.loss_weight")
            boolean(profile["enabled"], "dsa.enabled")
            if profile["stage"] not in ("warmup", "sparse"):
                raise ValueError("invalid DSA stage")
    p = r.get("proposal", {})
    fields(p, set().union(*(FIELDS[name] for name in PROPOSAL_TYPES)), "proposal")
    if not stepwise and p.get("type") not in PROPOSAL_TYPES:
        raise ValueError("unsupported proposal.type")
    fields(p, FIELDS.get(p.get("type"), set()), "proposal")
    if p and p["type"] != "parallel":
        integer(p.setdefault("rank", 256), "proposal.rank")
    if p.get("type") in SELECTOR_TYPES:
        integer(p.setdefault("top_k", 16), "selector.top_k")
        if p["top_k"] > config.vocab_size:
            raise ValueError("selector.top_k exceeds vocab_size")
        boolean(p.setdefault("stop_gradient", False), "selector.stop_gradient")
        p["selector_mode"] = selector_mode(p)
    if "dsa" in accesses:
        if (
            any(layer["access"] != "dsa" for layer in b["layers"])
            or p["type"] in SELECTOR_TYPES
        ):
            raise ValueError("DSA requires all layers using dsa and no Selector")
        if config.execution_config["attention_backend"] != "sdpa":
            raise ValueError("DSA currently supports SDPA only")
    if config.execution_config["attention_backend"] == "flex_attention":
        if config.attention_dropout:
            raise ValueError("FlexAttention requires attention_dropout=0")
        for kind, profile in profiles.items():
            if kind not in ("gdn", "kda"):
                qk = (
                    profile["qk_nope_head_dim"] + profile["qk_rope_head_dim"]
                    if kind == "mla"
                    else profile["head_dim"]
                )
                vd = profile["v_head_dim"] if kind == "mla" else profile["head_dim"]
                if min(qk, vd) < 16:
                    raise ValueError("FlexAttention requires QK/V dimensions >=16")
    check_objective(r["objective"])
    if (
        p.get("type") not in SELECTOR_TYPES
        and r["objective"].get("type") == "selector_losses"
    ):
        raise ValueError("selector_losses requires proposal=selector")
    r = normalize_training_recipe(r)
    return r


def runtime_recipe(config):
    """Expand only in memory: serialized recipes stay compact and shared."""
    config = _execution_recipe(config, validate=False)
    r = deepcopy(config.composable_config)
    profiles, accesses = r.pop("attention_configs"), r.pop("access_configs")
    layers = []
    for layer in r["backbone"]["layers"]:
        kind, access = layer["attention"], layer["access"]
        if kind in ("gdn", "kda"):
            layers.append(
                {
                    "attention": {"representation": kind, "access": "linear"},
                    "gdn_config": {
                        **profiles[kind],
                        "variant": kind,
                        "backend": config.execution_config["gdn_backend"],
                    },
                    **(
                        {"local_operator": layer["local_operator"]}
                        if "local_operator" in layer
                        else {}
                    ),
                    **(
                        {"block_attention": layer["block_attention"]}
                        if "block_attention" in layer
                        else {}
                    ),
                }
            )
        else:
            attention = {
                "representation": kind,
                "access": access,
                **profiles[kind],
                "backend": config.execution_config["attention_backend"],
            }
            if attention["backend"] == "flex_attention":
                attention["flex_backend"] = config.execution_config[
                    "flex_attention_backend"
                ]
            if access == "sliding_window":
                attention.update(accesses[access])
                if "window_size" in layer:
                    attention["window_size"] = layer["window_size"]
            layers.append(
                {
                    "attention": attention,
                    **(
                        {"local_operator": layer["local_operator"]}
                        if "local_operator" in layer
                        else {}
                    ),
                }
            )
    for layer in layers:
        if layer["attention"]["representation"] in ("gdn", "kda"):
            branch = layer.pop("block_attention")
            layer["gdn_config"]["gate_init"] = branch["gate_init"]
    r["backbone"]["layers"] = layers
    if "dsa" in accesses:
        r["indexer"] = accesses["dsa"]
    return r


def _execution_recipe(config, *, validate=True):
    result = deepcopy(config)
    r = deepcopy(config.composable_config)
    profiles = r["attention_configs"]
    gates = {}
    for variant in ("gdn", "kda"):
        if variant in profiles:
            gate = profiles[variant].pop("gate_init", 0.1)
            number(gate, variant + ".gate_init", strict=True)
            if gate >= 1:
                raise ValueError("gate_init must be between zero and one")
            gates[variant] = gate
    for layer in r["backbone"]["layers"]:
        fields(layer, {"attention", "local_operator"}, "layers[]")
        attn = layer["attention"]
        fields(
            attn, {"family", "variant", "access", "window_size"}, "layers[].attention"
        )
        family, variant = attn.get("family"), attn.get("variant")
        if family not in FAMILIES or variant not in FAMILIES[family]:
            raise ValueError("Invalid attention family/variant")
        layer["attention"] = variant
        if family == "linear":
            if "access" in attn or "window_size" in attn:
                raise ValueError(
                    "Linear attention reads recurrent state; token access does not apply"
                )
            layer["access"] = "linear"
            layer["block_attention"] = {
                "gate_init": gates.get(variant, 0.1),
            }
        else:
            access = attn.get("access", "full")
            if access not in ACCESS:
                raise ValueError("Token access must be full, swa or dsa")
            layer["access"] = ACCESS[access]
            if "window_size" in attn:
                layer["window_size"] = attn["window_size"]
    ac = r.setdefault("access_configs", {})
    fields(ac, set(ACCESS), "access_configs")
    r["access_configs"] = {ACCESS[k]: v for k, v in ac.items()}
    result.composable_config = r
    if validate:
        result.composable_config = _validate_expanded(result)
    return result
