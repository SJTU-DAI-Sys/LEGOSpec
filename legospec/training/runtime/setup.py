"""Build draft models and assemble one training session; no batch execution."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel,
    MixedPrecision,
    ShardingStrategy,
)
from transformers import AutoTokenizer

from legospec.distributed import get_tp_group, print_on_rank0
from legospec.modeling.config_utils import load_text_model_config
from legospec.modeling.draft import LegoSpecConfig, LegoSpecModel
from legospec.modeling.draft.loss_config import (
    normalize_training_recipe,
)
from legospec.modeling.draft.schema import fields
from legospec.training.data.loader import (
    _build_target_loader,
    build_loader,
)
from legospec.training.execution.step import TrainingStep
from legospec.training.runtime.checkpoint import (
    _load_common_state,
)
from legospec.training.runtime.state import phases
from legospec.training.target.setup import (
    build_target_and_components,
    build_target_model,
    resolve_tokenizer_and_components,
)
from legospec.training.target.streaming import packet_metadata, target_proxies


def wrap_training_model(draft_online, components, *, process_group=None):
    return FullyShardedDataParallel(
        draft_online,
        process_group=process_group,
        ignored_modules=[components.lm_head, components.embed_tokens],
        use_orig_params=True,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
            cast_root_forward_inputs=getattr(
                getattr(draft_online, "draft_model", None), "method_config_key", None
            )
            != "composable_config",
        ),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
    )


def validate_tp_draft_sharding(args) -> Optional[int]:
    """Validate one-rank/one-sample draft sharding and return the TP rank."""
    if not args.shard_draft_by_tp:
        return None
    if args.target_model_backend != "sglang":
        raise ValueError(
            "--shard-draft-by-tp requires --target-model-backend sglang so the target is actually tensor parallel."
        )
    tp_group = get_tp_group()
    tp_size = dist.get_world_size(tp_group)
    if tp_size <= 1:
        raise ValueError("--shard-draft-by-tp requires --tp-size > 1.")
    if int(args.batch_size) != tp_size:
        raise ValueError(
            f"One-sample-per-TP-rank draft sharding requires target batch size to equal tp_size; got batch_size={args.batch_size}, tp_size={tp_size}."
        )
    print_on_rank0(
        f"shard-draft-by-tp enabled: target batch={args.batch_size}; each TP rank trains one distinct draft sample."
    )
    return dist.get_rank(tp_group)


def build_training_session(args):
    """Prepare resources once; no training iteration or optimizer update here."""
    from legospec.distributed import init_disaggregated, init_distributed
    from legospec.distributed.transport import build_node_routes
    from legospec.training.logger.logging import create_tracker
    from legospec.training.optim.fsdp_layout import build_fsdp_layouts
    from legospec.training.optim.muon import muon_config_from_args
    from legospec.training.optim.optimizer import (
        BF16Optimizer,
        optimizer_step_count,
        register_optimizer_parameters,
    )
    from legospec.training.runtime.checkpoint import (
        load_distributed_training_state,
        load_training_state,
        reserve_output_dir,
    )
    from legospec.training.runtime.state import TrainingSession

    session = TrainingSession(args, phases(args))
    if args.disaggregate:
        session.topology = init_disaggregated(
            timeout=args.dist_timeout,
            target_ranks_per_node=args.target_ranks_per_node,
            draft_ranks_per_node=args.draft_ranks_per_node,
            target_tp_size=args.target_tp_size,
            target_ep_size=args.sglang_ep_size,
        )
    else:
        init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
        session.tp_draft_rank = validate_tp_draft_sharding(args)
    args.output_dir = reserve_output_dir(args.output_dir)
    if args.disaggregate:
        topology = session.topology
        if topology.rank == 0:
            os.makedirs(args.cache_dir, exist_ok=True)
        dist.barrier()
        session.resume_state = _load_common_state(args.resume_from)
        if topology.is_draft:
            initialize(session)
        source = topology.draft_global_ranks[0]
        payload = [packet_metadata(session) if topology.rank == source else None]
        dist.broadcast_object_list(payload, src=source)
        session.packet_metadata = payload[0]
        args.block_size = session.packet_metadata["block_size"]
        tokenizer = AutoTokenizer.from_pretrained(
            args.target_model_path, trust_remote_code=args.trust_remote_code
        )
        (session.dataloader, session.steps_per_epoch) = _build_target_loader(
            args, tokenizer, topology
        )
        session.routes = build_node_routes(
            producers=topology.target_replicas_per_node,
            drafts=topology.draft_ranks_per_node,
            node_batch_size=args.node_batch_size,
        )
        if topology.is_target:
            session.target = build_target_model(args, target_proxies(session))
            return session
        drafts = [session.draft]
        (_, session.components, _) = resolve_tokenizer_and_components(
            args, drafts, standalone_components=True
        )
        args.tracker_global_rank = session.coordinator_rank
    else:
        session.resume_state = load_training_state(args.resume_from)
        initialize(session)
        drafts = [session.draft]
        (session.target, tokenizer, session.components, _) = (
            build_target_and_components(args, drafts)
        )
        session.dataloader = build_loader(session, tokenizer)
        session.steps_per_epoch = len(session.dataloader)
    session.total_steps = optimizer_step_count(
        session.phases, session.steps_per_epoch, args.accumulation_steps
    )
    if session.total_steps <= 0:
        raise ValueError("The training schedule has no optimizer steps")
    session.draft_online = make_online(session)
    muon_config = muon_config_from_args(args)
    parameters = specs = layouts = None
    if muon_config is not None:
        (parameters, specs) = register_optimizer_parameters(
            session.draft, optimizer_parameters(session)
        )
    session.fsdp_model = wrap_training_model(
        session.draft_online, session.components, process_group=session.process_group
    )
    if muon_config is not None:
        layouts = build_fsdp_layouts(
            session.fsdp_model, parameters, specs, session.process_group
        )
    session.optimizer = BF16Optimizer(
        session.draft,
        parameters=parameters
        if muon_config is not None
        else optimizer_parameters(session),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        total_steps=session.total_steps,
        process_group=session.process_group,
        muon_config=muon_config,
        parameter_specs=specs,
        parameter_layouts=layouts,
    )
    if session.resume_state is not None:
        if args.disaggregate:
            session.resume_state = load_distributed_training_state(
                args.resume_from, process_group=session.process_group
            )
        from legospec.training.runtime.state import RunState

        session.state = RunState(**session.resume_state["cursor"])
        session.optimizer.load_state_dict(session.resume_state)
    session.tracker = create_tracker(args, args.output_dir)
    if session.dataloader is not None:
        session.dataloader.generator = torch.Generator().manual_seed(args.seed)
    return session


def build_config(args, *, source_config=None, model_role=None):
    target = (
        load_text_model_config(
            args.target_model_path, trust_remote_code=args.trust_remote_code
        )
        if source_config is None
        else copy.deepcopy(source_config)
    )
    payload = json.loads(Path(args.draft_config).read_text())
    fields(
        payload,
        {"architectures", "block_size", "composable_config"},
        "draft JSON",
    )
    if "architectures" in payload and payload["architectures"] != ["LegoSpecModel"]:
        raise ValueError("architectures must be [LegoSpecModel]")
    recipe = copy.deepcopy(payload["composable_config"])
    from legospec.modeling.target.adapters.base import get_model_adapter

    values = get_model_adapter(target).draft_config_values()
    target_depth = getattr(target, "num_target_layers", values["num_hidden_layers"])
    attention_keys = ("num_attention_heads", "num_key_value_heads", "head_dim")
    reference_attention = {key: values[key] for key in attention_keys}
    for key in (
        "model_type",
        "architectures",
        "auto_map",
        "layer_types",
        "composable_config",
        "block_size",
        "num_target_layers",
        "execution_config",
        "target_attention",
        "q_lora_rank",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
    ):
        values.pop(key, None)
    values["num_hidden_layers"] = len(recipe["backbone"]["layers"])
    draft_intermediate_size = getattr(args, "draft_intermediate_size", None)
    if draft_intermediate_size is not None:
        values["intermediate_size"] = int(draft_intermediate_size)
    execution = {
        "attention_backend": args.override_attention_backend or args.attention_backend,
        "flex_attention_backend": args.flex_attention_backend,
        "gdn_backend": getattr(args, "gdn_backend", "auto"),
    }
    config = LegoSpecConfig(
        **values,
        composable_config=recipe,
        block_size=payload.get("block_size", args.block_size),
        num_target_layers=target_depth,
        target_attention=reference_attention,
        execution_config=execution,
    )
    if config.indexer_config is not None and (not config.indexer_config["enabled"]):
        raise ValueError("DSA enabled=false is inference/reference only")
    if args.target_layer_ids:
        config.composable_config["context"]["target_layer_ids"] = [
            int(value) for value in args.target_layer_ids.split(",")
        ]
    config.validate_recipe()
    config.composable_config = normalize_training_recipe(config.composable_config)
    return config


def build_model(args, *, source_config=None, model_role=None):
    import torch

    config = build_config(args, source_config=source_config, model_role=model_role)
    return LegoSpecModel(config).cuda().to(torch.bfloat16)


def optimizer_parameters(session):
    return None


def initialize(session):
    args = session.args
    source = args.resume_from or args.init_from
    if args.init_from and (not args.resume_from):
        from safetensors.torch import load_file

        draft = build_model(args)
        state = load_file(str(Path(args.init_from) / "model.safetensors"))
        loading = draft.load_state_dict(state, strict=False)
        if loading.unexpected_keys or any(
            (not key.startswith("backbone.indexer.") for key in loading.missing_keys)
        ):
            raise ValueError("Initialization checkpoint mismatch: " + str(loading))
        del state
    elif source:
        (draft, loading) = LegoSpecModel.from_pretrained(
            source, torch_dtype=torch.bfloat16, output_loading_info=True
        )
        if any(
            (
                loading.get(key)
                for key in ("missing_keys", "unexpected_keys", "mismatched_keys")
            )
        ):
            raise ValueError(
                "Composable checkpoint weights do not match its model recipe: "
                + str(loading)
            )
        draft = draft.cuda()
    else:
        draft = build_model(args)
    draft.requires_grad_(True)
    if (
        not draft.uses_hidden_feedback
        and getattr(args, "supervision_mode", "sampled") != "sampled"
    ):
        raise ValueError("supervision_mode=all is currently EAGLE3-only")
    if draft.uses_hidden_feedback:
        args.complete_blocks = True
    draft.config.composable_config = normalize_training_recipe(
        draft.config.composable_config
    )
    args.require_target_last_hidden = True
    indexer = draft.config.indexer_config
    if indexer is not None and indexer["stage"] == "warmup":
        draft.requires_grad_(False)
        draft.backbone.indexer.requires_grad_(True)
    draft.backbone.gradient_checkpointing = args.backbone_gradient_checkpointing
    session.draft = draft
    args.block_size = draft.block_size
    args.num_draft_layers = draft.config.num_hidden_layers
    args.target_layer_ids = ",".join(map(str, draft.target_layer_ids))


def make_online(session):
    return TrainingStep(
        session.draft,
        session.components.lm_head,
        session.components.embed_tokens,
        session.args.mask_token_id,
        num_anchors=session.args.num_anchors,
        objective_chunk_blocks=session.args.objective_chunk_blocks,
        objective_gradient_checkpointing=session.args.objective_gradient_checkpointing,
        loss_process_group=session.process_group,
        complete_blocks=getattr(session.args, "complete_blocks", False),
        supervision_mode=getattr(session.args, "supervision_mode", "sampled"),
        training_mode=getattr(session.args, "training_mode", "legacy"),
    )
