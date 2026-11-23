"""Data, target, topology and FSDP preparation; no training loops."""

from __future__ import annotations
import hashlib
import math
import os
from typing import Optional
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel,
    MixedPrecision,
    ShardingStrategy,
)
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer
from legospec.data.json_training import load_training_json
from legospec.args import SGLangBackendArgs
from legospec.data import (
    RegenFullCollator,
    RegenFullDataset,
    ShardAwareDistributedSampler,
    build_training_dataset,
    load_hidden_cache_manifest,
    prepare_dp_dataloaders,
)
from legospec.distributed import get_dp_group, get_tp_group, print_on_rank0
from legospec.modeling.config_utils import is_qwen35_model_type, load_text_model_config
from legospec.modeling.target.target_utils import (
    SGLangTPEmbeddingAdapter,
    SGLangTPLMHeadAdapter,
    SharedTargetEmbeddingsAndHead,
    TargetEmbeddingsAndHead,
)
from legospec.modeling.target.training_target_model import get_training_target_model
from legospec.training.checkpoint import validate_training_identity


def _build_processed_dataset(
    args,
    tokenizer,
    *,
    train_data_path: str,
    cache_namespace: str,
    num_proc: Optional[int] = None,
):
    columns = getattr(args, "train_data_columns", None)
    columns_key = f"-columns={columns!r}" if columns is not None else ""
    cache_key = hashlib.md5(
        f"{train_data_path}-{args.max_length}-{args.chat_template}-{args.target_model_path}-preformatted={args.is_preformatted}{columns_key}".encode()
    ).hexdigest()
    raw_dataset = load_training_json(
        train_data_path,
        columns=columns,
        cache_dir=os.path.join(args.cache_dir, cache_namespace, "raw_dataset"),
    )
    return build_training_dataset(
        dataset=raw_dataset,
        tokenizer=tokenizer,
        chat_template=args.chat_template,
        max_length=args.max_length,
        is_preformatted=args.is_preformatted,
        cache_dir=os.path.join(args.cache_dir, cache_namespace, "processed_dataset"),
        cache_key=cache_key,
        num_proc=args.build_dataset_num_proc if num_proc is None else int(num_proc),
    )


def _has_valid_anchor_supervision(value, *, block_size: int) -> bool:
    """Match the anchor sampler's per-example supervision requirements."""
    loss_mask = torch.as_tensor(value["loss_mask"]).reshape(-1)
    minimum = 2 * int(block_size)
    if int(loss_mask.sum().item()) < minimum:
        return False
    max_anchor = int(loss_mask.numel()) - int(block_size)
    if max_anchor < 0:
        return False
    current = loss_mask[: max_anchor + 1] > 0.5
    following = loss_mask[1 : max_anchor + 2] > 0.5
    return bool((current & following).any().item())


def _prepare_dataloader(args, dataset, *, train_data_path: str):
    minimum = 2 * int(args.block_size)
    dataset = dataset.filter(
        _has_valid_anchor_supervision,
        fn_kwargs={"block_size": int(args.block_size)},
        desc=f"Filtering examples with at least {minimum} labels and one trainable anchor",
    )
    dataloader = prepare_dp_dataloaders(
        dataset,
        args.batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=True,
        process_group=get_dp_group(),
        pad_to_length=args.max_length if args.pad_to_max_length else None,
    )
    if len(dataloader) == 0:
        raise ValueError(
            f"Training dataset {train_data_path!r} has no full batches after filtering."
        )
    return dataloader


def build_train_dataloader(
    args,
    tokenizer,
    *,
    train_data_path: Optional[str] = None,
    cache_namespace: str = "single",
    num_proc: Optional[int] = None,
    required_layer_ids: Optional[set[int]] = None,
):
    if args.train_hidden_states_path:
        if train_data_path is not None:
            raise ValueError("online train_data_path cannot be used in offline mode")
        return build_hidden_cache_dataloader(
            args, tokenizer, required_layer_ids=set(required_layer_ids or ())
        )
    train_data_path = train_data_path or args.train_data_path
    if not train_data_path:
        raise ValueError("A training data path is required.")
    dataset = None
    if dist.get_rank() == 0:
        dataset = _build_processed_dataset(
            args,
            tokenizer,
            train_data_path=train_data_path,
            cache_namespace=cache_namespace,
            num_proc=num_proc,
        )
    dist.barrier()
    if dataset is None:
        dataset = _build_processed_dataset(
            args,
            tokenizer,
            train_data_path=train_data_path,
            cache_namespace=cache_namespace,
            num_proc=num_proc,
        )
    return _prepare_dataloader(args, dataset, train_data_path=train_data_path)


def build_hidden_cache_dataloader(args, tokenizer, *, required_layer_ids: set[int]):
    target_config = load_text_model_config(
        args.target_model_path, trust_remote_code=args.trust_remote_code
    )
    verify = bool(getattr(args, "verify_hidden_cache_checksums", False)) and (
        not dist.is_initialized() or dist.get_rank() == 0
    )
    manifest = load_hidden_cache_manifest(
        args.train_hidden_states_path,
        target_model=args.target_model_path,
        num_hidden_layers=int(target_config.num_hidden_layers),
        hidden_size=int(target_config.hidden_size),
        required_layer_ids=required_layer_ids,
        max_length=int(args.max_length),
        verify_checksums=verify,
    )
    if dist.is_initialized():
        dist.barrier()
        if not verify:
            manifest = load_hidden_cache_manifest(
                args.train_hidden_states_path,
                target_model=args.target_model_path,
                num_hidden_layers=int(target_config.num_hidden_layers),
                hidden_size=int(target_config.hidden_size),
                required_layer_ids=required_layer_ids,
                max_length=int(args.max_length),
                verify_checksums=False,
            )
    dataset = RegenFullDataset(
        args.train_hidden_states_path,
        manifest,
        required_layer_ids=required_layer_ids,
        max_length=args.max_length,
        validation_mode=getattr(args, "hidden_cache_validation", "per-read"),
        audit_path=getattr(args, "hidden_cache_audit", None),
        selective_read=getattr(args, "hidden_cache_selective_read", True),
    )
    valid_indices = dataset.eligible_indices(
        block_size=args.block_size,
        complete_blocks=getattr(args, "complete_blocks", False),
    )
    process_group = get_dp_group()
    sampler = ShardAwareDistributedSampler(
        dataset,
        valid_indices,
        num_replicas=dist.get_world_size(process_group),
        rank=dist.get_rank(process_group),
        shuffle=True,
        seed=args.seed,
    )
    if args.dataloader_num_workers == 0:
        prefetch_factor = None
    else:
        prefetch_factor = int(getattr(args, "dataloader_prefetch_factor", 2))
        if prefetch_factor < 1:
            raise ValueError("dataloader-prefetch-factor must be positive")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.dataloader_num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
        persistent_workers=args.dataloader_num_workers > 0,
        collate_fn=RegenFullCollator(
            tokenizer.pad_token_id or 0,
            args.max_length if args.pad_to_max_length else None,
        ),
        drop_last=True,
    )
    if len(dataloader) == 0:
        raise ValueError(
            f"Training cache {args.train_hidden_states_path!r} has no full batches after filtering."
        )
    return dataloader


def training_data_identity(args) -> str:
    path = args.train_hidden_states_path or args.train_data_path
    identity = os.path.realpath(path)
    if args.train_hidden_states_path:
        manifest_path = os.path.join(identity, "manifest.json")
        digest = hashlib.sha256()
        with open(manifest_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        identity = f"{identity}#manifest_sha256={digest.hexdigest()}#sampler_order_version={ShardAwareDistributedSampler.ORDER_VERSION}"
    return identity


def required_hidden_layer_ids(draft_models) -> set[int]:
    required_layer_ids: set[int] = set()
    for draft in draft_models:
        required_layer_ids.update(set(draft.target_layer_ids))
    return required_layer_ids


def hidden_states_to_cuda(hidden_states):
    if isinstance(hidden_states, torch.Tensor):
        return hidden_states.cuda(non_blocking=True)
    if isinstance(hidden_states, dict):
        return {
            key: value.cuda(non_blocking=True) for (key, value) in hidden_states.items()
        }
    return tuple((value.cuda(non_blocking=True) for value in hidden_states))


def project_cached_target_logits(
    final_hidden_state: torch.Tensor,
    anchor_positions: torch.Tensor,
    block_size: int,
    lm_head: torch.nn.Module,
) -> torch.Tensor:
    """Project cached target features at supervised positions."""
    offsets = torch.arange(int(block_size) - 1, device=anchor_positions.device).view(
        1, 1, -1
    )
    positions = anchor_positions.unsqueeze(-1) + offsets
    if bool((positions >= final_hidden_state.size(1)).any()):
        raise ValueError("Cached final hidden states do not cover target positions.")
    expanded = final_hidden_state.unsqueeze(1).expand(
        -1, anchor_positions.size(1), -1, -1
    )
    selected = torch.gather(
        expanded,
        2,
        positions.unsqueeze(-1).expand(-1, -1, -1, final_hidden_state.size(-1)),
    )
    with torch.no_grad():
        return lm_head(selected)


def load_cached_target_data(
    data,
    *,
    anchor_positions: torch.Tensor,
    block_size: int,
    lm_head: torch.nn.Module,
    need_logits: bool,
):
    hidden_states = hidden_states_to_cuda(data["hidden_states"])
    target_logits = None
    if need_logits:
        final_layer_id = int(data["final_norm_layer_id"])
        final_hidden = data.get("final_hidden_state")
        final_hidden = (
            hidden_states[final_layer_id]
            if final_hidden is None
            else hidden_states_to_cuda(final_hidden)
        )
        target_logits = project_cached_target_logits(
            final_hidden, anchor_positions, block_size, lm_head
        )
    return (hidden_states, target_logits)


def select_tp_rank_batch(value, tp_rank: int):
    """Copy rank ``tp_rank``'s sample so full target tensors can be released."""
    if isinstance(value, dict):
        return {
            key: select_tp_rank_batch(item, tp_rank) for (key, item) in value.items()
        }
    if isinstance(value, tuple):
        return tuple((select_tp_rank_batch(item, tp_rank) for item in value))
    if isinstance(value, list):
        return [select_tp_rank_batch(item, tp_rank) for item in value]
    if not isinstance(value, torch.Tensor) or value.ndim == 0:
        raise TypeError(
            "TP draft batch values must be batch-first tensors or containers."
        )
    if not 0 <= int(tp_rank) < value.size(0):
        raise ValueError(
            f"TP rank {tp_rank} cannot select from batch size {value.size(0)}."
        )
    return value.narrow(0, int(tp_rank), 1).clone(memory_format=torch.contiguous_format)


def build_target_model(args, draft_models):
    target_config = load_text_model_config(
        args.target_model_path, trust_remote_code=args.trust_remote_code
    )
    source_model_type = getattr(
        target_config, "target_source_model_type", target_config.model_type
    )
    if (
        is_qwen35_model_type(source_model_type)
        and args.target_model_backend != "sglang"
    ):
        raise ValueError(
            "Qwen3.5 targets require --target-model-backend sglang with the pinned Transformers version; the draft remains a dense Qwen3-style model."
        )
    capture_layer_ids = required_hidden_layer_ids(draft_models)
    if getattr(args, "require_target_last_hidden", False):
        capture_layer_ids.add(int(target_config.num_hidden_layers) - 1)
    backend_kwargs = {}
    if args.target_model_backend == "sglang":
        if is_qwen35_model_type(source_model_type):
            args.sglang_attention_backend = "fa3"
        backend_kwargs = SGLangBackendArgs.from_args(args).to_kwargs()
        if source_model_type == "gemma4_unified_text":
            if args.disaggregate:
                raise ValueError(
                    "Gemma 4 native SGLang teacher currently requires collocated training"
                )
            backend_kwargs["capture_layer_ids"] = sorted(capture_layer_ids)
            backend_kwargs["context_length"] = (
                backend_kwargs.get("context_length") or args.max_length + 1
            )
            if backend_kwargs.get("attention_backend") not in (
                "triton",
                "fa4",
                "trtllm_mha",
                "ascend",
                "intel_xpu",
                "intel_amx",
                "aiter",
            ):
                backend_kwargs["attention_backend"] = "triton"
        if backend_kwargs["max_running_requests"] is None:
            backend_kwargs["max_running_requests"] = int(args.batch_size)
        if backend_kwargs["max_total_tokens"] is None:
            backend_kwargs["max_total_tokens"] = int(args.batch_size) * int(
                args.max_length
            )
        if getattr(args, "opd_data_mode", "teacher_forced") == "sampled_anchor":
            backend_kwargs["page_size"] = 1
            requests = int(args.batch_size) * (int(args.num_anchors) + 1)
            tokens = int(args.batch_size) * (
                int(args.max_length)
                + int(args.num_anchors) * max(0, int(args.block_size) - 2)
            )
            backend_kwargs["max_running_requests"] = max(
                requests, int(backend_kwargs["max_running_requests"])
            )
            backend_kwargs["max_total_tokens"] = max(
                tokens, int(backend_kwargs["max_total_tokens"])
            )
    target = get_training_target_model(
        pretrained_model_name_or_path=args.target_model_path,
        backend=args.target_model_backend,
        torch_dtype=torch.bfloat16,
        device="cuda" if args.target_model_backend == "hf" else None,
        trust_remote_code=args.trust_remote_code,
        **backend_kwargs,
    )
    target.set_capture_layers(sorted(capture_layer_ids))
    return target


def build_target_and_components(args, draft_models):
    """Build one target and bind its tokenizer/embedding/head consistently."""
    offline = (
        bool(args.train_hidden_states_path)
        and getattr(args, "opd_data_mode", "teacher_forced") != "sampled_anchor"
    )
    target = None if offline else build_target_model(args, draft_models)
    (tokenizer, components, mask_token_id) = resolve_tokenizer_and_components(
        args, draft_models, target=target, standalone_components=offline
    )
    return (target, tokenizer, components, mask_token_id)


def _select_mask_token_id(
    *,
    explicit_id: Optional[int],
    configured_ids: list[Optional[int]],
    tokenizer_mask_id: Optional[int],
    used_token_ids: set[int],
    vocab_size: int,
) -> tuple[int, str]:
    """Select a real embedding row without assuming a model-specific token ID."""
    vocab_size = int(vocab_size)
    if vocab_size <= 0:
        raise ValueError(f"Target vocabulary size must be positive, got {vocab_size}.")
    if explicit_id is not None:
        candidate = int(explicit_id)
        source = "explicit --mask-token-id"
    else:
        configured = {int(value) for value in configured_ids if value is not None}
        if len(configured) > 1:
            raise ValueError(
                f"Draft checkpoints disagree on mask_token_id: {sorted(configured)}. Pass --mask-token-id explicitly."
            )
        if configured:
            candidate = configured.pop()
            source = "draft checkpoint"
        elif tokenizer_mask_id is not None:
            candidate = int(tokenizer_mask_id)
            source = "tokenizer.mask_token_id"
        else:
            candidate = next(
                (
                    token_id
                    for token_id in range(vocab_size)
                    if token_id not in used_token_ids
                ),
                -1,
            )
            source = "first tokenizer-unused model vocabulary row"
    if not 0 <= candidate < vocab_size:
        if candidate < 0 and explicit_id is None:
            raise ValueError(
                "No tokenizer-unused row exists inside the target embedding vocabulary. Configure a model-provided mask token or pass --mask-token-id."
            )
        raise ValueError(
            f"MASK token id {candidate} selected from {source} is outside target vocabulary [0, {vocab_size})."
        )
    return (candidate, source)


def resolve_tokenizer_and_components(
    args, draft_models, target=None, *, standalone_components: bool = False
):
    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model_path, trust_remote_code=args.trust_remote_code
    )
    target_config = load_text_model_config(
        args.target_model_path, trust_remote_code=args.trust_remote_code
    )
    configured_mask_ids = []
    for draft in draft_models:
        method_config = getattr(draft.config, draft.method_config_key)
        if method_config:
            configured_mask_ids.append(method_config.get("mask_token_id"))
    needs_mask = any(
        (not getattr(draft, "uses_hidden_feedback", False) for draft in draft_models)
    )
    if needs_mask:
        (mask_token_id, mask_source) = _select_mask_token_id(
            explicit_id=args.mask_token_id,
            configured_ids=configured_mask_ids,
            tokenizer_mask_id=tokenizer.mask_token_id,
            used_token_ids={int(value) for value in tokenizer.get_vocab().values()},
            vocab_size=int(target_config.vocab_size),
        )
    else:
        (mask_token_id, mask_source) = (0, "unused by EAGLE3")
    args.mask_token_id = mask_token_id
    print_on_rank0(
        f"Resolved MASK token id {mask_token_id} from {mask_source}; target vocab_size={int(target_config.vocab_size)}."
    )
    if not standalone_components and hasattr(target, "build_shared_components"):
        components = target.build_shared_components(mask_token_id)
    elif args.target_model_backend == "hf" and (not standalone_components):
        from legospec.modeling.target.target_utils import FlattenedLMHead

        target_model = target.model.requires_grad_(False).eval()
        native_head = target_model.get_output_embeddings()
        shared_head = native_head
        if isinstance(native_head, torch.nn.Linear):
            with torch.device("meta"):
                shared_head = FlattenedLMHead(
                    native_head.in_features,
                    native_head.out_features,
                    bias=native_head.bias is not None,
                    softcap=getattr(target_config, "final_logit_softcapping", None),
                )
            shared_head.weight = native_head.weight
            shared_head.bias = native_head.bias
        components = SharedTargetEmbeddingsAndHead(
            target_model.get_input_embeddings(), shared_head
        )
        components.requires_grad_(False)
    elif args.target_model_backend == "sglang" and (not standalone_components):
        if target is None or not hasattr(target, "model_runner"):
            raise ValueError("SGLang target is required to reuse target components.")
        target_model = target.model_runner.model
        target_embedding = target_model.get_input_embeddings()
        target_lm_head = target_model.lm_head
        components = SharedTargetEmbeddingsAndHead(
            SGLangTPEmbeddingAdapter(target_embedding, get_tp_group(), mask_token_id),
            SGLangTPLMHeadAdapter(target_lm_head, get_tp_group()),
        )
        components.requires_grad_(False)
        print_on_rank0(
            "Reusing SGLang target-resident TP embedding and LM head; no independent full-vocabulary copies were loaded."
        )
    else:
        components = TargetEmbeddingsAndHead.from_pretrained(
            args.target_model_path,
            embed_key=args.embedding_key,
            lm_head_key=args.lm_head_key,
            device="cuda",
            trust_remote_code=args.trust_remote_code,
        )
        if not 0 <= mask_token_id < components.embed_tokens.num_embeddings:
            raise ValueError(
                f"Draft vocab_row MASK mode requires an existing target embedding row, but mask_token_id={mask_token_id} and target vocab size={components.embed_tokens.num_embeddings}. Pass --mask-token-id with an unused in-vocabulary row."
            )
    for draft in draft_models:
        if getattr(draft, "uses_hidden_feedback", False):
            continue
        draft.mask_token_id = mask_token_id
        draft.mask_embedding_mode = "vocab_row"
        getattr(draft.config, draft.method_config_key)["mask_token_id"] = mask_token_id
    return (tokenizer, components, mask_token_id)


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


class FixedCollator:
    def __init__(self, max_length: int, pad_token_id: int):
        self.max_length = int(max_length)
        self.pad_token_id = int(pad_token_id)

    def _pad(self, value, *, fill=0, dtype=None):
        value = torch.as_tensor(value).reshape(-1)[: self.max_length]
        output = torch.full((self.max_length,), fill, dtype=dtype or value.dtype)
        output[: value.numel()].copy_(value.to(output.dtype))
        return output

    def __call__(self, features):
        return {
            "input_ids": torch.stack(
                [
                    self._pad(row["input_ids"], fill=self.pad_token_id)
                    for row in features
                ]
            ),
            "attention_mask": torch.stack(
                [self._pad(row["attention_mask"], fill=0) for row in features]
            ),
            "loss_mask": torch.stack(
                [
                    self._pad(row["loss_mask"], fill=0, dtype=torch.float32)
                    for row in features
                ]
            ),
        }


def topology_identity(args, topology) -> dict:
    return {
        "execution_mode": "disaggregate",
        "target_ranks_per_node": int(args.target_ranks_per_node),
        "draft_ranks_per_node": int(args.draft_ranks_per_node),
        "target_tp_size": int(args.target_tp_size),
        "target_ep_size": int(args.sglang_ep_size),
        "node_batch_size": int(args.node_batch_size),
        "global_batch_size": int(args.node_batch_size) * int(topology.nnodes),
        "draft_world_size": len(topology.draft_global_ranks),
    }


def validate_resume_topology(state: Optional[dict], identity: dict) -> None:
    if state is None:
        return
    for key, current in identity.items():
        saved = state[key]
        if saved != current:
            raise ValueError(
                f"Disaggregated resume topology mismatch for {key}: checkpoint={saved!r}, current={current!r}"
            )


def _load_common_state(path: Optional[str], *, args=None) -> Optional[dict]:
    if not path:
        return None
    common = os.path.join(path, "training_state.pt")
    if not os.path.isfile(common):
        raise FileNotFoundError(f"No training_state.pt in {path!r}")
    state = torch.load(common, map_location="cpu", weights_only=False)
    validate_training_identity(state, args)
    return state


def _build_target_loader(args, tokenizer, topology):
    dataset = None
    if topology.rank == 0:
        dataset = _build_processed_dataset(
            args,
            tokenizer,
            train_data_path=args.train_data_path,
            cache_namespace="disaggregate",
            num_proc=args.build_dataset_num_proc,
        )
        dataset = dataset.filter(
            _has_valid_anchor_supervision,
            fn_kwargs={"block_size": int(args.block_size)},
        )
    dist.barrier()
    if topology.is_target_leader and topology.rank != 0:
        dataset = _build_processed_dataset(
            args,
            tokenizer,
            train_data_path=args.train_data_path,
            cache_namespace="disaggregate",
            num_proc=args.build_dataset_num_proc,
        )
        dataset = dataset.filter(
            _has_valid_anchor_supervision,
            fn_kwargs={"block_size": int(args.block_size)},
        )
    dataset_size = torch.tensor(
        [len(dataset) if topology.rank == 0 else 0], device="cuda", dtype=torch.long
    )
    dist.broadcast(dataset_size, src=0)
    producers_total = topology.nnodes * topology.target_replicas_per_node
    local_batch_size = int(args.node_batch_size) // topology.target_replicas_per_node
    steps_per_epoch = int(dataset_size.item()) // (producers_total * local_batch_size)
    if steps_per_epoch <= 0:
        raise ValueError("dataset is too small for one disaggregated global batch")
    dataloader = None
    if topology.is_target_leader:
        producer_rank = topology.node_rank * topology.target_replicas_per_node + int(
            topology.target_replica_local_rank
        )
        sampler = DistributedSampler(
            dataset,
            num_replicas=producers_total,
            rank=producer_rank,
            shuffle=True,
            drop_last=True,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=local_batch_size,
            sampler=sampler,
            num_workers=args.dataloader_num_workers,
            collate_fn=FixedCollator(
                args.max_length, tokenizer.pad_token_id or tokenizer.eos_token_id or 0
            ),
            drop_last=True,
            prefetch_factor=2 if args.dataloader_num_workers else None,
        )
    return (dataloader, steps_per_epoch)


def _broadcast_input(args, topology, data):
    shape = (int(args.target_batch_size), int(args.max_length))
    if topology.is_target_leader:
        values = (
            data["input_ids"].cuda(non_blocking=True),
            data["attention_mask"].cuda(non_blocking=True),
            data["loss_mask"].cuda(non_blocking=True),
        )
    else:
        values = (
            torch.empty(shape, dtype=torch.long, device="cuda"),
            torch.empty(shape, dtype=torch.long, device="cuda"),
            torch.empty(shape, dtype=torch.float32, device="cuda"),
        )
    for value in values:
        dist.broadcast(
            value,
            src=topology.target_tp_leader_global_rank,
            group=topology.target_tp_group,
        )
    return values


def build_training_session(args):
    """Prepare resources once; no training iteration or optimizer update here."""
    from legospec.distributed import init_disaggregated, init_distributed
    from legospec.distributed.transport import build_node_routes
    from legospec.training.state import TrainingSession
    from legospec.training.checkpoint import (
        load_distributed_training_state,
        load_training_state,
        reserve_output_dir,
        validate_session_resume,
    )
    from legospec.training.fsdp_layout import build_fsdp_layouts
    from legospec.training.logging import create_tracker
    from legospec.training.muon import muon_config_from_args
    from legospec.training.optimizer import (
        BF16Optimizer,
        optimizer_step_count,
        register_optimizer_parameters,
    )

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
        session.resume_state = _load_common_state(args.resume_from, args=args)
        validate_resume_topology(
            session.resume_state, topology_identity(args, topology)
        )
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
            from dataclasses import asdict

            if session.resume_state:
                for key, value in {
                    "phase_plan": [asdict(p) for p in session.phases],
                    "steps_per_epoch": session.steps_per_epoch,
                    "train_data_identity": training_data_identity(args),
                }.items():
                    if session.resume_state[key] != value:
                        raise ValueError(f"Resume mismatch for {key}")
            session.target = build_target_model(args, target_proxies(session))
            return session
        drafts = [session.draft]
        (_, session.components, _) = resolve_tokenizer_and_components(
            args, drafts, standalone_components=True
        )
        args.tracker_global_rank = session.coordinator_rank
    else:
        session.resume_state = load_training_state(args.resume_from, args=args)
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
    validate_session_resume(session)
    if session.resume_state is not None:
        if args.disaggregate:
            session.resume_state = load_distributed_training_state(
                args.resume_from, process_group=session.process_group
            )
        session.optimizer.load_state_dict(session.resume_state)
    session.tracker = create_tracker(args, args.output_dir)
    if session.dataloader is not None:
        session.dataloader.generator = torch.Generator().manual_seed(args.seed)
    return session


import copy
from legospec.distributed.transport import TargetFeaturePacket, TargetFeatureSpec
import json
from pathlib import Path
from legospec.training.batch import (
    PreparedLogicalBatch,
    cpu_copy,
    tensor_bytes,
)
from legospec.modeling.draft.composable.objective.composition import LossBranches
from legospec.modeling.draft.composable.schema import fields
from legospec.modeling.draft.composable.loss_config import (
    LOSS_CONTRACT_VERSION,
    SAMPLED_ANCHOR_OBJECTIVES,
    normalize_training_recipe,
)
from legospec.training.step import TrainingStep
from legospec.modeling.draft.composable import LegoSpecConfig, LegoSpecModel


def concatenate_target_hidden(
    hidden_states, target_layer_ids: list[int], *, num_target_layers=None
) -> torch.Tensor:
    if isinstance(hidden_states, dict):
        selected = [hidden_states[int(layer_id)] for layer_id in target_layer_ids]
    else:
        if num_target_layers is None:
            offset = 1 if len(hidden_states) > max(target_layer_ids) + 1 else 0
        elif len(hidden_states) in (int(num_target_layers), int(num_target_layers) + 1):
            offset = int(len(hidden_states) == int(num_target_layers) + 1)
        else:
            raise ValueError("Target hidden tuple length does not match target depth")
        selected = [
            hidden_states[int(layer_id) + offset] for layer_id in target_layer_ids
        ]
    return torch.cat(selected, dim=-1)


def final_target_hidden(hidden_states, num_target_layers: int) -> torch.Tensor:
    layer_id = int(num_target_layers) - 1
    if isinstance(hidden_states, dict):
        return hidden_states[layer_id]
    offset = 1 if len(hidden_states) == int(num_target_layers) + 1 else 0
    return hidden_states[layer_id + offset]


def teacher_target_hidden(target_output, num_target_layers: int) -> torch.Tensor:
    """Prefer the explicit post-norm teacher feature; accept legacy backends."""
    normalized = getattr(target_output, "final_hidden_state", None)
    if normalized is not None:
        return normalized
    return final_target_hidden(target_output.hidden_states, num_target_layers)


def build_feature_dataloader(args, tokenizer, draft, algorithm: str):
    required_layer_ids = required_hidden_layer_ids([draft])
    if args.require_target_last_hidden:
        required_layer_ids.add(int(draft.config.num_target_layers) - 1)
    return build_train_dataloader(
        args,
        tokenizer,
        train_data_path=args.train_data_path,
        cache_namespace=algorithm,
        num_proc=args.build_dataset_num_proc,
        required_layer_ids=required_layer_ids,
    )


def prepare_target_features(data, target_output, draft, require_last: bool):
    target_hidden_states = (
        hidden_states_to_cuda(data["hidden_states"])
        if target_output is None
        else hidden_states_to_cuda(target_output.hidden_states)
    )
    target_context_hidden = concatenate_target_hidden(
        target_hidden_states,
        list(draft.target_layer_ids),
        num_target_layers=int(draft.config.num_target_layers),
    )
    target_last_hidden = None
    if require_last:
        if target_output is not None:
            normalized = teacher_target_hidden(
                target_output, int(draft.config.num_target_layers)
            )
        else:
            normalized = data.get("final_hidden_state")
            if normalized is None:
                normalized = final_target_hidden(
                    target_hidden_states, int(draft.config.num_target_layers)
                )
        target_last_hidden = hidden_states_to_cuda(normalized)
    return (target_context_hidden, target_last_hidden)


def _packet_spec(args, topology, metadata):
    local_batch_size = (
        int(args.node_batch_size) // topology.target_replicas_per_node
        if topology.is_target
        else int(args.node_batch_size) // topology.draft_ranks_per_node
    )
    return TargetFeatureSpec(
        batch_size=local_batch_size,
        max_length=int(args.max_length),
        context_width=int(metadata["context_width"]),
        hidden_size=int(metadata["hidden_size"]),
        include_target_last_hidden=bool(metadata["include_target_last_hidden"]),
    )


def _target_packet(args, topology, target, metadata, values):
    (input_ids, attention_mask, loss_mask) = values
    target_output = target.generate_training_data(
        input_ids, attention_mask, loss_mask, return_logits=False
    )
    if not topology.is_target_leader:
        return None
    target_hidden_states = target_output.hidden_states
    target_context_hidden = concatenate_target_hidden(
        target_hidden_states,
        metadata["target_layer_ids"],
        num_target_layers=metadata["num_target_layers"],
    )
    target_last_hidden = (
        teacher_target_hidden(target_output, metadata["num_target_layers"])
        if metadata["include_target_last_hidden"]
        else None
    )
    return TargetFeaturePacket(
        input_ids=input_ids,
        loss_mask=loss_mask,
        target_context_hidden=target_context_hidden.to(torch.bfloat16),
        target_last_hidden=target_last_hidden.to(torch.bfloat16)
        if target_last_hidden is not None
        else None,
    )


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
    values = target.to_dict()
    target_depth = getattr(target, "num_target_layers", values["num_hidden_layers"])
    attention_keys = ("num_attention_heads", "num_key_value_heads", "head_dim")
    if values.get("model_type") == "gemma4_unified_text":
        reference_attention = {key: values[key] for key in attention_keys}
        values.pop("per_layer_config", None)
        values["rope_parameters"] = values["rope_parameters"]["full_attention"]
        values["hidden_act"] = values.pop("hidden_activation")
    else:
        reference_attention = {key: getattr(target, key) for key in attention_keys}
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


def enter_phase(session, phase):
    session.draft.train()


def detail_metrics(session):
    return {}


def log_metrics(session, reduced):
    if getattr(session.args, "opd_data_mode", "teacher_forced") == "sampled_anchor":
        if "rollout/accepted_length" not in reduced:
            raise RuntimeError(
                "Sampled anchor OPD requires actual verifier acceptance metrics"
            )
    required = {"hard_label/block_accepted_length", "expected_accept_len"}
    missing = required.difference(reduced)
    if missing:
        raise RuntimeError(
            "Composable training acceptance diagnostics missing: "
            + ", ".join(sorted(missing))
            + "; prepare target_last_hidden for metric collection"
        )
    return reduced


def training_metadata(session):
    metadata = {
        "loss_config": copy.deepcopy(
            session.draft.config.composable_config["objective"]
        ),
        "composable_recipe": copy.deepcopy(session.draft.config.composable_config),
        "loss_contract_version": LOSS_CONTRACT_VERSION,
        "objective_chunk_blocks": session.args.objective_chunk_blocks,
        "objective_gradient_checkpointing": getattr(
            session.args, "objective_gradient_checkpointing", True
        ),
        "num_anchors": session.args.num_anchors,
        "anchor_sampling": "sample_epoch",
        "anchor_sampling_version": "sample_epoch_v1",
        "anchor_seed": session.args.seed,
        "complete_blocks": getattr(session.args, "complete_blocks", False),
    }
    if session.draft.uses_hidden_feedback:
        metadata.update(
            supervision_mode=getattr(session.args, "supervision_mode", "sampled"),
            ttt_length=session.draft.ttt_length,
        )
    branches = LossBranches(session.draft.config.composable_config)
    objective_types = {o.recipe["type"] for o in branches.objectives.values()}
    if "dpal_renyi" in objective_types:
        import hashlib
        from legospec.modeling.draft.composable.objective import dpal

        metadata.update(
            dpal_reference_commit=dpal.REFERENCE_COMMIT,
            dpal_source_sha256=hashlib.sha256(
                Path(dpal.__file__).read_bytes()
            ).hexdigest(),
            dpal_distribution="full_vocabulary_temperature_1",
            dpal_credit="detached_overlap_prefix_suffix",
            dpal_interpretation="offline_teacher_forced_surrogate",
        )
    if any(
        (
            getattr(o, "needs_component_normalization", False)
            for o in branches.objectives.values()
        )
    ):
        metadata.update(
            opd_interpretation="offline_empirical_teacher_forced",
            opd_accepted_prefix="detached_greedy_match_to_fixed_labels",
            opd_reachability="min_accepted_plus_one_including_first_failure",
        )
    if getattr(session.args, "opd_data_mode", "teacher_forced") == "sampled_anchor":
        metadata.update(
            opd_data_mode="sampled_anchor",
            opd_interpretation="current_policy_sampled_blocks",
            opd_accepted_prefix="probabilistic_rejection_sampling",
            opd_temperature=1.0,
            opd_context_source="offline_hidden"
            if session.args.train_hidden_states_path
            else "online",
            opd_target_chunk_anchors=0,
        )
    return metadata


def checkpoint_metadata(model):
    return {
        "algorithm": "composable",
        "architecture": "LegoSpecModel",
        "block_size": int(model.block_size),
        "num_draft_layers": int(model.config.num_hidden_layers),
        "target_layer_ids": list(model.target_layer_ids),
        "num_target_layers": int(model.config.num_target_layers),
        "target_hidden_size": int(model.config.hidden_size),
        model.method_config_key: dict(getattr(model.config, model.method_config_key)),
    }


def phases(args):
    from dataclasses import replace

    from legospec.training.state import Phase

    phases = [Phase("composable", 1, int(args.num_epochs))]
    if args.max_steps is not None:
        phases[0] = replace(
            phases[0],
            max_batches=args.max_steps * args.accumulation_steps,
            schedule_full_epochs=True,
        )
    return phases


def initialize(session):
    args = session.args
    source = args.resume_from or args.init_from
    if (
        args.resume_from
        and (getattr(session, "resume_state", None) or {}).get("loss_contract_version")
        != LOSS_CONTRACT_VERSION
    ):
        raise ValueError(
            "Cannot resume a historical loss contract; use init_from with an explicitly migrated objective to start new training"
        )
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
        requested_width = getattr(args, "draft_intermediate_size", None)
        if (
            requested_width is not None
            and draft.config.intermediate_size != requested_width
        ):
            raise ValueError(
                f"Cannot resume with a different draft FFN width: checkpoint={draft.config.intermediate_size}, requested={requested_width}"
            )
        draft = draft.cuda()
    else:
        draft = build_model(args)
    draft.requires_grad_(True)
    if draft.uses_hidden_feedback and args.opd_data_mode != "teacher_forced":
        raise ValueError(
            "EAGLE3 requires teacher_forced TTT; sampled_anchor OPD is unsupported"
        )
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
    loss_branches = LossBranches(draft.config.composable_config)
    if getattr(args, "opd_data_mode", "teacher_forced") == "sampled_anchor":
        recipe = draft.config.composable_config
        if (
            recipe["proposal"]["type"] not in ("parallel", "rnn")
            or recipe["objective"]["type"] not in SAMPLED_ANCHOR_OBJECTIVES
        ):
            raise ValueError(
                "sampled_anchor requires Parallel/RNN with e2e_tv"
            )
        if float(draft.config.attention_dropout) != 0.0:
            raise ValueError("sampled_anchor requires zero attention dropout")
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
    if getattr(session.args, "opd_data_mode", "teacher_forced") == "sampled_anchor":
        if session.args.target_model_backend == "sglang":
            from legospec.modeling.target.sglang_anchor_opd import SGLangAnchorScorer

            session.anchor_scorer = SGLangAnchorScorer(session.target)
        else:
            from legospec.modeling.target.anchor_opd import Qwen3AnchorScorer

            session.anchor_scorer = Qwen3AnchorScorer(session.target.model)
    return TrainingStep(
        session.draft,
        session.components.lm_head,
        session.components.embed_tokens,
        session.args.mask_token_id,
        num_anchors=session.args.num_anchors,
        objective_chunk_blocks=session.args.objective_chunk_blocks,
        objective_gradient_checkpointing=session.args.objective_gradient_checkpointing,
        loss_process_group=session.process_group,
        opd_data_mode=getattr(session.args, "opd_data_mode", "teacher_forced"),
        complete_blocks=getattr(session.args, "complete_blocks", False),
        supervision_mode=getattr(session.args, "supervision_mode", "sampled"),
        training_mode=getattr(session.args, "training_mode", "legacy"),
    )


def build_loader(session, tokenizer):
    return build_feature_dataloader(
        session.args, tokenizer, session.draft, "composable"
    )


def packet_metadata(session):
    draft = session.draft
    metadata = {
        "algorithm": "composable",
        "target_layer_ids": list(draft.target_layer_ids),
        "num_target_layers": int(draft.config.num_target_layers),
        "hidden_size": int(draft.config.hidden_size),
        "context_width": len(draft.target_layer_ids) * int(draft.config.hidden_size),
        "block_size": int(draft.block_size),
        "include_target_last_hidden": bool(session.args.require_target_last_hidden),
    }
    recipe = session.draft.config.composable_config
    branches = LossBranches(recipe)
    if (
        getattr(session.args, "training_mode", "legacy") == "optimized"
        and (recipe["objective"].get("loss_decay_gamma") is None)
        and (not session.draft.uses_hidden_feedback)
        and (branches.is_selector or session.draft.config.indexer_config is not None)
        and (
            not any(
                (
                    getattr(o, "needs_component_normalization", False)
                    for o in branches.objectives.values()
                )
            )
        )
    ):
        metadata["streamed_window_recipe"] = copy.deepcopy(recipe)
        metadata["streamed_window_indexer"] = (
            session.draft.config.indexer_config is not None
        )
    return metadata


def target_proxies(session):
    from types import SimpleNamespace

    session.args.require_target_last_hidden = session.packet_metadata[
        "include_target_last_hidden"
    ]
    proxies = [
        SimpleNamespace(target_layer_ids=session.packet_metadata["target_layer_ids"])
    ]
    for proxy in proxies:
        proxy.method_config_key = "composable_config"
    return proxies


def packet_spec(session, phase):
    from legospec.distributed.transport import TrainingPacketSpec

    base = _packet_spec(session.args, session.topology, session.packet_metadata)
    return TrainingPacketSpec(
        base.batch_size,
        base.max_length,
        base.context_width,
        base.hidden_size,
        base.include_target_last_hidden,
        session.args.num_anchors
        if session.packet_metadata.get("streamed_window_recipe")
        else 0,
    )


def empty_packet(spec):
    from legospec.distributed.transport import TrainingBatchPacket

    return TrainingBatchPacket.empty(spec)


def produce_packet(session, phase, values, batch_id):
    from legospec.distributed.transport import TrainingBatchPacket

    packet = _target_packet(
        session.args, session.topology, session.target, session.packet_metadata, values
    )
    if packet is None:
        return None
    return TrainingBatchPacket(
        packet.input_ids,
        packet.loss_mask,
        packet.target_context_hidden,
        packet.target_last_hidden,
        values[1].bool(),
    )


def target_epoch_offset(phase):
    return 0


def compute_step(
    session,
    data,
    *,
    microbatch_start=None,
    microbatch_end=None,
    collect_detailed_metrics=False,
):
    if getattr(session.args, "opd_data_mode", "teacher_forced") == "sampled_anchor":
        if isinstance(data, PreparedLogicalBatch):
            values = data.materialize()
            record = values["sampled_opd_batch"].to("cuda")
            return session.fsdp_model(
                input_ids=values["input_ids"].cuda(non_blocking=True),
                loss_mask=values["loss_mask"].cuda(non_blocking=True),
                target_hidden_states=None,
                context_valid_mask=record.context_valid,
                sampled_opd_batch=record,
                opd_global_denominators=data.opd_component_denominators,
                opd_reference_samples=data.opd_reference_samples,
                collect_detailed_metrics=collect_detailed_metrics,
            )
        return session.fsdp_model(
            **_anchor_values(session, data),
            opd_scorer=session.anchor_scorer,
            collect_detailed_metrics=collect_detailed_metrics,
        )
    prepared = data if isinstance(data, PreparedLogicalBatch) else None
    data = prepared.materialize() if prepared is not None else data
    if prepared is not None and prepared.opd_inputs_prepared:
        values = {
            key: None
            if value is None
            else value[microbatch_start:microbatch_end].cuda(non_blocking=True)
            for (key, value) in data.items()
        }
    else:
        values = _step_values(session, data, microbatch_start, microbatch_end)
    layout = (
        None
        if prepared is None
        else prepared.layout.batch_slice(microbatch_start, microbatch_end)
    )
    if prepared is None:
        layout = anchor_layout(
            session,
            data,
            microbatch_start=microbatch_start,
            microbatch_end=microbatch_end,
        )
    execution_plan = None
    if (
        layout is not None
        and getattr(session.args, "training_mode", "legacy") == "optimized"
        and isinstance(data, dict)
        and (layout.anchor_positions.device.type == "cpu")
    ):
        from legospec.training.batch import prepare_packed_plan

        valid = data.get("attention_mask", data.get("context_valid_mask"))
        if valid is not None and valid.device.type == "cpu":
            if session.tp_draft_rank is not None and valid.size(
                0
            ) != layout.anchor_positions.size(0):
                valid = valid[session.tp_draft_rank : session.tp_draft_rank + 1]
            valid = valid[microbatch_start:microbatch_end]
            execution_plan = prepare_packed_plan(
                valid,
                layout,
                session.draft.block_size,
                query_len=session.draft.ttt_length
                if session.draft.uses_hidden_feedback
                else None,
            )
    if layout is not None:
        layout = layout.to(values["input_ids"].device)
    component = {}
    if prepared is not None and prepared.opd_metadata is not None:
        from legospec.training.checkpoint import restore_rng_state

        meta = prepared.opd_metadata[microbatch_start, microbatch_end]
        component = dict(
            opd_accepted_lengths={
                name: r.to(values["input_ids"].device)
                for (name, r) in meta.accepted_lengths.items()
            },
            opd_global_denominators=prepared.opd_component_denominators,
            opd_reference_samples=prepared.opd_reference_samples,
        )
        restore_rng_state(meta.rng_state)
    return session.fsdp_model(
        **values,
        layout=layout,
        execution_plan=execution_plan,
        window_denominators=getattr(session, "planned_window_denominators", None)
        if prepared is None
        else prepared.window_denominators,
        epoch_progress=session.state.phase_step / session.steps_per_epoch,
        collect_detailed_metrics=collect_detailed_metrics,
        **component,
    )


def prepare_target_batches(session, phase, batches):
    recipe = session.packet_metadata.get("streamed_window_recipe")
    if not recipe:
        yield from batches
        return
    from legospec.training.windows import plan_target_windows

    print(
        "Async CPU metadata prefetch active: one future token-only window.", flush=True
    )
    yield from plan_target_windows(
        batches,
        branches=LossBranches(recipe),
        block_size=session.args.block_size,
        num_anchors=session.args.num_anchors,
        seed=session.args.seed,
        accumulation_steps=session.args.accumulation_steps,
        steps_per_epoch=session.steps_per_epoch,
        indexer=session.packet_metadata["streamed_window_indexer"],
        asynchronous=True,
    )


def annotate_target_packet(session, phase, packet, data):
    if session.packet_metadata.get("streamed_window_recipe"):
        device = packet.input_ids.device

        def stage_metadata(value):
            if device.type == "cuda" and (not value.is_pinned()):
                value = value.pin_memory()
            return value.to(device, non_blocking=True)

        packet.anchor_positions = stage_metadata(data["_streamed_anchor_positions"])
        packet.block_keep_mask = stage_metadata(data["_streamed_block_keep_mask"])
        packet.window_counts = stage_metadata(data["_streamed_window_counts"])
    return packet


def prepare_optimizer_window(session, first, iterator, count):
    if getattr(session.args, "opd_data_mode", "teacher_forced") == "sampled_anchor":
        if session.draft.config.composable_config["objective"]["type"] == "e2e_tv":
            return None
        from legospec.training.windows import prepare_anchor_opd_window

        return prepare_anchor_opd_window(session, first, iterator, count)
    recipe = session.draft.config.composable_config
    indexer = session.draft.config.indexer_config
    branches = LossBranches(recipe)
    if any(
        (
            getattr(o, "needs_component_normalization", False)
            for o in branches.objectives.values()
        )
    ):
        from legospec.training.windows import prepare_opd_window

        return prepare_opd_window(session, first, iterator, count)
    if (
        session.topology is not None
        and getattr(first, "window_counts", None) is not None
    ):
        totals = first.window_counts.sum(0)
        if dist.is_initialized():
            dist.all_reduce(totals, group=session.process_group)
        for name in branches.objectives:
            if totals[1 if name == "final" else 2] <= 0:
                raise ValueError(
                    "Optimizer window has no supervised proposal positions"
                )
        session.planned_window_denominators = totals
        if not getattr(session, "_streamed_window_announced", False):
            print(
                "Streaming optimizer windows active: consume hidden packets directly from the receive ring; token-only lookahead, no hidden replay.",
                flush=True,
            )
            session._streamed_window_announced = True
        return None
    if (
        recipe.get("proposal", {}).get("type")
        not in ("selector", "residual_selector", "topk_selector")
        and indexer is None
    ):
        return None
    if isinstance(first, dict) and "_planned_window_counts" in first:
        totals = first["_planned_window_counts"]
        if session.tp_draft_rank is not None:
            totals = totals[session.tp_draft_rank : session.tp_draft_rank + 1]
        totals = totals.sum(0).cuda()
        if dist.is_initialized():
            dist.all_reduce(totals, group=session.process_group)
        for name in branches.objectives:
            if totals[1 if name == "final" else 2] <= 0:
                raise ValueError(
                    "Optimizer window has no supervised proposal positions"
                )
        session.planned_window_denominators = totals
        return None
    (prepared, used) = ([], 0)
    totals = torch.zeros(3, device="cuda", dtype=torch.float64)
    for index in range(count):
        raw = first if index == 0 else next(iterator)
        data = cpu_copy(raw)
        if session.topology is not None:
            (ids, mask) = (data.input_ids, data.loss_mask)
        else:
            (ids, mask) = (data["input_ids"], data["loss_mask"])
            if session.tp_draft_rank is not None:
                (ids, mask) = (
                    ids[session.tp_draft_rank : session.tp_draft_rank + 1],
                    mask[session.tp_draft_rank : session.tp_draft_rank + 1],
                )
        layout = anchor_layout(session, data)
        denominators = branches.native_denominators(layout)
        primary = denominators["final"].item()
        if indexer is None:
            auxiliary = denominators["base"].item()
        else:
            context_valid = (
                data.context_valid_mask
                if session.topology is not None
                else data["attention_mask"]
            )
            if session.tp_draft_rank is not None:
                context_valid = context_valid[
                    session.tp_draft_rank : session.tp_draft_rank + 1
                ]
            positions = torch.arange(ids.size(1))
            has_history = (
                context_valid[:, None].bool()
                & (positions[None, None] < layout.anchor_positions[..., None])
            ).any(-1)
            auxiliary = (
                (
                    layout.block_keep_mask
                    & (layout.prediction_mask > 0).any(-1)
                    & has_history
                )
                .sum()
                .item()
            )
        totals += totals.new_tensor([ids.size(0), primary, auxiliary])
        item = PreparedLogicalBatch(data, layout, totals)
        size = tensor_bytes(data)
        if used + size > session.args.window_buffer_mb * 1024**2:
            item.spill(Path(session.args.cache_dir) / "composable_windows")
        else:
            used += size
        prepared.append(item)
    if dist.is_initialized():
        dist.all_reduce(totals, group=session.process_group)
    invalid = (
        totals[1] <= 0
        if not branches.is_selector
        else any(
            (totals[1 if name == "final" else 2] <= 0 for name in branches.objectives)
        )
    )
    if invalid:
        for item in prepared:
            item.release()
        raise ValueError("Optimizer window has no supervised proposal positions")
    for item in prepared:
        item.window_denominators = totals
    return prepared


def release_batch(data):
    if isinstance(data, PreparedLogicalBatch):
        data.release()


def _step_values(session, data, microbatch_start=None, microbatch_end=None):
    (args, draft) = (session.args, session.draft)
    if session.topology is not None:
        input_ids = data.input_ids[microbatch_start:microbatch_end].cuda()
        attention_mask = data.context_valid_mask[microbatch_start:microbatch_end].cuda()
        loss_mask = data.loss_mask[microbatch_start:microbatch_end].cuda()
        target_hidden_states = data.target_context_hidden[
            microbatch_start:microbatch_end
        ].cuda()
        last = data.target_last_hidden
        last = None if last is None else last[microbatch_start:microbatch_end].cuda()
    else:
        from legospec.training.setup import select_tp_rank_batch

        input_ids = data["input_ids"].cuda(non_blocking=True)
        attention_mask = data["attention_mask"].cuda(non_blocking=True)
        loss_mask = data["loss_mask"].cuda(non_blocking=True)
        target_output = None
        if session.target is not None:
            target_inputs = (input_ids, attention_mask, loss_mask)
            if args.target_model_backend == "sglang":
                target_inputs = (
                    data["input_ids"],
                    data["attention_mask"],
                    data["loss_mask"],
                )
            target_output = session.target.generate_training_data(
                *target_inputs, return_logits=False
            )
        (target_hidden_states, last) = prepare_target_features(
            data, target_output, draft, args.require_target_last_hidden
        )
        if session.tp_draft_rank is not None:
            (input_ids, attention_mask, loss_mask, target_hidden_states, last) = [
                select_tp_rank_batch(value, session.tp_draft_rank)
                for value in (
                    input_ids,
                    attention_mask,
                    loss_mask,
                    target_hidden_states,
                    last,
                )
            ]
    return dict(
        input_ids=input_ids,
        target_hidden_states=target_hidden_states,
        loss_mask=loss_mask,
        target_last_hidden_states=last,
        context_valid_mask=attention_mask.bool(),
    )


def anchor_layout(session, data, *, microbatch_start=None, microbatch_end=None):
    if (
        session.topology is not None
        and getattr(data, "anchor_positions", None) is not None
    ):
        from legospec.training.batch import layout_from_anchors

        subset = slice(microbatch_start, microbatch_end)
        return layout_from_anchors(
            data.input_ids[subset],
            data.loss_mask[subset] * data.context_valid_mask[subset],
            data.anchor_positions[subset],
            data.block_keep_mask[subset],
            session.draft.block_size,
        )
    if isinstance(data, dict) and "_planned_layout" in data:
        layout = data["_planned_layout"]
        if session.tp_draft_rank is not None:
            layout = layout.batch_slice(
                session.tp_draft_rank, session.tp_draft_rank + 1
            )
        return layout.batch_slice(microbatch_start, microbatch_end)
    hidden_feedback = session.draft.uses_hidden_feedback
    supervision_mode = getattr(session.args, "supervision_mode", "sampled")
    from legospec.training.batch import prepare_sample_epoch_layout

    if session.topology is None:
        (ids, mask, valid) = (
            data[key] for key in ("input_ids", "loss_mask", "attention_mask")
        )
        indices = data.get("sample_index")
        epoch = int(data.get("_anchor_epoch", session.state.epoch))
        if session.tp_draft_rank is not None:
            start = session.tp_draft_rank
            (ids, mask, valid) = (
                value[start : start + 1] for value in (ids, mask, valid)
            )
            if indices is not None:
                indices = indices[start : start + 1]
    else:
        (ids, mask, valid) = (data.input_ids, data.loss_mask, data.context_valid_mask)
        indices = None
        epoch = session.state.epoch
    subset = slice(microbatch_start, microbatch_end)
    return prepare_sample_epoch_layout(
        ids[subset],
        mask[subset],
        valid[subset],
        sample_indices=None if indices is None else indices[subset],
        block_size=session.draft.block_size,
        num_anchors=session.args.num_anchors,
        seed=session.args.seed,
        epoch=epoch,
        supervision_mode=supervision_mode if hidden_feedback else "sampled",
        minimum_anchor=1 if hidden_feedback else 0,
    )


def _anchor_values(session, data):
    ids = data["input_ids"].cuda(non_blocking=True)
    hidden = last = None
    if session.args.train_hidden_states_path:
        (hidden, last) = prepare_target_features(data, None, session.draft, True)
    return dict(
        input_ids=ids,
        loss_mask=data["loss_mask"].cuda(non_blocking=True),
        target_hidden_states=hidden,
        target_last_hidden_states=last,
        context_valid_mask=data["attention_mask"].cuda(non_blocking=True).bool(),
        layout=anchor_layout(session, data),
    )
