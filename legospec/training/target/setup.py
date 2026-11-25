"""Resolve training target requirements, capacity and tokenizer/mask policy."""

from typing import Optional
import torch
from transformers import AutoTokenizer
from legospec.args import SGLangBackendArgs
from legospec.distributed import print_on_rank0
from legospec.modeling.config_utils import load_text_model_config
from legospec.modeling.target.components import TargetEmbeddingsAndHead
from legospec.modeling.target.training_target_model import get_training_target_model


def required_hidden_layer_ids(draft_models) -> set[int]:
    required_layer_ids: set[int] = set()
    for draft in draft_models:
        required_layer_ids.update(set(draft.target_layer_ids))
    return required_layer_ids


def build_target_model(args, draft_models):
    target_config = load_text_model_config(
        args.target_model_path, trust_remote_code=args.trust_remote_code
    )
    capture_layer_ids = required_hidden_layer_ids(draft_models)
    if getattr(args, "require_target_last_hidden", False):
        capture_layer_ids.add(int(target_config.num_hidden_layers) - 1)
    backend_kwargs = {}
    if args.target_model_backend == "sglang":
        backend_kwargs = SGLangBackendArgs.from_args(args).to_kwargs()
        backend_kwargs["dist_timeout"] = int(args.dist_timeout) * 60
        backend_kwargs["context_length"] = (
            backend_kwargs.get("context_length") or args.max_length + 1
        )
        if backend_kwargs["max_running_requests"] is None:
            backend_kwargs["max_running_requests"] = int(args.batch_size)
        if backend_kwargs["max_total_tokens"] is None:
            backend_kwargs["max_total_tokens"] = int(args.batch_size) * int(
                args.max_length
            )
    target = get_training_target_model(
        pretrained_model_name_or_path=args.target_model_path,
        backend=args.target_model_backend,
        capture_layer_ids=sorted(capture_layer_ids),
        torch_dtype=torch.bfloat16,
        device="cuda" if args.target_model_backend == "hf" else None,
        trust_remote_code=args.trust_remote_code,
        **backend_kwargs,
    )
    return target


def build_target_and_components(args, draft_models):
    """Build one target and bind its tokenizer/embedding/head consistently."""
    offline = bool(args.train_hidden_states_path)
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
    if not standalone_components:
        components = target.build_shared_components(mask_token_id)
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
