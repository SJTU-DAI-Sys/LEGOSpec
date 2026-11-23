"""Build online JSON and cached-hidden training loaders and collators."""

from __future__ import annotations

import hashlib
import os
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from legospec.data import (
    RegenFullCollator,
    RegenFullDataset,
    ShardAwareDistributedSampler,
    build_training_dataset,
    load_hidden_cache_manifest,
    prepare_dp_dataloaders,
)
from legospec.data.json_training import load_training_json
from legospec.distributed import get_dp_group
from legospec.modeling.config_utils import load_text_model_config
from legospec.training.target.setup import required_hidden_layer_ids


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


def build_loader(session, tokenizer):
    return build_feature_dataloader(
        session.args, tokenizer, session.draft, "composable"
    )
