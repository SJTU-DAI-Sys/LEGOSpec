"""Export models and save/restore complete rank-local training state."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType

from legospec.modeling.draft.composable.objective.composition import LossBranches


def capture_rng_state():
    import random

    import numpy as np

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state):
    import random

    import numpy as np

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def session_metadata(session):
    from dataclasses import asdict

    state = session.state
    phase = session.phases[min(state.phase_index, len(session.phases) - 1)]
    metadata = {
        "cursor": asdict(state),
        "training_stage": phase.name,
        "phase_plan": [asdict(phase) for phase in session.phases],
        "steps_per_epoch": session.steps_per_epoch,
        "optimizer_step": state.optimizer_step,
        "scheduler_total_steps": session.total_steps,
        "scheduler_learning_rate": float(session.args.learning_rate),
        "scheduler_warmup_ratio": float(session.args.warmup_ratio),
        "train_data_mode": (
            "regen_full" if session.args.train_hidden_states_path else "online"
        ),
        **training_metadata(session),
    }
    if getattr(session.args, "use_muon", False):
        from legospec.training.optim.muon import muon_config_from_args

        metadata["muon_config"] = muon_config_from_args(session.args)
        metadata["scheduler_learning_rate"] = metadata["muon_config"]["adamw_lr"]
    if getattr(session.args, "training_mode", "legacy") == "optimized":
        planner = getattr(session, "training_planner", None)
        metadata["scheduler_plan"] = (
            planner.state_dict() if planner is not None else None
        )
    return metadata


def save_session(session, name):
    return save_checkpoint(
        output_dir=session.args.output_dir,
        name=name,
        fsdp_model=session.fsdp_model,
        draft_model=session.draft,
        optimizer=session.optimizer,
        metadata=session_metadata(session),
        training_yaml=session.training_yaml,
        process_group=session.process_group,
        coordinator_global_rank=session.coordinator_rank,
    )


def ranked_training_state_path(checkpoint_dir: str, rank: int) -> str:
    """Return the training-state filename owned by one global rank."""
    return os.path.join(checkpoint_dir, f"training_state_rank_{rank:05d}.pt")


def _atomic_torch_save(state: dict[str, Any], path: str) -> None:
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def save_distributed_training_state(
    checkpoint_dir: str,
    state: dict[str, Any],
    *,
    rank: int | None = None,
    process_group: dist.ProcessGroup | None = None,
) -> None:
    """Save each rank's complete state, then publish common metadata."""
    if rank is None:
        rank = dist.get_rank(process_group) if dist.is_initialized() else 0
    os.makedirs(checkpoint_dir, exist_ok=True)
    _atomic_torch_save(state, ranked_training_state_path(checkpoint_dir, rank))
    if dist.is_initialized():
        dist.barrier(group=process_group)
    if rank == 0:
        common = {
            key: value
            for key, value in state.items()
            if key not in {"optimizer_state_dict", "fp32_master_params", "rng_state"}
        }
        _atomic_torch_save(common, os.path.join(checkpoint_dir, "training_state.pt"))
    if dist.is_initialized():
        dist.barrier(group=process_group)


def load_distributed_training_state(
    checkpoint_dir: str,
    *,
    map_location: Any = "cpu",
    rank: int | None = None,
    process_group: dist.ProcessGroup | None = None,
) -> dict[str, Any]:
    """Load this rank's saved training state directly."""
    if rank is None:
        rank = dist.get_rank(process_group) if dist.is_initialized() else 0
    return torch.load(
        ranked_training_state_path(checkpoint_dir, rank),
        map_location=map_location,
        weights_only=False,
    )


def export_draft(model, save_dir, *, state_dict=None):
    """Export a self-contained model without importing the training package."""
    from legospec.modeling.draft.composable.export import export_composable

    return export_composable(model, save_dir, state_dict=state_dict)


def save_checkpoint(
    *,
    output_dir: str,
    name: str,
    fsdp_model: FSDP,
    draft_model,
    optimizer,
    metadata: dict,
    training_yaml: str | None = None,
    process_group=None,
    coordinator_global_rank: int = 0,
) -> str:

    save_dir = os.path.join(output_dir, name)
    group_rank = dist.get_rank(process_group)
    if group_rank == 0:
        os.makedirs(save_dir, exist_ok=True)
    dist.barrier(group=process_group)
    with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT):
        full_state = fsdp_model.state_dict()
        draft_state = {
            key.split("draft_model.", 1)[1]: value
            for key, value in full_state.items()
            if "draft_model." in key
        }
        model_metadata = checkpoint_metadata(draft_model)
        save_distributed_training_state(
            save_dir,
            {
                **metadata,
                **model_metadata,
                **optimizer.state_dict(),
                "rng_state": capture_rng_state(),
            },
            process_group=process_group,
        )
        if group_rank == 0:
            export_draft(draft_model, save_dir, state_dict=draft_state)
            if training_yaml is not None:
                Path(save_dir, "training_config.yaml").write_bytes(
                    training_yaml.encode("utf-8")
                )
    dist.barrier(group=process_group)
    if dist.get_rank() == coordinator_global_rank:
        print(f"Saved checkpoint to {save_dir}", flush=True)
    return save_dir


def load_training_state(checkpoint_dir: Optional[str]) -> Optional[dict]:
    if checkpoint_dir is None:
        return None
    return load_distributed_training_state(checkpoint_dir, map_location="cpu")


def reserve_output_dir(output_dir: str) -> str:
    """Atomically reserve a non-existing output directory on rank zero."""

    output_dir = os.path.normpath(os.path.expanduser(output_dir))
    selected = None
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    if rank == 0:
        suffix = 0
        while selected is None:
            candidate = output_dir if suffix == 0 else f"{output_dir}_{suffix}"
            try:
                os.makedirs(candidate, exist_ok=False)
            except FileExistsError:
                suffix += 1
            else:
                selected = candidate
    if dist.is_available() and dist.is_initialized():
        payload = [selected]
        dist.broadcast_object_list(payload, src=0)
        selected = payload[0]
        dist.barrier()
    assert selected is not None
    if rank == 0:
        print(f"Training output directory: {selected}", flush=True)
    return selected


def _load_common_state(path: Optional[str]) -> Optional[dict]:
    if path is None:
        return None
    return torch.load(
        os.path.join(path, "training_state.pt"), map_location="cpu", weights_only=False
    )


def training_metadata(session):
    metadata = {
        "loss_config": copy.deepcopy(
            session.draft.config.composable_config["objective"]
        ),
        "composable_recipe": copy.deepcopy(session.draft.config.composable_config),
        "objective_chunk_blocks": session.args.objective_chunk_blocks,
        "objective_gradient_checkpointing": getattr(
            session.args, "objective_gradient_checkpointing", True
        ),
        "num_anchors": session.args.num_anchors,
        "anchor_sampling": "sample_epoch",
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
