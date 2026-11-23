"""Produce target features and stream training packets to draft ranks."""

from __future__ import annotations

import copy

import torch
import torch.distributed as dist

import legospec.training.target.streaming as target_streaming
from legospec.distributed.transport import TargetFeaturePacket, TargetFeatureSpec
from legospec.modeling.draft.objective.composition import LossBranches
from legospec.training.target.features import select_target_features


def run_target(session):
    """Producer lifecycle with optional token-only optimizer-window planning."""

    from legospec.distributed.transport import NodePacketTransport

    args, topology = session.args, session.topology
    transport = (
        NodePacketTransport(
            topology=topology, routes=session.routes, profile=args.profile
        )
        if topology.is_target_leader
        else None
    )
    start_id = (
        session.resume_state["cursor"]["global_step"] if session.resume_state else 0
    )
    batch_id = 0
    for phase in session.phases:
        spec = packet_spec(session, phase)
        batch_id, bridge_ms = run_target_phase(
            args=args,
            topology=topology,
            dataloader=session.dataloader,
            steps=session.steps_per_epoch,
            epochs=phase.epochs,
            epoch_offset=target_epoch_offset(phase),
            start_id=start_id,
            batch_id=batch_id,
            limit=phase.batch_count(session.steps_per_epoch),
            transport=transport,
            phase_id=phase.phase_id,
            schema_id=spec.schema_id,
            broadcast=lambda data: _broadcast_input(args, topology, data),
            produce=lambda values, current: produce_packet(
                session, phase, values, current
            ),
            replay_skipped=True,
            prepare_batches=(
                lambda batches: prepare_target_batches(session, phase, batches)
            )
            if hasattr(target_streaming, "prepare_target_batches")
            else None,
            annotate_packet=(
                lambda packet, data: annotate_target_packet(
                    session, phase, packet, data
                )
            )
            if hasattr(target_streaming, "annotate_target_packet")
            else None,
        )
        if args.profile and topology.is_target_leader:
            print(
                f"disagg target bridge phase={phase.name} ms={bridge_ms:.3f}",
                flush=True,
            )


def run_target_phase(
    *,
    args,
    topology,
    dataloader,
    steps,
    epochs,
    epoch_offset,
    start_id,
    batch_id,
    limit,
    transport,
    phase_id,
    schema_id,
    broadcast,
    produce,
    replay_skipped=False,
    prepare_batches=None,
    annotate_packet=None,
):
    """Execute target prefill and drain one packet schema before returning."""
    handles = [None] * int(args.pipeline_depth)
    wait_ms = 0.0

    def source_batches():
        produced = 0
        for epoch in range(epochs):
            if topology.is_target_leader:
                dataloader.sampler.set_epoch(epoch_offset + epoch)
                iterator = iter(dataloader)
            for _ in range(steps):
                if produced >= limit:
                    return
                yield next(iterator) if topology.is_target_leader else None
                produced += 1

    batches = source_batches()
    if topology.is_target_leader and prepare_batches is not None:
        batches = prepare_batches(batches)
    for data in batches:
        slot = batch_id % len(handles)
        if batch_id >= start_id or replay_skipped:
            if topology.is_target_leader and handles[slot] is not None:
                wait_ms += transport.wait_send(handles[slot])
            packet = produce(broadcast(data), batch_id)
            if batch_id >= start_id and topology.is_target_leader:
                if annotate_packet is not None:
                    packet = annotate_packet(packet, data)
                handles[slot] = transport.send(
                    packet,
                    batch_id=batch_id,
                    phase_id=phase_id,
                    schema_id=schema_id,
                )
        batch_id += 1
    if topology.is_target_leader:
        for handle in handles:
            if handle is not None:
                wait_ms += transport.wait_send(handle)
    return batch_id, wait_ms


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
    target_context_hidden, target_last_hidden = select_target_features(
        target_output.hidden_states,
        target_output.final_hidden_state,
        metadata["target_layer_ids"],
        metadata["include_target_last_hidden"],
    )
    return TargetFeaturePacket(
        input_ids=input_ids,
        loss_mask=loss_mask,
        target_context_hidden=target_context_hidden.to(torch.bfloat16),
        target_last_hidden=target_last_hidden.to(torch.bfloat16)
        if target_last_hidden is not None
        else None,
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


def prepare_target_batches(session, phase, batches):
    recipe = session.packet_metadata.get("streamed_window_recipe")
    if not recipe:
        yield from batches
        return
    from legospec.training.data.planning import plan_target_windows

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
