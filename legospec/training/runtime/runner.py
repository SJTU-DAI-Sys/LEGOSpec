"""Own training iteration, backward/update timing and resource teardown."""

from __future__ import annotations

import os
from pathlib import Path

import legospec.training.execution.step as execution_step
import legospec.training.execution.windows as execution_windows
import legospec.training.logger.logging as logger_logging
import legospec.training.target.streaming as target_streaming
from legospec.training.target.streaming import run_target


def configure_rank_caches():
    root = Path(__file__).resolve().parents[3]
    rank = os.environ.get("RANK", "0")
    for name, directory in (
        ("TVM_FFI_CACHE_DIR", "tvm-ffi"),
        ("TORCHINDUCTOR_CACHE_DIR", "torchinductor"),
        ("TRITON_CACHE_DIR", "triton"),
    ):
        value = (
            os.environ.get(name.removesuffix("_DIR") + "_ROOT")
            or os.environ.get(name)
            or str(root / ".cache" / directory)
        )
        os.environ[name] = str(Path(value) / f"rank_{rank}")


def epoch_checkpoint_due(state, accumulation_steps):
    """First optimizer boundary at/after an epoch end, including odd loader lengths."""
    return (
        state.epoch > 0
        and state.next_batch < accumulation_steps
        and state.phase_step % accumulation_steps == 0
    )


class OptimizerWindow:
    """Sum loss terms, normalize FSDP gradients once, and aggregate telemetry."""

    def __init__(self, optimizer, accumulation_steps, process_group=None):
        from legospec.training.logger.logging import TrainingMetricAccumulator

        self.optimizer = optimizer
        self.accumulation_steps = int(accumulation_steps)
        if self.accumulation_steps <= 0:
            raise ValueError("accumulation_steps must be positive")
        self.process_group = process_group
        self.metrics = TrainingMetricAccumulator()
        self.micro_steps = 0
        self.denominator = None

    def backward(self, output):
        self.metrics.update(output)
        self.backward_terms(output.loss_terms.numerator, output.loss_terms.denominator)

    def backward_terms(self, numerator, denominator):
        (numerator / self.accumulation_steps).backward()
        value = denominator.detach()
        self.denominator = (
            value.clone() if self.denominator is None else self.denominator + value
        )
        self.micro_steps += 1

    def step(self):
        from legospec.training.optim.optimizer import normalize_accumulated_gradients

        if not self.micro_steps:
            raise RuntimeError("Cannot step an empty accumulation window")
        normalize_accumulated_gradients(
            self.optimizer,
            self.denominator,
            self.accumulation_steps,
            self.process_group,
        )
        norm = self.optimizer.step()
        self.micro_steps = 0
        self.denominator = None
        return norm


def iter_epoch(dataloader, epoch, *, start_batch=0, description=None):
    """Common deterministic epoch/cursor handling, also usable by stage runners."""
    import torch.distributed as dist
    from tqdm import tqdm

    dataloader.sampler.set_epoch(epoch)
    show_progress = not dist.is_initialized() or dist.get_rank() == 0
    iterator = tqdm(dataloader, desc=description) if show_progress else dataloader
    for index, batch in enumerate(iterator):
        if index >= start_batch:
            yield index, batch


def iter_logical_batches(session, phase):
    """Yield one complete logical batch at a time, keeping receive slots alive."""
    depth = int(getattr(session.args, "cpu_batch_prefetch", 0))
    if depth < 0:
        raise ValueError("cpu-batch-prefetch must be nonnegative")
    from legospec.training.data.prefetch import (
        cuda_prefetched_batches,
        ordered_cpu_batches,
    )

    batches = _iter_logical_batches(session, phase)
    if depth and session.topology is None and session.args.train_hidden_states_path:
        batches = ordered_cpu_batches(batches, depth)
    if getattr(session.args, "h2d_prefetch", False):
        batches = cuda_prefetched_batches(
            batches, session.draft.target_layer_ids, session.args.require_target_last_hidden
        )
    yield from batches


def _iter_logical_batches(session, phase):
    if (
        getattr(session.args, "training_mode", "legacy") == "optimized"
        and session.topology is None
    ):
        from legospec.training.data.planning import planned_batches

        yield from planned_batches(session, phase)
        return
    if session.topology is not None:
        from legospec.distributed.transport import NodePacketTransport, PacketReceiver

        transport = NodePacketTransport(
            topology=session.topology,
            routes=session.routes,
            profile=session.args.profile,
        )
        spec = target_streaming.packet_spec(session, phase)
        slots = [
            target_streaming.empty_packet(spec)
            for _ in range(session.args.pipeline_depth)
        ]
        start = session.state.global_step
        stop = (
            start
            - session.state.phase_step
            + phase.batch_count(session.steps_per_epoch)
        )
        packets = PacketReceiver(
            transport,
            slots,
            start=start,
            stop=stop,
            phase_id=phase.phase_id,
            schema_id=spec.schema_id,
        )
        for _, packet in packets:
            yield packet
        if session.args.profile:
            print(
                f"disagg draft bridge phase={phase.name} ms={packets.wait_ms:.3f}",
                flush=True,
            )
        return
    state = session.state
    initial_epoch, initial_batch = int(state.epoch), int(state.next_batch)
    remaining = phase.batch_count(session.steps_per_epoch) - int(state.phase_step)
    produced = 0
    for epoch in range(initial_epoch, phase.epochs):
        start = initial_batch if epoch == initial_epoch else 0
        for _, data in iter_epoch(
            session.dataloader,
            epoch,
            start_batch=start,
            description=f"{phase.name} epoch {epoch}",
        ):
            if produced >= remaining:
                return
            if isinstance(data, dict):
                # Stamp at input consumption: an accumulation window may cross
                # an epoch before the training cursor itself has advanced.
                data["_anchor_epoch"] = epoch
            produced += 1
            yield data


def iter_microbatch_slices(session):
    """Yield sample-axis (start, exclusive end) bounds; (None, None) uses the full batch."""
    if session.topology is None:
        yield None, None
    else:
        local_batch_size = (
            session.args.node_batch_size // session.args.draft_ranks_per_node
        )
        micro_batch_size = session.args.draft_micro_batch_size
        for microbatch_start in range(0, local_batch_size, micro_batch_size):
            yield microbatch_start, microbatch_start + micro_batch_size


def log_window(session, window, grad_norm):
    import torch.distributed as dist

    from legospec.training.logger.logging import build_training_log_payload

    reduced_metrics = logger_logging.log_metrics(
        session, window.metrics.reduce(session.process_group)
    )
    details = {
        "optimizer_step": session.state.optimizer_step,
        "micro_step": session.state.global_step,
        "grad_norm": grad_norm,
        **logger_logging.detail_metrics(session),
    }
    prefix = "composable"
    payload = build_training_log_payload(
        reduced_metrics,
        session.optimizer.get_learning_rate(),
        detail_metrics=details,
        detail_prefix=prefix,
    )
    payload["train/optimizer_step"] = session.state.optimizer_step
    if getattr(session.optimizer, "use_muon", False):
        payload.update(
            {
                f"train/{name}_learning_rate": value
                for name, value in session.optimizer.get_learning_rates().items()
            }
        )
    session.tracker.log(payload, step=session.state.optimizer_step)
    if not dist.is_initialized() or dist.get_rank() == session.coordinator_rank:
        print(
            f"{prefix} optimizer_step={session.state.optimizer_step} "
            f"micro_step={session.state.global_step} "
            f"loss={float(reduced_metrics['objective/loss']):.4f}",
            flush=True,
        )


def run_draft(session, *, batch_source=iter_logical_batches, save=None, log=log_window):
    """The sole training loop, also exercised directly by CPU runner tests.

    ``batch_source`` yields logical batches, not forwards. All trainable work is
    performed by the recipe; this loop alone owns backward and update timing.
    """
    import torch

    from legospec.training.logger.logging import log_cuda_peak
    from legospec.training.runtime.checkpoint import restore_rng_state, save_session

    save = save_session if save is None else save
    args, state = session.args, session.state
    window = OptimizerWindow(
        session.optimizer, args.accumulation_steps, session.process_group
    )
    # Restore RNG state from checkpoint
    pending_rng = (
        session.resume_state["rng_state"] if session.resume_state is not None else None
    )
    accumulated = 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Perform a model parameter update after each gradient accumulation step.
    def update(*, tail=False):
        nonlocal accumulated
        grad_norm = window.step()
        state.optimizer_step += 1
        accumulated = 0
        if (
            tail
            or state.optimizer_step == 1
            or state.optimizer_step % args.log_interval == 0
            or (
                session.phase.log_at_schedule_end
                and state.optimizer_step == session.total_steps
            )
        ):
            log(session, window, grad_norm)
        window.metrics.reset()

    while state.phase_index < len(session.phases):
        cur_phase = session.phase
        phase_metrics = enter_phase(session, cur_phase)
        if phase_metrics:
            session.tracker.log(phase_metrics, step=state.optimizer_step)
        remaining_batches = (
            cur_phase.batch_count(session.steps_per_epoch) - state.phase_step
        )
        batch_iterator = iter(batch_source(session, cur_phase))
        for batch_index in range(remaining_batches):
            try:
                data = next(batch_iterator)
            except StopIteration as exc:
                raise RuntimeError(
                    f"{cur_phase.name} input ended before the declared phase length"
                ) from exc
            # Skip/rebuild the input iterator first, then restore anchor/dropout RNG.
            if pending_rng is not None:
                restore_rng_state(pending_rng)
                pending_rng = None
            prepare_window = getattr(
                execution_windows, "prepare_optimizer_window", None
            )
            if accumulated == 0 and prepare_window is not None:
                prepared = prepare_window(
                    session,
                    data,
                    batch_iterator,
                    min(args.accumulation_steps, remaining_batches - batch_index),
                )
                if prepared is not None:
                    from itertools import chain

                    data = prepared[0]
                    batch_iterator = iter(chain(prepared[1:], batch_iterator))
            epoch = state.epoch
            collect_detailed_metrics = (
                state.optimizer_step == 0
                or (state.optimizer_step + 1) % args.log_interval == 0
                # Every microbatch in a final partial window is logged by
                # update(tail=True), even outside the regular interval.
                or (
                    accumulated
                    + cur_phase.batch_count(session.steps_per_epoch)
                    - state.phase_step
                )
                < args.accumulation_steps
                or (
                    cur_phase.log_at_schedule_end
                    and state.optimizer_step + 1 == session.total_steps
                )
            )
            for start, end in iter_microbatch_slices(session):
                output = execution_step.compute_step(
                    session,
                    data,
                    microbatch_start=start,
                    microbatch_end=end,
                    collect_detailed_metrics=collect_detailed_metrics,
                )
                window.backward(output)
                del output
            release_batch = getattr(execution_windows, "release_batch", None)
            if release_batch is not None:
                release_batch(data)
            state.phase_step += 1
            state.global_step += 1
            state.epoch, state.next_batch = divmod(
                state.phase_step, session.steps_per_epoch
            )
            accumulated += 1
            planner = getattr(session, "training_planner", None)
            if planner is not None:
                planner.commit(state.phase_step)
            if accumulated == args.accumulation_steps:
                update()
            epoch_boundary = getattr(
                args, "save_every_epoch", False
            ) and epoch_checkpoint_due(state, args.accumulation_steps)
            if accumulated == 0 and (
                state.optimizer_step % args.save_interval == 0 or epoch_boundary
            ):
                suffix = (
                    f"step_{state.global_step}"
                    if session.topology is not None
                    else f"epoch_{epoch}_step_{state.phase_step}"
                )
                save(session, cur_phase.checkpoint_prefix + suffix)
        # Resume the source once to release its last receive slot and collect profile data.
        try:
            next(batch_iterator)
        except StopIteration:
            pass
        else:
            raise RuntimeError(
                f"{cur_phase.name} input exceeded the declared phase length"
            )
        if accumulated:
            update(tail=True)
        if session.topology is None:
            prefix = "composable"
            memory = log_cuda_peak(cur_phase.name)
            session.tracker.log(
                {
                    f"detail/{prefix}/cuda_peak_{key}": value
                    for key, value in memory.items()
                },
                step=state.optimizer_step,
            )
        # Boundary checkpoints point to the NEXT phase, not a finished old phase.
        state.phase_index += 1
        state.epoch = state.next_batch = state.phase_step = 0
        session.training_planner = None
        if getattr(args, "final_save", True):
            save(session, cur_phase.final_checkpoint)


def run_training(args):
    import logging
    import traceback

    import torch.distributed as dist
    from accelerate.utils import set_seed

    from legospec.distributed import destroy_distributed
    from legospec.training.runtime.setup import build_training_session

    logging.basicConfig(level=logging.INFO)
    if args.disaggregate and os.environ.get("LEGOSPEC_DISAGG_DEBUG") == "1":
        import faulthandler
        import signal

        faulthandler.register(signal.SIGUSR1, all_threads=True)
    set_seed(args.seed)
    session = None
    completed = False
    try:
        session = build_training_session(args)
        if session.topology is not None and session.topology.is_target:
            run_target(session)
        else:
            run_draft(session)
        dist.barrier()
        completed = True
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if session is not None and session.tracker is not None:
            session.tracker.close()
        # On failure let torch elastic terminate blocked peers; collective teardown
        # could hide the original error until the distributed timeout.
        if completed:
            destroy_distributed()


def enter_phase(session, phase):
    session.draft.train()
