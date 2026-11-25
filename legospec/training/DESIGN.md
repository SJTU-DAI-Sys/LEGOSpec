# Training

Training assembles models and data, schedules batches, obtains target features, computes gradients, updates parameters, logs results, and restores saved state.

The entrypoint is `scripts/composable/train.py`; `config.py` parses arguments, and `runtime/runner.py` coordinates the run using `runtime/setup.py`.

## Runtime

- `runtime/setup.py`: draft/target/shared components, FSDP, optimizer, and TrainingStep construction.
- `runtime/runner.py`: epochs, logical batches, microbatches, accumulation, updates, and saving.
- `runtime/state.py`: phases, cursors, and session state.
- `runtime/checkpoint.py`: weights and full training-state save/restore.

Full restoration includes optimizer, scheduler, FP32 master parameters, per-rank RNG, cursor, and pending plans. Weights-only exports support inference or initialization of a new run.

## Batch preparation

- `data/loader.py`: dataset/collator assembly, DP loaders, explicit sequence-parallel collator settings, and shard-aware distributed sampling.
- `data/batch.py`: anchor sampling, label/mask alignment, packing, and TP batch selection.
- `data/layout.py`: sample plans, packed layouts, and indices.
- `data/planning.py`: cost estimates, balanced microsteps, token windows, and target workload planning.
- `data/prefetch.py`: ordered CPU batch prefetch and optional single-batch CUDA H2D prefetch.

General tokenization and disk formats belong to `legospec/data`.

## Target supervision

- `target/setup.py`: capture requirements, online/cache selection, tokenizer, mask token, and shared components.
- `target/features.py`: layer selection, concatenation, device transfer, and teacher-feature selection.
- `target/streaming.py`: disaggregated target producers, packets, broadcasts, and scheduling.

Capture and model-specific normalization belong to `modeling/target`; communication primitives belong to `distributed`.

## Step execution

- `execution/step.py`: model, proposal, and objective execution within an FSDP forward.
- `execution/supervision.py`: alignment of supervision rows, blocks, and vocabulary projections.
- `execution/windows.py`: optimizer-window denominators and temporary batch lifetime.

Execution manages packed/chunked computation and normalization. Objective mathematics remain in `modeling/draft/objective`.

## Optimization

- `optim/optimizer.py`: parameter grouping, FP32 master parameters, AdamW/Muon integration, and gradient updates.
- `optim/muon.py`: Muon math, matrix exchange, and sharded updates.
- `optim/fsdp_layout.py`: FSDP layouts and matrix routing.
- `optim/scheduler.py`: warmup, cosine, delayed, and staged learning-rate schedules.

## Logging

- `logger/metrics.py`: accuracy and acceptance-related additive statistics.
- `logger/logging.py`: accumulation/reduction, log payloads, memory statistics, and tracker integration.

Logging consumes results without advancing the training cursor.

## H2D prefetch

Optimized offline training can enable `--h2d-prefetch`; it is disabled by default. A separate CUDA stream transfers one future batch, with events and record_stream controlling dependencies and storage lifetime. CPU token/mask/planning metadata remain available for layout construction. Prefetching does not commit the training cursor and requires additional device memory for the prefetched inputs.
