# Training execution

Start training with a launcher YAML:

```bash
bash scripts/composable/run_training_composable.sh /path/to/train.yaml
```

See the [main README](../README.md) for a complete example. The YAML launcher forwards `arguments` to the shared training CLI.

## Execution profiles

`training_mode: legacy` uses padded execution and full raw-cache reads by default.

`training_mode: optimized` uses packed contexts and valid anchor blocks, segmented SDPA/eager or indexed FlexAttention, and projections limited to active supervision rows. It also enables cost-based batch planning. Supported offline collocated training defaults to selective cache reads, a CPU prefetch queue of depth 2, and asynchronous planning of the next window.

Online and disaggregated training do not enable the offline CPU queue. Supported disaggregated selector/indexer recipes stream planned counts; other recipes use the window execution path.

Example fragment:

```yaml
arguments:
  training_mode: optimized
  schedule_window_updates: 1
  save_interval: 15000
  save_strategy: interval
  save_training_state: true
```

Supply model/data/output paths and GPU topology in the full launcher.

## Planning and prefetch

`schedule_window_updates: 1` preserves the set of source occurrences in each optimizer update while balancing microsteps and ranks. Larger windows change update composition and training order. Kernel shapes and reduction order can produce numerical differences between profiles.

Optional offline tuning settings include:

```yaml
arguments:
  cpu_batch_prefetch: 0
  async_schedule_planning: false
  hidden_cache_selective_read: false
  h2d_prefetch: false
```

These settings disable the corresponding optimizations. Asynchronous scheduling supports optimized collocated execution. Offline training enables it by default and reads token-only cache metadata; online training can opt in with `async_schedule_planning: true` and reads the processed token/mask dataset. One background thread plans the next window while the current window trains. The committed cursor and pending-plan checkpoint contract are unchanged.

For online training, `dataloader_num_workers` and `dataloader_prefetch_factor` control the independent DataLoader queue (up to workers × prefetch-factor batches per rank). More workers increase process and IPC costs; more queued batches use additional host memory and can advance planning requests further ahead. `cpu_batch_prefetch` and `h2d_prefetch` remain offline-only.

H2D prefetch is disabled by default. Set `h2d_prefetch: true` to transfer one future batch on a separate CUDA stream. This requires additional memory for the prefetched inputs. CPU planning metadata remain available, and prefetching does not advance the committed training cursor.

Cache finite-value validation defaults to per-read. Audited loading requires a complete audit receipt; execution profiles do not disable validation.

## Checkpoints

Use `save_interval: 15000`, `save_strategy: interval`, and `save_training_state: true` for resumable intermediate checkpoints and final weights.

A full checkpoint stores optimizer and scheduler state, FP32 master parameters, per-rank RNG, the committed cursor, and the pending planner window. To continue a run, set `resume_from` and use the original training configuration and rank topology.

A weights-only checkpoint can initialize a new run through `init_from`. It cannot restore the optimizer or training cursor.

## Validate your setup

Run `python -m scripts.composable.train --help` to see supported arguments. The [test suite](../tests/README.md) covers losses, gradients, planning, prefetch lifecycle, distributed protocols, and restoration.

Measure acceptance and decoding throughput separately on your target model and hardware. Training correctness tests do not establish end-to-end speedups.
