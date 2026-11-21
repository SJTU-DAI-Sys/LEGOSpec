# Training execution modes

H800_2 main checkout is the integration baseline after 2026-11-21.
Existing experiment `tests/*/source` snapshots and queues stay pinned.

Use the normal framework entry point `python -m scripts.composable.train`,
or `scripts/composable/run_training_composable.sh CONFIG.yaml`.
The shared YAML launcher forwards `arguments.training_mode` to `--training-mode`.
No experiment monkeypatch or alternate trainer is required.

```yaml
arguments:
  training_mode: optimized  # legacy remains the default for existing configs
  schedule_window_updates: 1
  save_interval: 15000
  save_strategy: interval
  save_training_state: true
```

This is a configuration fragment; supply model/data/output and topology in your
launcher. Final saving remains enabled. Do not change historical run YAMLs.

## What the switch controls

`legacy`: padded execution, existing sampler, no automatic CPU queue/lookahead,
full raw-cache reads by default.

`optimized`: packed context and valid anchor blocks, segmented SDPA/eager or
indexed Flex attention, active supervision projection and cached unique teacher
normalization, metadata-only cost-balanced planning. Offline collocated training
also defaults to selective raw-cache reads, CPU queue depth 2 and asynchronous
next-window planning. Online/disaggregated execution does not enable the offline
queue. Disaggregated supported selector/indexer recipes stream planned counts;
unsupported streaming recipes retain their existing window path.

Optional tuning overrides remain available: `cpu_batch_prefetch: 0`,
`async_schedule_planning: false`, `hidden_cache_selective_read: false`.
Do not use async scheduling with legacy/online/disaggregated execution.
Finite validation still defaults to per-read; audited loading requires an
explicit complete audit receipt. The switch never silently disables validation.

W=1 preserves the optimizer update's source occurrence set, while balancing
microsteps/ranks. W>1 explicitly changes update composition and training order.
Loss/gradient tolerance tests are required; bitwise trajectory identity is not
promised. Shape-dependent kernels and floating-point reduction order can differ.

Correctness fixes (stream lifetime, target softcap, TP collective contracts),
bounded acceptance-metric memory, and fused soft-CE reductions are shared by both
modes. `legacy` is the padded execution profile, not a byte-for-byte historical
source rollback. Historical experiments use their frozen source copies.

## Checkpoint compatibility

Optimized checkpoints store `execution_contract=composable_packed_v1` and the
pending scheduler plan/committed cursor, in addition to full optimizer/scheduler,
FP32 parameters and per-rank RNG. Resume using the original configuration and
rank topology. Changing legacy/optimized requires `init_from` as a new run;
`resume_from` rejects the mismatch. Never resume the same run from weights only.

## Validation

CPU tests cover packed/padded loss and gradient equivalence, sampling/ownership,
TP protocol, planner ordering/resume, prefetch lifecycle and profile resolution.
Run from this checkout with its supported environment:

```sh
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python -m pytest tests/unit -q
```

GPU acceptance/performance must be measured separately on available hardware.
Do not interpret CPU tests or historical synthetic timings as an end-to-end
production speed guarantee.
