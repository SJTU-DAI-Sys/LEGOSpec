# OPD deferral — 2026-11-23

Completed directly in H800_2 LEGOSpec_public.

## Deferred implementation

development/opd/target/anchor_opd.py — HF Qwen3 anchor scorer.
development/opd/target/sglang_anchor_opd.py — native SGLang anchor scorer.
development/opd/sampled.py — sampled-policy collection, verification and replay.
development/opd/configs/anchor_comparison_sft_rnn.yaml — paired SFT control.
development/opd/integration_snapshots/ — original integration files for future work.
tests/development/opd/ — dedicated test and original mixed-test snapshots.

These are development references, not installed or runnable training plugins.
The previously deferred opd_survival objective remains under development/objectives.

## Removed active integration

- Target backend scorer factory and abstract interface.
- OPD CLI options and sampled-anchor defaults/validation.
- TrainingStep collection/replay and dedicated forward arguments.
- Special target allocation budgets and offline-cache override.
- Optimizer-window OPD bypass, checkpoint metadata and rollout metric requirement.
- SAMPLED_ANCHOR_OBJECTIVES constant and dedicated active launcher.

Unknown TrainingStep options now fail at the constructor boundary instead of
silently ignoring a removed training mode. CLI options are rejected by argparse.
Normal E2E-TV, fixed-data supervision, anchor location sampling, causal block
attention and inference rejection sampling remain active.

## Validation

- Active legospec/scripts/configs source/config scan: no OPD references.
- Old scorer/sampled source paths are absent; no forwarding modules.
- Built wheel: 120 entries, no OPD modules or development files.
- Full CUDA/unit regression on GPU 2: 572 passed, 9 pre-existing skips,
  13 subtests passed, 93.79 seconds.
- Dedicated negative tests reject both removed CLI switches and constructor mode.
- Archive integrity: 9 moved/copied implementation/config/test files match the
  pre-edit archive byte-for-byte.
- Source whitespace and focused import/static checks pass.

Logs: regression.log, focused.log, build.log.
Machine-readable audits: audit.json, archive_integrity.json, moves.json.
Changes: change.diff and changed_files.json.
