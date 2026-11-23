# H800_2 SGLang environment upgrade — completed

Main project: /share/dai-sys/wanghanzhen/projects/DAC27/LEGOSpec_public

SGLang 0.5.9 was missing distributed.bootstrap. The main Python entrypoint now
selects an independent environment with SGLang 0.5.21, Torch 2.13.0+cu130,
Transformers 5.12.1, Triton 3.7.1 and TileLang 0.1.12. The bootstrap API and
rebuilt causal-conv1d CUDA extension import successfully. All 226 installed
packages pass dependency checks. pyproject.toml and uv.lock pin the new stack.

Main entrypoint:
/mnt/dai-sys/wanghanzhen/projects/DAC27/envs/legospec-public-py311-cu130-h8002/bin/python

The repository .venv points to this entrypoint. The previous overlay is preserved
at the backup path in environment_switch.json. The shared LiteSpec base was not
modified.

## Code changes

All SGLang target models use the native runner. Removed the old custom runner,
patch and logits wrapper. Adapted request EXTEND ranges, KV slot ownership and
explicit normalized-teacher capture. Model adapters retain layer semantics,
including raw final residual capture before in-place RMSNorm.

Preserved collocated TP and disaggregated target/draft execution. Updated the
Qwen3 anchor scorer, Transformers RoPE metadata access, DSA theta lookup and
DynamicCache test access. FSDP layout validation now checks the actual layout
rather than rejecting a Torch version string.

## Validation

- Full unit/CUDA regression: 582 passed, 9 pre-existing candidate-only CE skips,
  13 subtests passed, 87.66 seconds.
- Main entrypoint after switching and synchronizing dependencies: 23 passed,
  including actual GDN/KDA CUDA forward/backward and cache checks.
- Gemma 4 12B: actual weights, TP1 and TP2.
- Qwen3-4B: actual weights, TP1 and separate target/draft ranks.
- Qwen3.6-35B-A3B: actual weights, TP2.
- Real-model checks: packed/padded feature agreement, finite decoder features,
  teacher head vs native logits and draft-projection backward.
- TP1 Gemma/Qwen: final-only capture matches capture with other selected layers.
- Qwen3 shared-prefix anchor branches: BF16 relative L2 error below 0.81% and
  minimum cosine above 0.99994 against independent causal branches.
- Source whitespace checks and focused import/static checks passed.

These are short synthetic-prefill and regression checks, not long-context
acceptance evaluations or a training run. No model weights/data/checkpoints were
modified.

Exact versions: environment_after.json and freeze_after.txt.
Logs: regression.log, main_entrypoint_checks.log, gemma.log, qwen.log,
gemma_tp2.log, qwen_disagg.log, qwen35_tp2.log and pip_check.log.
