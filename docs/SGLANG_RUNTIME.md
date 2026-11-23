# SGLang runtime

The current target backend uses SGLang 0.5.21, PyTorch 2.13.0 (CUDA 13.0),
Transformers 5.12.1, Triton 3.7.1, sglang-kernel 0.4.7 and TileLang 0.1.12.
The old SGLang 0.5.9 environment has no distributed.bootstrap API and cannot
execute the native capture entrypoint.

Install from the repository on Linux with CUDA 13:

```sh
uv sync --locked --no-default-groups --group test --extra cu130 --extra hopper
```

The PyPI PyTorch 2.13.0 wheel uses CUDA 13.0. The former cu128 extra has been
removed because PyTorch 2.13.0 is not available in that index. causal-conv1d is
built against the selected Torch ABI. The uv build settings force a local build, avoiding an unrelated prebuilt extension.

H800_2 main interpreter:
`/mnt/dai-sys/wanghanzhen/projects/DAC27/envs/legospec-public-py311-cu130-h8002/bin/python`.

The new environment is independent of the previous LiteSpec shared environment.
The old overlay is retained as
`/mnt/dai-sys/wanghanzhen/projects/DAC27/envs/legospec-public-py311-cu130-h8002.before-sglang0521-20261123`. Existing training snapshots and
checkpoints are unchanged; continuing an old run still requires its original
training configuration and complete optimizer state.

## Native capture contract

All SGLang targets use sglang_backend/capture.py and the upstream ModelRunner.
Model adapters select decoder layers and preserve model-specific semantics.
Context features are raw decoder outputs; teacher features are the normalized
input to the native logits processor. They are captured separately.
Qwen captures its final residual before the in-place final RMSNorm. Gemma's
native auxiliary output already includes the final residual.

Online training restores padded batches into ordinary tensors that trainable
draft projections can save for backward. Offline collection uses packed rows
and token offsets from the same engine. Each prefill releases its request/KV
pool allocations, including on forward failure.

The former OPD anchor scorer is now deferred under development/opd and has no
active backend entrypoint. Collocated
teachers use TP-scoped SGLang groups; disaggregated teachers reuse process groups
created before the target/draft split. The trainer WORLD is preserved.

## Validation

Task artifacts and reproducible smoke scripts:
`tests/environment_upgrade_20261123/`.
The full regression passed: 582 tests, 9 existing candidate-only CE skips and
13 subtests. Real-weight prefill passed for Gemma 12B (TP1/TP2), Qwen3-4B
(TP1 and disaggregated target/draft ranks), and Qwen3.6-35B-A3B (TP2).
These checks cover online/offline capture, raw context vs normalized teacher,
teacher logits and trainable draft-projection backward. Qwen3 shared-prefix
anchor scoring was also compared with independent causal branches; BF16
relative L2 error was below 0.81%, with minimum cosine similarity above 0.99994.
The anchor results describe the environment-upgrade snapshot before OPD was
deferred; current mainline no longer supports that training path.
These are short-prefill checks, not acceptance benchmarks or long training runs.

See environment_after.json, freeze_after.txt, regression.log and the model result
JSON files in the task artifact directory for exact evidence.
