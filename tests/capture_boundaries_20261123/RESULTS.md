# Capture boundary refactor — 2026-11-23

Host: H800_2. Repository: LEGOSpec_public. Python: legospec-public-py311-cu130-h8002.

Implemented:
- Deleted modeling/target/sglang_gemma4_capture.py; native packed execution is in sglang_backend/capture.py.
- Moved native TP runtime to sglang_backend/parallel.py; online and offline use the common LEGOSpec distributed mesh and get_training_target_model entrypoint.
- Gemma adapter no longer creates an engine or initializes distributed state. Shared backend constructs NativeCapture directly.
- Moved cached Gemma norm checkpoint keys, RMSNorm and precision handling to the model adapter; reader consumes a callable reconstruction module and raw config metadata.
- Removed eager import of the alternate SGLang runner from the backend namespace.
- No old-path forwarding module or compatibility version added. Existing unrelated working-tree changes and historical artifacts remain untouched.

Verification:
- Full unit/GPU regression: 585 passed, 9 existing candidate-only CE skips, 13 subtests passed in 102.44 s. See pytest.log.
- Native capture protocol tests use a controlled SGLang fixture: initialization options, TP mesh handoff, independent request causal boundaries, layer/teacher tensors, packed offsets, pool cleanup on failure, online/offline factory routing and padded restoration input gradients.
- Existing actual HF small-model forward/gradients, CUDA D2H/pinned slots, cache round trips and full optimizer-resume tests are included.
- AST import/ownership audit passed for the changed target/data boundaries; no active imports of the deleted capture file remain. Other confirmed boundary findings are recorded separately, not silently treated as fixed.

Real-runtime limitation:
- Main environment: SGLang 0.5.9, Transformers 4.57.1.
- Both the original file from before.tgz and the new native engine fail real import because sglang.srt.distributed.bootstrap is absent. See environment.json / environment.log.
- No matching bootstrap.py was found in the inspected project envs directory. Large Gemma native runtime loading/training was not validated, and this task did not upgrade the shared environment or introduce runtime-version branches.

Structure and follow-up ownership plan: docs/TARGET_CAPTURE_PLAN.md.
Snapshots and per-file changes: before.tgz, status_before.txt, audit.json.
