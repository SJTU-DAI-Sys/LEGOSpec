# Target boundary / model adapter refactor — 2026-11-22

Host: H800_2. Repository: LEGOSpec_public.

- Training target setup owns training requirements, memory/request budgets, tokenizer/mask and assembly. Features only selects/concatenates/moves tensors; streaming uses the same selection contract.
- HF and SGLang own execution. Default, Gemma and Qwen3.5/3.6 adapters own model semantics and capture differences. Gemma training uses the common SGLang target with its existing native packed capture engine.
- Shared components and TP operations are in components.py. Runtime creates anchor scorers through the target interface.
- Decoder layer IDs are explicit dictionary keys. Final decoder residual and post-final-norm teacher features are separate. Generic SGLang now captures the raw final residual before fused norm changes the inputs.
- Cache producers, readers and pinned-buffer budgeting use the explicit two-stream contract; no missing residual/teacher is substituted for the other.
- Removed target_utils.py and the independent sglang_gemma4_training.py. No forwarding modules or compatibility versions added.

Validation:
- Final full unit suite on GPU 0: 580 passed, 9 existing candidate-only CE skips, 13 subtests passed (106.12 s). See pytest.log.
- Tests cover actual small HF Qwen forward and shared-head input gradients, Gemma native-capture fixtures/scaling/softcap/TP storage, generic SGLang packed/padded contracts, instance hook and communicator patch cleanup, CUDA async capture and pinned-slot reuse, and existing full optimizer resume.
- SGLang backend dependency import succeeds. Large pretrained SGLang/Gemma model loading and full training were not run for this refactor.
- Python syntax audit and target boundary audit passed; target code lint passed.

before.tgz is the pre-task source/test snapshot. audit.json lists changes relative to it. Existing unrelated working-tree changes were preserved.
