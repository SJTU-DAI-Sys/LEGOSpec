# Code organization

H800_2 main checkout is the maintained source. Historical experiment snapshots
are immutable; never replace their imports or modify their data/checkpoints.

## Dependency direction

`draft/composable` and the concrete `dflash`, `dflash2`, `dspark`, `dlite`
architectures depend on shared modules in `litespec/modeling/draft/`:

- attention.py: GQA/MHA/MLA, RoPE, mask helpers and decoder layer.
- convolution.py: grouped block convolution and its decoder integration.
- selector.py: low-rank candidate selector.
- markov.py: Markov/RNN logit heads and acceptance predictor.
- sequential.py: recurrent serial vocabulary head.
- context_utils.py: target embedding/head access and context-layer helpers.
- kernels.py: norm/MLP factories and optional kernel integration.

The shared modules never import concrete algorithms. Composable does not import
those algorithms either. Algorithm modules retain compatibility re-exports;
there is only one implementation of each extracted component. Existing class
names and parameter names are retained to preserve checkpoint structure.
The modules remain flat because existing standalone HF exports copy a flat
source dependency closure. All four algorithm exports are tested outside the
checkout. Composable exports include the shared components, not full algorithm
implementations.

Concrete algorithm inheritance (DFlash2/DSpark extending the DFlash model)
remains architecture composition; it is not the dependency path for shared
heads or Composable. `dflash_kernels.py` is a compatibility import only.

## Training execution

`scripts/composable/train.py` -> `training/recipes/composable.py` + shared
`training/runner.py`. Keep `training_mode: legacy | optimized` and the shared
`execution.py` defaults; see training-execution.md. This organization-only change
does not change loss math, tensor parameter names, sampling, or checkpoint mode
compatibility. No live experiment is migrated to another mode.

## Tests and history

- tests/unit/: maintained tests; `python -m pytest` selects only these.
- tests/reference/: pinned upstream and old-version numerical/export oracles,
  with source paths and SHA256 provenance. Not a runtime dependency.
- tests/artifacts/: generated test output and maintenance records.
- tests/archive/: completed experiments, migration snapshots and retired code.
  Archive scripts are historical records, not supported entry points.
- Existing active experiment directories, data caches and checkpoint paths stay
  in place. They must not be moved merely to make the directory tree uniform.

Unimplemented proposals are documented in roadmap.md, outside runtime modules.
