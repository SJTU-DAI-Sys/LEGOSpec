# LEGOSpec

Composable speculative draft models built from **Context, Backbone, Proposal and Objective**, with shared training, data, distributed and inference facilities.

Public API: `from legospec import LegoSpecModel, LegoSpecConfig`.

- [Architecture and code organization](docs/COMPONENT_ORGANIZATION.md)
- [Completed simplification plan](docs/SIMPLIFICATION_PLAN.md)
- [Attention families and access](docs/ATTENTION_CONFIG.md)
- [Execution options](docs/training-execution.md)
- [Selected optimization versions](docs/VERSION_BRANCH_REVIEW.md)

Use `python -m scripts.composable.train` or `scripts/composable/run_training_composable.sh CONFIG.yaml`.
Algorithms are recipes; there are no standalone full-algorithm implementations. Supported combinations are checked by the schema. Hidden caches use raw tar only.

Dependencies are pinned in `pyproject.toml` and `uv.lock`; testing tools are centralized in the `test` dependency group. See [SGLang runtime and installation](docs/SGLANG_RUNTIME.md) for the CUDA 13 environment.

```sh
OMP_NUM_THREADS=1 python -m pytest tests/unit -q
```

Set `CUDA_VISIBLE_DEVICES` to available devices for CUDA coverage, or an empty value for CPU coverage. Generated test files stay in `tests/artifacts`.
Keep `training_mode: legacy | optimized`. New launchers explicitly save every 15,000 optimizer steps and final weights, with full training state for resume.

`model_publish` is a separate native-format weight release. Its config and weights are not implicitly converted by this source refactor. Active training snapshots, caches and checkpoints remain in their existing locations.

Publication license/version provenance remains documented in [publication notes](docs/publication-notes.md).

OPD training is deferred under [development/opd](development/opd/README.md) and is not supported by the current training entrypoint.
