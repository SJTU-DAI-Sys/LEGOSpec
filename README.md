# LEGOSpec

A composable framework for training and evaluating speculative draft models.
The Python package remains named `litespec` for import and checkpoint compatibility.

## Source layout

- `litespec/`: model components, composable recipes, training and inference.
- `scripts/`: training, data preparation and YAML launchers.
- `configs/`: draft recipes and benchmark configurations.
- `docs/`: component organization and execution-mode documentation.
- `tests/unit/`: maintained tests.
- `tests/reference/`: pinned numerical/export oracles used only by tests.
- `model_publish/`: the separately prepared Gemma DFlash2 weight release.

The concrete DFlash, DFlash2, DSpark and DLite entry points and Composable
import attention, convolution and proposal components from independent shared
modules. See [code organization](docs/code-organization.md).

## Development

Use Python 3.11 and the dependencies in `pyproject.toml`. From this directory:

```sh
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python -m pytest tests/unit -q
```

Dependencies must already be installed in the selected environment. Test results
and the exact environment used for this migration are recorded under
`tests/artifacts/public_source_migration_20261121/`.

Launch composable training with `python -m scripts.composable.train`, or use
`scripts/composable/run_training_composable.sh CONFIG.yaml`.

Retain the overall `training_mode: legacy | optimized` switch; `legacy` is
the default. See [training execution](docs/training-execution.md) for packed
execution, scheduling, checkpoint and resume contracts. Training launchers
must specify their own model, data, output paths and rank topology.

This iteration tree was populated from the maintained H800_2 framework on
2026-11-21. Historical experiments, active run snapshots, caches and training
checkpoints remain at their existing locations. New development takes place
in this directory. The earlier source tree is retained for path compatibility.

See [publication metadata](docs/publication-notes.md) for the inherited license
and version metadata that still need reconciliation before a public package release.
