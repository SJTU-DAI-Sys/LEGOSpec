# Running tests

Install the test dependencies from the repository root:

```bash
uv sync --locked --no-default-groups --group test --extra cu130 --extra hopper
source .venv/bin/activate
```

Run CPU coverage:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 python -m pytest tests/unit -q -rs
```

Run with an available CUDA device:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m pytest tests/unit -q -rs
```

Tests that require unavailable hardware or optional dependencies report skips. Use `-k` to select a topic, for example `-k selector`.

The suite covers model outputs and gradients, attention and cache state, data loading, training losses, distributed protocols, checkpoint restoration, and export. These checks do not measure end-to-end decoding speed or acceptance on a language-model benchmark.

- `unit/` contains runnable correctness and regression tests.
- `reference/` contains small fixed numerical references and source provenance.
- `conftest.py` configures shared fixtures and temporary storage.

Generated files and default compilation caches are written under `tmp_exp/artifacts/`.
