# Runtime installation

Use Linux, Python 3.11, and an NVIDIA driver compatible with CUDA 13. Install the locked environment from the repository root:

```bash
uv sync --locked --no-default-groups --extra cu130 --extra hopper
source .venv/bin/activate
```

Add `--group test` when installing the regression-test dependencies. See the [test guide](../tests/README.md).

## Dependencies

The repository pins SGLang 0.5.21, PyTorch 2.13.0, Transformers 5.12.1, and sglang-kernel 0.4.7. The `hopper` extra pins TileLang 0.1.12 for the recurrent-attention backend on Hopper. Use `pyproject.toml` and `uv.lock` as the dependency source of truth.

The selected Torch wheel uses CUDA 13.0. Building CUDA extensions requires a compatible local CUDA toolkit and C++ compiler. The uv configuration builds causal-conv1d against the selected Torch ABI.

Check the installed runtime:

```bash
python -c "import torch, transformers, sglang; print(torch.__version__, torch.version.cuda, transformers.__version__, sglang.__version__)"
```

## Target backends

Select `target_model_backend: sglang` for SGLang target execution or `hf` for the Hugging Face backend. The SGLang native capture integration depends on interfaces in the pinned release; SGLang 0.5.9 does not provide the required bootstrap interface.

Online training and offline feature capture share target loading and model adapters. Captured context features are raw decoder outputs; teacher features are the normalized features used for vocabulary supervision. Gemma embedding scaling, logit softcapping, and final normalization are handled by its model adapter and shared components.

For offline Hugging Face capture, use `--target-model-backend hf --no-packed-requests`. See the [data guide](../scripts/data/README.md) for capture commands and cache formats.

## GPU capacity

Set the target memory fraction, batch size, sequence length, and tensor-parallel topology for your hardware. Target weights, KV allocations, draft activations, and optimizer state all consume device memory. A successful short capture does not establish that a larger training batch will fit.

For training execution options and restoration, see [training execution](training-execution.md). For backend interfaces and ownership, see [model structure](../legospec/modeling/DESIGN.md).

Native training/capture forces `enforce_disable_flashinfer_allreduce_fusion=True` alongside disabled custom AllReduce and CUDA graphs. This prevents model-specific auto-enablement of the FlashInfer fused communication workspace during eager warmup. Training passes its `dist_timeout` in minutes to SGLang as seconds; TP coordinators inherit that configured timeout when the existing training WORLD bypasses SGLang distributed bootstrap.
