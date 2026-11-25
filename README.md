<p align="center">
  <img src="assets/legospec_logo.svg" alt="LEGOSpec logo" width="480"/>
</p>

# LEGOSpec

LEGOSpec trains and runs speculative draft models for a frozen target language model. Build a draft by choosing how it combines target hidden states (**Context**), processes them (**Backbone**), predicts tokens (**Proposal**), and learns from supervision (**Objective**).

## Install

Use Linux, Python 3.11, an NVIDIA GPU, and a driver compatible with CUDA 13. Run the following from the repository root with [uv](https://docs.astral.sh/uv/) installed:

```bash
uv sync --locked --no-default-groups --extra cu130 --extra hopper
source .venv/bin/activate
```

The lockfile selects the runtime dependencies. CUDA extension builds also require a compatible local CUDA toolkit and C++ compiler. The `hopper` extra includes TileLang for GDN training on Hopper GPUs.

## Train a draft

You need a target checkpoint, a model recipe, and training conversations. Online training computes target features during training; offline training reads previously captured features.

The following example uses Qwen3-4B and a five-layer GQA draft. Save it as `train.yaml`, replacing the model, data, and output paths with your own:

```yaml
algorithm: composable
launcher:
  target_model: /path/to/Qwen3-4B
  train_data: /path/to/answers.jsonl
  output_dir: /path/to/checkpoints
  nnodes: 1
  node_rank: 0
  nproc_per_node: 1
  master_addr: 127.0.0.1
  master_port: 29501

arguments:
  draft_config: configs/composable_parallel_dpal_renyi.json
  target_model_backend: sglang
  sglang_attention_backend: triton
  sglang_mem_fraction_static: 0.3
  attention_backend: sdpa
  chat_template: qwen
  train_data_columns: [conversations]
  training_mode: legacy
  max_length: 4096
  batch_size: 1
  accumulation_steps: 1
  tp_size: 1
  block_size: 8
  num_draft_layers: 5
  mask_token_id: 151669
  num_anchors: 64
  num_epochs: 1
  learning_rate: 0.0006
  warmup_ratio: 0.04
  seed: 42
  save_interval: 15000
  save_strategy: interval
  save_training_state: true
```

This example requires enough GPU memory for the target, draft, optimizer, and activations. Adjust the batch size, sequence length, and GPU topology for your hardware.

Inspect the launch command, then start training:

```bash
bash scripts/composable/run_training_composable.sh train.yaml --dry-run
bash scripts/composable/run_training_composable.sh train.yaml
```

For all training arguments:

```bash
python -m scripts.composable.train --help
```

For offline training, replace `launcher.train_data` with `launcher.train_hidden_states`, pointing to a complete raw-tar feature cache. Keep the matching target checkpoint available for its tokenizer and shared embedding/output head. See the [data guide](scripts/data/README.md) for preparing conversations and capturing features.

Use [the recipe guide](configs/README.md) to choose attention, proposal, and loss settings. Target layer IDs, hidden dimensions, vocabulary, and mask token must match the target model; the Qwen3 example is not a universal recipe.

## Load a trained draft

A project-format inference checkpoint needs `config.json` and `model.safetensors`. Load it using the installed package:

```python
import torch
from legospec import LegoSpecConfig, LegoSpecModel

checkpoint = "/path/to/draft/current"
config = LegoSpecConfig.from_pretrained(checkpoint)
draft = LegoSpecModel.from_pretrained(
    checkpoint, config=config, dtype=torch.bfloat16
).cuda().eval()
```

The draft uses features, embeddings, and an output head from its matching target model. It is not a standalone language model. Project-format checkpoints do not require `auto_map`, generated Python files, or `trust_remote_code=True`.

See the [draft usage guide](legospec/modeling/draft/README.md) for a Qwen3 speculative-generation example and the lower-level feature API.

## Save and resume

The example saves every 15,000 optimizer steps and saves final weights. Full training checkpoints also contain optimizer and scheduler state, FP32 master parameters, per-rank RNG state, and the training cursor.

- Set `arguments.resume_from` to a full training checkpoint to continue the same run, using its original training configuration and rank topology.
- Set `arguments.init_from` to a compatible weights checkpoint to initialize a new run.

## Guides

- [Data preparation and feature capture](scripts/data/README.md)
- [Recipes and launcher configuration](configs/README.md)
- [Draft loading and generation](legospec/modeling/draft/README.md)
- [Attention families and access patterns](docs/ATTENTION_CONFIG.md)
- [Training execution and prefetch](docs/training-execution.md)
- [Runtime installation](docs/SGLANG_RUNTIME.md)
- [Data structure](legospec/data/DESIGN.md)
- [Model structure](legospec/modeling/DESIGN.md)
- [Training structure](legospec/training/DESIGN.md)
- [Running tests](tests/README.md)

The current training entrypoint does not support OPD. Deferred implementations are kept in `development/`.
