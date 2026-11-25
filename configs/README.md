# Model recipes and training configuration

LEGOSpec uses two configuration files:

- A **JSON model recipe** selects Context, Backbone, Proposal, and Objective.
- A **YAML launcher** selects model/data paths, GPUs, batching, optimizer, logging, and checkpoint settings.

Start with the complete launcher example in the [main README](../README.md). Set `arguments.draft_config` to a JSON recipe in this directory.

## Choose a recipe

- [Parallel + D-PAL](composable_parallel_dpal_renyi.json): five GQA layers, parallel token prediction, and D-PAL supervision.
- [RNN + D-PAL](composable_rnn_dpal_renyi.json): an RNN proposal with D-PAL supervision.
- [Residual selector + D-PAL](composable_residual_selector_dpal_renyi.json): top-k corrections to a full-vocabulary proposal.
- [GDN + RNN](dflash_gdn5_shared_block_attention_rnn.json): five GDN layers with integrated gated block attention.
- [KDA + RNN](dflash_kda5_rnn.json): a KDA backbone with an RNN proposal.
- [EAGLE3](composable_eagle3.json): hidden-feedback drafting.

Inspect the selected recipe before changing targets. Target layer IDs are zero-based decoder layer indices. Head counts, head dimensions, hidden size, vocabulary, RoPE, and mask token need to match the intended draft/target pair. Draft and target head layouts may differ.

## Attention settings

Attention is described by family, variant, and access:

```json
{"attention": {"family": "dense", "variant": "gqa", "access": "swa"}}
```

Dense attention includes MHA, GQA, and MQA; latent attention uses MLA. Full, sliding-window (`swa`), and DSA access are available for supported dense/latent combinations. Set shared parameters in `attention_configs` and `access_configs`.

GDN and KDA use the linear family and include gated block attention internally. See [attention configuration](../docs/ATTENTION_CONFIG.md) for supported fields and combinations.

## Run settings

Set exactly one of `launcher.train_data` (online target features) and `launcher.train_hidden_states` (offline feature cache). The target checkpoint remains required in both modes.

YAML keys under `arguments` map to CLI flags: for example, `batch_size` becomes `--batch-size`. Inspect the resulting command before launching:

```bash
bash scripts/composable/run_training_composable.sh /path/to/train.yaml --dry-run
python -m scripts.composable.train --help
```

`training_mode: legacy` uses padded execution. `training_mode: optimized` enables packed execution and, for supported offline training, selective reads and prefetching. Start with `schedule_window_updates: 1` when using optimized planning.

Set `save_interval: 15000`, `save_strategy: interval`, and `save_training_state: true` to retain resumable intermediate checkpoints and final weights. Use `resume_from` for a full resume and `init_from` for a new run initialized from weights.

Date-stamped directories contain experiment-specific configurations. Their paths and GPU settings need editing before reuse.
