# Modeling

This package implements draft computation, frozen-target execution, configuration, and KV caches. Public model classes are `LegoSpecConfig` and `LegoSpecModel`. Training loops and dataset storage live outside this package.

## Shared utilities

- `config_utils.py`: target/text configuration loading and Qwen3.5/3.6 draft configuration derivation.
- `kv_cache.py`: preallocated target KV caches.
- `__init__.py`: package entrypoint.

## Draft configuration and execution

Within `draft/`:

- `configuration.py`: serializable configuration and default target-layer selection.
- `schema.py`: recipe fields, defaults, and supported combinations.
- `runtime_config.py`: expansion into per-layer execution parameters.
- `loss_config.py`: loss weights, reductions, and proposal-specific objective settings.
- `modeling.py`: Context → Backbone → Proposal orchestration and hidden-feedback interfaces.
- `contracts.py`: inputs, component outputs, and shared embedding/head contracts.
- `export.py`: optional standalone Transformers source-bundle export.
- `__init__.py`: model, configuration, input/output, and state exports.

Components own their parameters and initialization. The model applies component reset hooks after Hugging Face initialization. Project-format checkpoints can load directly through the installed classes without a source bundle.

## Context

`draft/context/fusion.py` implements concatenation followed by projection, and scalar/channelwise gated fusion. Context returns a target-feature tensor for each backbone layer. The package entrypoint builds the configured context.

## Backbone

Within `draft/backbone/`:

- `model.py`: layer traversal, indexer integration, and state publication; each Layer combines normalization, attention, residual connections, optional convolution, and MLP.
- `convolution.py`: grouped dynamic convolution and initialization.
- `state.py`: dense/latent/indexer caches, DraftState, and rollback.
- `stepwise.py`: shared hidden-feedback scheduling, masks, and state extension.

Layers own independent attention modules. Linear-attention position additions precede input normalization and the residual path. Confirmed-history state is published after successful execution of all layers.

## Attention

Within `draft/backbone/attention/`:

- `base.py`: shared dense/latent softmax execution, KV commits, and backend dispatch.
- `dense.py`: MHA/GQA/MQA projections and RoPE.
- `latent.py`: MLA projections, expanded computation, absorbed inference, and latent caches.
- `linear/recurrence.py`: GDN/KDA history mixers, internal convolution, decay, scans, and output projection.
- `linear/block.py`: bidirectional block attention with per-head gates.
- `linear/events.py`: confirmed-context write and draft-read event layouts.
- `linear/cache.py`: recurrent states, convolution tails, and confirmed-prefix commits.
- `access/visibility.py`: position, document, and validity masks for full/SWA/DSA access.
- `access/indexer.py`: anchor-query history scoring, top-k selection, and auxiliary supervision.
- `access/dsa.py`: selected-KV gathering and attention/teacher statistics.
- `backends/flex.py`: FlexAttention compilation, masks, and kernel options.
- `backends/torch_compat.py`: PyTorch compilation integration.

Family selects parameterization, access selects visible history, and backend selects execution. GDN/KDA share Q/K/V/O projections between history reads and gated block attention; block reads do not write recurrent history.

## Proposal

Within `draft/proposal/`:

- `parallel.py`: independent vocabulary projections and shared output handling.
- `markov.py`: low-rank predecessor biases and Markov/RNN proposals.
- `sequential.py`: sequential hidden-replacement proposals.
- `selector.py`: candidate scoring, candidate-only selection, and full-vocabulary residual proposals.
- `sampling.py`: probability processing, sampling, and request seeds.

Proposal produces distributions and tokens. Objective computes losses, while inference performs acceptance/rejection.

## Objective

Within `draft/objective/`:

- `base.py`: supervision contracts and position weighting.
- `outputs.py`: additive loss/metric numerators and denominators.
- `token.py`: token CE, KL, TV, L1, NLA, and candidate CE.
- `block.py`: cross-position objectives, including DPACE and E2E-TV.
- `dpal.py`: D-PAL Rényi computation and gradients.
- `composition.py`: named loss branches, selector losses, and weighted objectives.
- `vocab.py`: teacher distributions, indexed projections, and temporary reuse.
- `kernels.py`: shared vocabulary-loss kernels and compilation.

Objectives are independent of training orchestration. Training consumes their additive outputs for accumulation and distributed reduction.

## Frozen targets

Within `target/`:

- `training_target_model.py`: backend construction and feature-output contracts.
- `hf.py`: Hugging Face loading, forwards, layer capture, and shared components.
- `sglang.py`: SGLang entrypoint and packed-to-padded output adaptation.
- `components.py`: embeddings/output heads, scaling/softcap, vocabulary TP communication, and component loading.
- `adapters/base.py`: default model structure and feature semantics.
- `adapters/gemma.py`: Gemma scaling, softcap, configuration, and final-norm reconstruction.
- `adapters/qwen35.py`: Qwen3.5/3.6 native capture differences.
- `sglang_backend/capture.py`: packed prefill, request boundaries, feature capture, and KV-pool lifetime.
- `sglang_backend/parallel.py`: collocated/disaggregated process-group integration.

Raw decoder outputs and post-final-norm teacher features are separate outputs. Adapters own model semantics; backends own execution and resources. Online training and offline collection share the target interface.

`target/custom_backend/` contains architecture-specific model implementations; the unified HF/SGLang constructors do not currently select them. OPD implementations are deferred under repository-level `development/`.
