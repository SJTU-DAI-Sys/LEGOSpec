# Draft component boundary refactor — completed on H800_2

Main project: /share/dai-sys/wanghanzhen/projects/DAC27/LEGOSpec_public

## Implemented

Four component axes now live directly under modeling/draft.
The composable directory and old import forwarding paths are absent.
The composable_config serialized recipe key and public LegoSpecModel/Config API remain.

Layer owns a real attention child for dense, latent and linear families.
LinearAttention owns position parameters, GDN/KDA mixer, intrinsic block attention
and the shared output projection. Position addition remains before Layer's
input norm/residual. External convolution remains owned by Layer.

Components define their special initialization; top-level modeling invokes the
uniform reset_draft_parameters hook after HF post_init without knowing parameter names.
Removed the single-provider BackboneKernels factory; use actual Qwen3 norm/MLP.

contracts contains inter-component messages; backbone/state owns cache types and
rollback. backbone/stepwise owns the shared hidden-feedback schedule.

Objective owns RatioTerms/ObjectiveOutput and position-weight mathematics.
training/outputs.py is removed. Draft implementation has no training imports.

configuration selects default target layers and validates the public recipe.
schema describes the public boundary, runtime_config expands the executable
representation. Models consume already parsed configuration.
Current recipes and test fixtures use family/variant/access. Removed string
recipe support and independently projected/optional linear block paths.

## Validation

- Final unit/CUDA regression: 543 passed, 9 existing skips,
  13 subtests passed in 89.05 seconds.
- With identical weights and inputs, pre-refactor vs current GQA, MLA, GDN and KDA
  forward outputs and every parameter gradient are bitwise equal.
  The test-only mapping translates relocated parameter names; production has no
  checkpoint compatibility mapping.
- Tests cover cache commit/rollback, grouped heads, gate gradients, packing,
  hidden-feedback, distributed reductions, optimizer resume and standalone export.
- New tests check draft does not import training, component initialization and
  rejection of string attention recipes.
- Clean wheel contains the new state/output modules and no old composable path.
- Focused import/static checks and source whitespace checks pass.

Tests for retired independently projected or no-block linear variants were archived
under tests/development/legacy_attention. The original source/test baseline is
preserved in before.tgz. Current intrinsic branch tests remain in tests/unit.

## Scope

Parameter registration paths changed: self_attn becomes attention; linear mixer,
position_embedding and block_attention become children of attention.
Old training resumes use their original source snapshot/configuration/rank topology.
Current code does not adapt old checkpoints. Existing weights, data, caches,
model_publish and training snapshots were not migrated.

Details: modeling/draft/DESIGN.md.
Evidence: regression.log, numerical.json, numerical_before/after.log, audit.json,
wheel.json, build.log. Numerical reproducer: numerical.py before/after.
