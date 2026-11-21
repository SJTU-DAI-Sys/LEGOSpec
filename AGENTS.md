Store debugging, testing, and evaluation scripts and files in the repository's
`tests/` directory.

For new training runs, save weights every 15,000 optimizer steps and save the
final weights. Set `save_interval: 15000` and, where supported,
`save_strategy: interval` explicitly when preparing a launcher YAML.
Every intermediate checkpoint must support full training resume. Set
`save_training_state: true` on launchers that expose the option. Preserve
optimizer and scheduler state, FP32 master parameters, per-rank RNG, and the
training cursor. Use `resume_from` with the original training configuration and
rank topology. A weights-only export is suitable for inference or
`init_from`, not for continuing the same training run.

Choose training and evaluation hosts from the current task's instructions and
the location of its model, data, and checkpoints. Prefer using existing large
caches in place. Host addresses from one experiment are not global defaults.
Use separate training and evaluation hosts when specified for that task.
An evaluation can start once its checkpoint is verified. Training may proceed
while evaluations are queued or running when resources permit.

For new W&B projects, give the project a descriptive name that identifies the
target model and experiment series. Use detailed W&B run display names in the
form `context__backbone__proposal__objective__dataT0__s42`, with the actual data
temperature and seed. Include relevant layer counts, attention patterns,
convolution and proposal sizes, and loss weights or other objective parameters.
For example,
`dflash__gdn4_gqa1_dense_conv2g16__rnn_r256__ce0p1_l10p9_g4__dataT0__s42`.
Set the display name with `wandb_name` and keep `wandb_run_id` stable for upload
and resume.

Before starting a task, explicitly inspect `.agents/skills/*/SKILL.md` in this
repository. The `.agents` directory is hidden; do not rely on default search
results that exclude hidden files. If a task matches a repository skill's
description, read its entire `SKILL.md` and any required references before
proceeding. Follow its confirmation steps and use autonomous fallbacks only
where explicitly allowed.

For downloading, inspecting, extracting prompts from, or converting a new
dataset, use `.agents/skills/prepare-spec-decoding-dataset/SKILL.md`. Before
creating an adapter or running the full conversion, inspect five records.
When interaction is available, confirm both of the following with the user:

1. Which field or path to use for prompt extraction.
2. Whether to use `multi` to retain all user turns or `first` to retain only the
   first user turn.

When interaction is available, wait for explicit answers to both questions
before creating an adapter or running the full conversion. If interaction is
unavailable, choose the highest-confidence non-empty prompt candidate from the
inspected records and default to `multi`, unless the user has already specified
these choices. Report the selected field or path and turn mode, then proceed
with adapter creation, sample validation, and full conversion. Stop if no
credible prompt candidate exists.
