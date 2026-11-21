# OpenPerfectBlend 400k: two nodes, eight GPUs per node

Five independent, fresh runs: **01, 25, 23, 09, 05**. Markov is recipe **23**;
recipe 03 is a different MHA model. The JSON model configurations are byte
identical to the referenced round-one experiments.

Additional variant **25conv** starts from the server's original recipe 25 and
adds `dflash2_conv` (`kernel_size=2`, `group_size=16`) to all five layers, as in
recipe 05. Selector full-vocabulary residual scoring, rank 256, top-k 16, and
both CE weights remain unchanged. Set `RECIPE=25conv` in the launch command;
its YAML supplies a separate output directory and W&B run ID
`opb400k2n8g_25conv_s42_20261101`. Preparation and validation evidence is in
`tests/retrain_opb400k_2n8g_20261030/25conv/`.

Each run uses 2 nodes × 8 GPU workers, batch size 2 per GPU, accumulation 1,
global batch size 32, 4 epochs, seed 42, and complete supervised blocks.
Original loss weights and Muon/AdamW settings are preserved. Intermediate
checkpoints are saved every 15,000 optimizer steps, followed by the final
checkpoint. Training state is included so these checkpoints can be resumed.

Shared paths on both nodes:

- Target: `/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/models/Qwen/Qwen3-4B`
- Hidden cache: `/inspire/hdd2/project/inference-chip/public/wanghanzhen/SD_hidden/Qwen3-4B/openperfectblend/400k`
- Repository: `/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/LEGOSpec`
- Output: repository `output/opb400k_2n8g_20261030/<recipe>/checkpoints`

The `sglang` backend label is retained. Offline training reads the completed
hidden cache and loads standalone embedding/head components; it does not run
a new target-model prefill engine.

Launch **once per node**, using the same recipe on both nodes. Use node 0's
internal address reachable from node 1. The wrapper resolves `MASTER_ADDR`
and `NODE_RANK` into the launcher's existing `PET_*` overrides; do not run a
template YAML directly without those overrides. The literal `${MASTER_ADDR}`
in YAML is a runtime placeholder, not an address to use as-is.

Run the following once on each node. Machine environment variables supply
the master address, master port and node rank. Training writes directly to
files so browser terminal backpressure cannot block the training loop.

```bash
export LEGOSPEC_ROOT=/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/LEGOSpec
source "$LEGOSPEC_ROOT/tests/remote_env/activate.sh"
export NODE_RANK="${NODE_RANK:-${PET_NODE_RANK:?Missing node rank}}"
RECIPE=01
LOG_DIR="$LEGOSPEC_ROOT/tests/retrain_opb400k_2n8g_20261030/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${RECIPE}.node${NODE_RANK}.$(date +%Y%m%d_%H%M%S).log"
WANDB_MODE=offline TQDM_DISABLE=1 CUDA_LAUNCH_BLOCKING=1 \
nohup bash "$LEGOSPEC_ROOT/scripts/composable/run_training_opb400k_2n8g.sh" "$RECIPE" \
  --wandb-run-id "opb400k2n8g_${RECIPE}_s42_20261030_e4_r3" \
  > "$LOG_FILE" 2>&1 < /dev/null &
echo "PID=$! LOG=$LOG_FILE"
```

Set `RECIPE` to `25`, `23`, `09`, or `05` for the other runs. Each recipe
has a distinct default master port; `MASTER_PORT` can override it, identically
on both nodes. `--dry-run` prints the resolved command without starting workers.
Existing scheduler `PET_MASTER_ADDR`, `PET_NODE_RANK` or `GROUP_RANK` are also
accepted. Neither host addresses nor network-interface names are guessed.

The wrapper selects the isolated, hash-verified runtime in
`tests/retrain_opb400k_2n8g_20261030/source`, copied from the referenced original
experiment. This preserves its complete-block and optimizer implementation
while applying the requested epoch count and checkpoint policy. It defaults
to `.venv/bin/python` and offline W&B logging. Model/data mounts and that Python
environment must be available on both GPU nodes.

Configuration checks are in
`tests/retrain_opb400k_2n8g_20261030/validation.json` and cache/schedule checks
in `cache_validation.json` in the same directory. No GPU training is started
by configuration preparation or validation.

The complete cache contains 400,000 samples. Original complete-block filtering retains 398,125 and excludes 1,875. With the existing sampler padding and drop-last behavior, the configured global batch of 32 yields 12,441 optimizer steps per epoch and 49,764 across four epochs. Intermediate saves are due at steps 15,000, 30,000 and 45,000; final is saved after step 49,764.

`CUDA_LAUNCH_BLOCKING=1` enables synchronous error reporting for the unresolved
CUDA failure. This can reduce throughput; it does not fix the underlying fault.
View logs separately with `tail -n 100 -f "$LOG_FILE"`.
