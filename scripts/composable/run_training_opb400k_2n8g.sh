#!/usr/bin/env bash
# Run once on each node, with identical recipe and MASTER_ADDR; NODE_RANK is 0/1.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EXPERIMENT="${ROOT}/tests/retrain_opb400k_2n8g_20261030"
CONFIG_DIR="${ROOT}/scripts/config/opb400k_2n8g_20261030"
RUNTIME="${EXPERIMENT}/source"
KEEPER_LOG_DIR="${EXPERIMENT}/logs"

if [[ $# -eq 0 || "$1" == --help || "$1" == -h ]]; then
  cat <<'HELP'
Usage: MASTER_ADDR=<node-0 internal IP> NODE_RANK=<0|1> bash scripts/composable/run_training_opb400k_2n8g.sh <05|09|25conv|25conv_dpal|25conv_dpal_basece|25swa|25swa_dpal|25swa_dpal_online|25swa_dpal_online_b1|25swa_dpal_online_b1_qwen35_9b|35|35swa2048|35swa5_2048|35swa5|35dpal|05dpal|09conv_tv|05conv_tv> [--dry-run]
Start once per node. Both nodes must use the same recipe and master address.
MASTER_PORT is optional; each recipe has a distinct configured default port.
PYTHON_BIN may select the shared Python environment. Default: repository .venv.
LITESPEC_LOSS_BACKEND=inductor opts into the fused D-PAL runtime.
Real runs stop all keepers first and restore the original keep.sh schedule after exit.
--dry-run prints the training command without changing keeper processes.
HELP
  exit 0
fi

case "$1" in
  35swa5_2048) STEM=35swa5_2048__dflash__rnn_r256__ce0p1_l10p9_g4__dataT0 ;;
  35swa2048) STEM=35swa2048__dflash__rnn_r256__ce0p1_l10p9_g4__dataT0 ;;
  35swa5) STEM=35swa5__dflash__gqa5_swa512_conv2g16__rnn_r256__ce0p1_l10p9_g4__dataT0 ;;
  35) STEM=35__dflash__gqa4_swa512_gqa1_dense_conv2g16__rnn_r256__ce0p1_l10p9_g4__dataT0 ;;
  1|01|01dpal|01e2etv|23|25|25topk|25_topk|25topkconv|25topk_conv|25conv_r64|25conv_k64|25conv-k64|25conv64)
    echo "Recipe $1 is outside the reconciled QZ 400K training scope." >&2
    exit 2
    ;;
  25conv|25_conv) STEM=25conv__dflash__gqa5_dense_conv2g16__selector_r256_k16__basece0p1_finalce0p1_finall10p9_g4 ;;
  25conv_dpal) STEM=25conv_dpal__dflash__gqa5_dense_conv2g16__selector_r256_k16__final_dpal_renyi; RUNTIME="${EXPERIMENT}/block_objectives/source" ;;
  25conv_dpal_basece) STEM=25conv_dpal_basece__dflash__gqa5_dense_conv2g16__selector_r256_k16__basece0p1_final_dpal_renyi; RUNTIME="${EXPERIMENT}/block_objectives/source" ;;
  25swa_dpal) STEM=25swa_dpal__dflash__gqa4_swa512_gqa1_dense_conv2g16__selector_r256_k16__final_dpal_renyi; RUNTIME="${EXPERIMENT}/block_objectives/source" ;;
  25swa_dpal_online|25swa_dpal_online_b1)
    STEM="$1"
    CONFIG_DIR="${ROOT}/scripts/config/regen_mix_2n8g_20261110"
    KEEPER_LOG_DIR="${ROOT}/tests/regen_mix_2n8g_20261110/logs"
    RUNTIME="${ROOT}/tests/regen_mix_2n8g_20261110/source"
    if [[ "$STEM" == 25swa_dpal_online_b1 ]]; then
      RUNTIME="${ROOT}/tests/regen_mix_2n8g_20261110/source_b1"
      export LITESPEC_LOSS_BACKEND=inductor
    fi
    ;;
  25swa_dpal_online_b1_qwen35_9b)
    STEM="$1"
    CONFIG_DIR="${ROOT}/scripts/config/qwen35_9b_mixed_2n8g_20261111"
    KEEPER_LOG_DIR="${ROOT}/tests/qz_math8_wildchat_20261111/training_logs"
    RUNTIME="${ROOT}/tests/regen_mix_2n8g_20261110/source_b1"
    export LITESPEC_LOSS_BACKEND=inductor
    ;;
  35dpal) STEM=35dpal__dflash__gqa4_swa512_gqa1_dense_conv2g16__rnn_r256__dpal_renyi; RUNTIME="${EXPERIMENT}/block_objectives/source" ;;
  05dpal) STEM=05dpal__dflash__gqa5_dense_conv2g16__rnn_r256__dpal_renyi; RUNTIME="${EXPERIMENT}/block_objectives/source" ;;
  25swa|25_swa|25swaconv) STEM=25swa__dflash__gqa4_swa512_gqa1_dense_conv2g16__selector_r256_k16__basece0p1_finalce0p1_finall10p9_g4 ;;
  9|09) STEM=09__dflare__gqa5_dense__rnn_r256__ce0p1_l10p9_g4 ;;
  09conv_tv)
    STEM=09conv_tv__dflare__gqa5_dense_conv2g16__rnn_r256__ce0p1_tv0p9_g4
    CONFIG_DIR="${ROOT}/scripts/config/regen_mix_2n8g_20261105"
    KEEPER_LOG_DIR="${ROOT}/tests/regen_mix_2n8g_20261105/logs"
    ;;
  5|05) STEM=05__dflash__gqa5_dense_conv2g16__rnn_r256__ce0p1_l10p9_g4 ;;
  05conv_tv)
    STEM=05conv_tv__dflash__gqa5_dense_conv2g16__rnn_r256__ce0p1_tv0p9_g4
    CONFIG_DIR="${ROOT}/scripts/config/regen_mix_2n8g_20261105"
    KEEPER_LOG_DIR="${ROOT}/tests/regen_mix_2n8g_20261105/logs"
    ;;
  *) echo "Unknown recipe: $1. Run with --help for the active recipes." >&2; exit 2 ;;
esac
shift

if [[ "$STEM" == *dpal_renyi && "${LITESPEC_LOSS_BACKEND:-auto}" == inductor ]]; then
  RUNTIME="${EXPERIMENT}/dpal_fused/source"
fi

TASK_MASTER_ADDR="${MASTER_ADDR:-${PET_MASTER_ADDR:-}}"
TASK_NODE_RANK="${NODE_RANK:-${PET_NODE_RANK:-${GROUP_RANK:-}}}"
if [[ -z "$TASK_MASTER_ADDR" || "$TASK_MASTER_ADDR" == 127.* || "$TASK_MASTER_ADDR" == localhost || "$TASK_MASTER_ADDR" == ::1 || "$TASK_MASTER_ADDR" == 0.0.0.0 ]]; then
  echo "Set MASTER_ADDR to node 0's address reachable from both nodes." >&2
  exit 2
fi
if [[ "$TASK_NODE_RANK" != 0 && "$TASK_NODE_RANK" != 1 ]]; then
  echo "Set NODE_RANK=0 on the first node and NODE_RANK=1 on the second." >&2
  exit 2
fi
if [[ -n "${LOCAL_RANK:-}" ]]; then
  echo "Run this node launcher once per node, outside an existing torchrun worker." >&2
  exit 2
fi
if [[ ! -f "${RUNTIME}/scripts/config/launch_composable.py" ]]; then
  echo "Missing frozen runtime: ${RUNTIME}" >&2
  exit 2
fi
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${ROOT}/.venv/bin/python"
  else
    PYTHON_BIN=python3
  fi
fi
export PET_NNODES=2 PET_NPROC_PER_NODE=8 PET_NODE_RANK="$TASK_NODE_RANK" PET_MASTER_ADDR="$TASK_MASTER_ADDR"
TASK_MASTER_PORT="${MASTER_PORT:-${PET_MASTER_PORT:-}}"
if [[ -n "$TASK_MASTER_PORT" ]]; then
  if [[ ! "$TASK_MASTER_PORT" =~ ^[0-9]+$ ]] || (( 10#$TASK_MASTER_PORT < 1 || 10#$TASK_MASTER_PORT > 65535 )); then
    echo "MASTER_PORT must be an integer in [1, 65535]." >&2
    exit 2
  fi
  export PET_MASTER_PORT="$TASK_MASTER_PORT"
else
  unset PET_MASTER_PORT || true
fi
export WANDB_MODE="${WANDB_MODE:-offline}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${RUNTIME}${PYTHONPATH:+:${PYTHONPATH}}"
cd "$ROOT"
for TASK_ARG in "$@"; do
  if [[ "$TASK_ARG" == --dry-run ]]; then
    exec "$PYTHON_BIN" "${RUNTIME}/scripts/config/launch_composable.py" \
      --config "${CONFIG_DIR}/${STEM}.yaml" "$@"
  fi
done
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  exec "$PYTHON_BIN" "${RUNTIME}/scripts/config/launch_composable.py" \
    --config "${CONFIG_DIR}/${STEM}.yaml" "$@"
fi
if [[ "$STEM" == 25swa_dpal_online_b1_qwen35_9b ]]; then
  MIXED_DATA="${ROOT}/../training_data/regen/nemotron_math8_wildchat200k_20261111/pb_nm_dedup_nemotron_math8_wildchat200k_qwen35_9b_mix_s42.jsonl"
  [[ -s "$MIXED_DATA" ]] || { echo "Verified mixed dataset is not ready: $MIXED_DATA" >&2; exit 1; }
fi
exec "$PYTHON_BIN" "${SCRIPT_DIR}/run_with_keeper.py" \
  --stop-script /inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/stop_keeper.sh \
  --keeper-script /inspire/hdd/project/inference-chip/xujiaming-253308120313/keep_alive.py \
  --keeper-launcher /inspire/hdd/project/inference-chip/xujiaming-253308120313/keep.sh \
  --keeper-log "${KEEPER_LOG_DIR}/keeper.node${TASK_NODE_RANK}.$(date +%Y%m%d_%H%M%S).log" \
  -- "$PYTHON_BIN" "${RUNTIME}/scripts/config/launch_composable.py" \
  --config "${CONFIG_DIR}/${STEM}.yaml" "$@"
