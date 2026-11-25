#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NODE_RANK="${1:?usage: run_qz.sh NODE_RANK [--dry-run | training CLI overrides]}"
shift
case "${NODE_RANK}" in 0|1) ;; *) echo 'NODE_RANK must be 0 or 1' >&2; exit 2 ;; esac
TOOLS=/inspire/hdd/global_user/xujiaming-253308120313/tools
export CUDA_HOME="${TOOLS}/cuda-13.0"
export PATH="${CUDA_HOME}/bin:${TOOLS}/bin:${PATH}"
export PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TORCHINDUCTOR_CACHE_DIR="${PROJECT_DIR}/tmp_exp/artifacts/inductor"
export TRITON_CACHE_DIR="${PROJECT_DIR}/tmp_exp/artifacts/triton"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PET_NODE_RANK="${NODE_RANK}"
cd "${PROJECT_DIR}"
CONFIG="configs/qwen38_27b_dflash_swa2048_rselector_dpal/train_qz_node${NODE_RANK}.yaml"
for argument in "$@"; do
  if [[ "${argument}" == --dry-run ]]; then
    exec bash scripts/composable/run_training_composable.sh "${CONFIG}" "$@"
  fi
done
GEN_JOB=/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/LEGOSpec/tests/qwen38_27b_qz_return192_20261123
for part in 0 1; do
  if [[ ! -f "${GEN_JOB}/results/part${part}.complete.json" ]]; then
    echo "Generation part${part} is still unfinished; training was not started." >&2
    exit 1
  fi
done
mkdir -p "${PROJECT_DIR}/tmp_exp/qwen38_27b_training_20261124"
exec "${PYTHON_BIN}" -u "${GEN_JOB}/keeper_lifecycle.py" \
 --stop-script "${GEN_JOB}/stop_keeper.sh" \
 --keeper-script /inspire/hdd/project/inference-chip/xujiaming-253308120313/keep_alive.py \
 --keeper-launcher /inspire/hdd/project/inference-chip/xujiaming-253308120313/keep.sh \
 --keeper-log "${PROJECT_DIR}/tmp_exp/qwen38_27b_training_20261124/keeper.$(hostname).log" \
 -- bash scripts/composable/run_training_composable.sh "configs/qwen38_27b_dflash_swa2048_rselector_dpal/train_qz_node${NODE_RANK}.yaml" "$@"
