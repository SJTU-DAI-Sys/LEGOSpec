#!/usr/bin/env bash
# Also installed at /inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/stop_keeper.sh.
set -uo pipefail
LEGOSPEC_ROOT="${LEGOSPEC_ROOT:-/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/LEGOSpec}"
KEEPER_PYTHON="${PYTHON_BIN:-${LEGOSPEC_ROOT}/.venv/bin/python}"
PID_FILE=/tmp/gpu_keeper.pid
echo 'Stopping GPU keepers...'
"$KEEPER_PYTHON" "$LEGOSPEC_ROOT/scripts/composable/run_with_keeper.py" \
    --keeper-script /inspire/hdd/project/inference-chip/xujiaming-253308120313/keep_alive.py \
    --keeper-launcher /inspire/hdd/project/inference-chip/xujiaming-253308120313/keep.sh \
    --stop-only || exit $?
# Retain cleanup for the earlier stress script after collecting the scheduler tree.
pkill -f '(^|[ /])gpu_matmul_stress[.]py([ ]|$)' || true
rm -f "$PID_FILE"
sleep 3
echo 'GPU keepers stopped.'
nvidia-smi --query-gpu=memory.used --format=csv
