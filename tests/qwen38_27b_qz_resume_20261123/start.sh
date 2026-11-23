#!/usr/bin/env bash
set -euo pipefail
JOB="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$JOB/activate.sh"
# 1024 concurrent HTTP requests also need descriptors for logs and input/output.
ulimit -n 65536
mkdir -p "$JOB/logs"
LOG="$JOB/logs/generate.$(date +%Y%m%d_%H%M%S).log"
nohup "$PYTHON_BIN" -u "$JOB/launch.py" --parts 0,1 --reuse-smoke > "$LOG" 2>&1 < /dev/null &
echo "$!" > "$JOB/logs/controller.pid"
ln -sfn "$(basename "$LOG")" "$JOB/logs/latest.log"
echo "PID=$! LOG=$LOG"
