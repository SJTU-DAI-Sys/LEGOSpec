#!/usr/bin/env bash
set -euo pipefail
JOB=/mnt/dai-sys/wanghanzhen/projects/DAC27/LEGOSpec_public/tests/qwen38_27b_qz_resume_20261123
source "$JOB/activate.sh"
"$PYTHON_BIN" "$JOB/prepare_resume.py"
"$PYTHON_BIN" "$JOB/test_generation.py"
