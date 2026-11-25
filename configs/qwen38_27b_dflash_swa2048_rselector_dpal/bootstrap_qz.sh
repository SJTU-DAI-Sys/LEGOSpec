#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TOOLS=/inspire/hdd/global_user/xujiaming-253308120313/tools
export CUDA_HOME="${TOOLS}/cuda-13.0"
export PATH="${CUDA_HOME}/bin:${TOOLS}/bin:${PATH}"
export UV_CACHE_DIR="${TOOLS}/uv-cache"
export UV_PROJECT_ENVIRONMENT="${PROJECT_DIR}/.venv"
export TORCH_CUDA_ARCH_LIST=9.0
export MAX_JOBS=4
cd "${PROJECT_DIR}"
uv sync --locked --python "${TOOLS}/python/cpython-3.11-linux-x86_64-gnu/bin/python3.11" --no-default-groups --group test --extra cu130 --extra hopper
