#!/usr/bin/env bash
export LEGOSPEC_ROOT=/mnt/dai-sys/wanghanzhen/projects/DAC27/LEGOSpec_public
export QWEN38_REGEN_JOB="$LEGOSPEC_ROOT/tests/qwen38_27b_qz_resume_20261123"
export PYTHON_BIN=/mnt/dai-sys/wanghanzhen/projects/DAC27/envs/legospec-public-sglang0521-py311-cu130/bin/python
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$(dirname "$PYTHON_BIN"):$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
unset CUDA_LAUNCH_BLOCKING
export PYTHONPATH="$QWEN38_REGEN_JOB/backend_source"

export CUDA_VISIBLE_DEVICES=0,1,4,5,2,3
