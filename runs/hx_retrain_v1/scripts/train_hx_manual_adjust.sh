#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SRC_DIR="${REPO_ROOT}/src/SongFormer"
export SONGFORMER_REPO_ROOT="${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
echo "Use GPU: ${CUDA_VISIBLE_DEVICES}"

export WANDB_MODE=disabled
export PYTHONPATH="${SRC_DIR}:${REPO_ROOT}/src/third_party:/home/hbli/songformer/repo/SongFormer/src/third_party:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
export MPI_NUM_THREADS=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/home/hbli/songformer/cache/hf_cache}"

CFG="${REPO_ROOT}/runs/hx_retrain_v1/configs/SongFormer_manual_adjust.yaml"
INIT_SEED=42

cd "${SRC_DIR}"
gpustat --id "${CUDA_VISIBLE_DEVICES}"

accelerate launch --config_file train/accelerate_config/single_gpu.yaml \
  train/train.py \
  --config "${CFG}" \
  --log_interval 5 \
  --init_seed "${INIT_SEED}"
