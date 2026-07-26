#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export WANDB_MODE=disabled
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/home/hbli/songformer/cache/hf_cache}
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
export MPI_NUM_THREADS=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "${SCRIPT_DIR}/../../.." && pwd)
CFG=${REPO}/runs/hx_train_hooktheory_v3/configs/SongFormer_hx_hookboundary_adjusted.yaml

cd "${REPO}/src/SongFormer"
export PYTHONPATH=$(realpath .):${PYTHONPATH:-}

gpustat --id "${CUDA_VISIBLE_DEVICES}"

accelerate launch --config_file train/accelerate_config/single_gpu.yaml \
  train/train.py \
  --config "${CFG}" \
  --log_interval 5 \
  --init_seed 42
