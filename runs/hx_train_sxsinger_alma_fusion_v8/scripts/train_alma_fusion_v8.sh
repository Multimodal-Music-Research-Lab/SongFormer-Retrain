#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
echo "Use GPU: ${CUDA_VISIBLE_DEVICES}"

export WANDB_MODE=disabled
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
export MPI_NUM_THREADS=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/home/hbli/songformer/cache/hf_cache}

cd /home/hbli/songformer/repo/SongFormer/src/SongFormer
export PYTHONPATH=$(realpath .):/home/hbli/songformer/repo/SongFormer/src/third_party:${PYTHONPATH:-}

if command -v gpustat >/dev/null 2>&1; then
  gpustat --id "${CUDA_VISIBLE_DEVICES}"
else
  echo "gpustat not found; skip GPU status print"
fi

CFG=/home/hbli/songformer/repo/SongFormer/runs/hx_train_sxsinger_alma_fusion_v8/configs/SongFormer.yaml
INIT_SEED=42

/home/hbli/songformer/env/miniforge3/envs/songformer/bin/accelerate launch --config_file train/accelerate_config/single_gpu.yaml \
  train/train.py \
  --config "${CFG}" \
  --log_interval 5 \
  --init_seed "${INIT_SEED}"
