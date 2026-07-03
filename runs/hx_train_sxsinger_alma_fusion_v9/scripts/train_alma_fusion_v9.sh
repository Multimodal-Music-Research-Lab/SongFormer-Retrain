#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export WANDB_MODE=disabled
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/home/hbli/songformer/cache/hf_cache}
export OMP_NUM_THREADS=1

REPO=/home/hbli/songformer/repo/SongFormer
CFG=${REPO}/runs/hx_train_sxsinger_alma_fusion_v9/configs/alma_v9.yaml
export PYTHONPATH=${REPO}/src/SongFormer:${REPO}/src/third_party:${PYTHONPATH:-}

python -m alma_v9.train \
  --config "${CFG}" \
  --seed 42

