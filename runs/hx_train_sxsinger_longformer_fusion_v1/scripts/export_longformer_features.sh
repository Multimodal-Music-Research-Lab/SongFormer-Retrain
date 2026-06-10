#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/home/hbli/songformer/cache/hf_cache}

cd /home/hbli/songformer/repo/SongFormer
source /home/hbli/songformer/env/miniforge3/etc/profile.d/conda.sh
conda activate songformer

LONGFORMER_CFG=/home/hbli/songformer/repo/SongFormer/runs/hx_train_sxsinger_longformer_probe_hook_mask_v1/configs/lyrics_longformer.yaml
LONGFORMER_CKPT=/mnt/ssd/hbli/songformer/runs/hx_train_sxsinger_longformer_probe_hook_mask_v1/results/best.pt
FEATURE_ROOT=/mnt/ssd/hbli/songformer/runs/hx_train_sxsinger_longformer_probe_hook_mask_v1/results/songformer_line_features

python src/lyrics_only_longformer/run_lyrics_longformer.py \
  --config "${LONGFORMER_CFG}" \
  --mode export_features \
  --split train \
  --ckpt "${LONGFORMER_CKPT}" \
  --output_dir "${FEATURE_ROOT}/train"

python src/lyrics_only_longformer/run_lyrics_longformer.py \
  --config "${LONGFORMER_CFG}" \
  --mode export_features \
  --split val \
  --ckpt "${LONGFORMER_CKPT}" \
  --output_dir "${FEATURE_ROOT}/val"

python src/lyrics_only_longformer/run_lyrics_longformer.py \
  --config "${LONGFORMER_CFG}" \
  --mode export_features \
  --split bench \
  --ckpt "${LONGFORMER_CKPT}" \
  --output_dir "${FEATURE_ROOT}/bench"
