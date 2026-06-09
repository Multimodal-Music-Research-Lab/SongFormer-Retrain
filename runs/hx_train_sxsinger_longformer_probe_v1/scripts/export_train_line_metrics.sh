#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/home/hbli/songformer/cache/hf_cache}

cd /home/hbli/songformer/repo/SongFormer
source /home/hbli/songformer/env/miniforge3/etc/profile.d/conda.sh
conda activate songformer

python src/lyrics_only_longformer/run_lyrics_longformer.py \
  --config runs/hx_train_sxsinger_longformer_probe_v1/configs/lyrics_longformer.yaml \
  --mode infer_split \
  --split train \
  --ckpt /mnt/ssd/hbli/songformer/runs/hx_train_sxsinger_longformer_probe_v1/results/best.pt
