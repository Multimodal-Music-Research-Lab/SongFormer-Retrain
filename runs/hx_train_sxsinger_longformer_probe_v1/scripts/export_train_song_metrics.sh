#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/home/hbli/songformer/cache/hf_cache}

cd /home/hbli/songformer/repo/SongFormer
source /home/hbli/songformer/env/miniforge3/etc/profile.d/conda.sh
conda activate songformer

python src/lyrics_only_longformer/run_lyrics_longformer.py \
  --config runs/hx_train_sxsinger_longformer_probe_v1/configs/lyrics_longformer.yaml \
  --mode infer_split_segments \
  --split train \
  --ckpt /mnt/ssd/hbli/songformer/runs/hx_train_sxsinger_longformer_probe_v1/results/best.pt

cd /home/hbli/songformer/repo/SongFormer/src/SongFormer
export PYTHONPATH=/home/hbli/songformer/repo/SongFormer/src/SongFormer:/home/hbli/songformer/repo/SongFormer/src/third_party:${PYTHONPATH:-}

python utils/convert_res2msa_txt.py \
  --input_folder /mnt/ssd/hbli/songformer/runs/hx_train_sxsinger_longformer_probe_v1/results/train_pred/json \
  --output_folder /mnt/ssd/hbli/songformer/runs/hx_train_sxsinger_longformer_probe_v1/results/eval/train/est_txt

python evaluation/eval_infer_results.py \
  --ann_dir /mnt/ssd/hbli/songformer/runs/hx_train_sxsinger_longformer_probe_v1/results/eval/train/ann_txt \
  --est_dir /mnt/ssd/hbli/songformer/runs/hx_train_sxsinger_longformer_probe_v1/results/eval/train/est_txt \
  --output_dir /mnt/ssd/hbli/songformer/runs/hx_train_sxsinger_longformer_probe_v1/results/eval/train/metrics \
  --prechorus2what verse
