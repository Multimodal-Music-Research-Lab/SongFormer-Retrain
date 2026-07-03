#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/home/hbli/songformer/cache/hf_cache}

REPO=/home/hbli/songformer/repo/SongFormer
RUN=/mnt/ssd/hbli/songformer/runs/hx_train_sxsinger_alma_fusion_v9
CFG=${REPO}/runs/hx_train_sxsinger_alma_fusion_v9/configs/alma_v9.yaml
CKPT=${1:-${RUN}/results/train_output_42/model.ckpt-48000.pt}
TAG=$(basename "${CKPT}")
TAG=${TAG%.pt}

BENCH_SCP=/mnt/ssd/hbli/songformer/runs/hx_retrain_v1/results/hx_bench.scp
LYRICS_DIR=/mnt/ssd/hbli/datasets/songformer/songformbench/data/soulxsinger_lyrics/HarmonixSet
PRED_TXT=${RUN}/results/eval/hx_${TAG}/est_txt
METRICS=${RUN}/results/eval/hx_${TAG}/metrics_per_class

export PYTHONPATH=${REPO}/src/SongFormer:${REPO}/src/third_party:${PYTHONPATH:-}

python -m alma_v9.infer \
  --config "${CFG}" \
  --checkpoint "${CKPT}" \
  --scp "${BENCH_SCP}" \
  --lyrics_dir "${LYRICS_DIR}" \
  --output_dir "${PRED_TXT}"

cd ${REPO}/src/SongFormer
python evaluation/eval_infer_results.py \
  --ann_dir /mnt/ssd/hbli/datasets/songformer/songformbench/data/labels/HarmonixSet/ \
  --est_dir "${PRED_TXT}/" \
  --output_dir "${METRICS}/" \
  --prechorus2what verse

