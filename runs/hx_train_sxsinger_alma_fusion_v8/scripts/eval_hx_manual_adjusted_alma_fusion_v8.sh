#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/home/hbli/songformer/cache/hf_cache}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "${SCRIPT_DIR}/../../.." && pwd)
RUN=/mnt/ssd/hbli/songformer/runs/hx_train_sxsinger_alma_fusion_v8
CKPT=${1:-${RUN}/results/train_output_42/model.ckpt-38400.pt}
CKPT_TAG=$(basename "${CKPT}")
CKPT_TAG=${CKPT_TAG%.pt}

ANN_DIR=/mnt/ssd/hbli/datasets/songformer/songformbench/data/labels/HarmonixSet_manual_adjusted
PRED_DIR=${RUN}/results/bench_pred/hx_${CKPT_TAG}
EST_DIR=${RUN}/results/eval/hx_${CKPT_TAG}/est_txt
METRICS_DIR=${RUN}/results/eval/hx_manual_adjusted_${CKPT_TAG}/metrics_per_class

CANONICAL_THIRD_PARTY=/home/hbli/songformer/repo/SongFormer/src/third_party
export PYTHONPATH=${REPO}/src/SongFormer:${REPO}/src/third_party:${CANONICAL_THIRD_PARTY}:${PYTHONPATH:-}

if [ ! -d "${EST_DIR}" ]; then
  if [ ! -d "${PRED_DIR}" ]; then
    echo "Missing existing prediction directory: ${PRED_DIR}" >&2
    echo "Run the normal benchmark infer/eval first, or pass a checkpoint whose hx predictions already exist." >&2
    exit 1
  fi
  cd "${REPO}/src/SongFormer"
  python utils/convert_res2msa_txt.py \
    --input_folder "${PRED_DIR}" \
    --output_folder "${EST_DIR}"
fi

cd "${REPO}/src/SongFormer"
python evaluation/eval_infer_results.py \
  --ann_dir "${ANN_DIR}/" \
  --est_dir "${EST_DIR}/" \
  --output_dir "${METRICS_DIR}/" \
  --prechorus2what verse
