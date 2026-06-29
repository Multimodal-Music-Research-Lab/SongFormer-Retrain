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

HOOK_SCP=/mnt/ssd/hbli/songformer/runs/hx_train_hooktheory_full/results/hooktheory_test_full_unique_636.scp
RAW_ANN_DIR=/mnt/ssd/hbli/datasets/hooktheory/labels
PRED_DIR=${RUN}/results/bench_pred/hooktheory_test_full_636_${CKPT_TAG}
EST_DIR=${RUN}/results/eval/hooktheory_test_full_636_${CKPT_TAG}/est_txt
ANN_DIR=${RUN}/results/eval/hooktheory_test_full_636_${CKPT_TAG}/ann_txt_normalized
METRICS_DIR=${RUN}/results/eval/hooktheory_test_full_636_${CKPT_TAG}/metrics_partial

CANONICAL_THIRD_PARTY=/home/hbli/songformer/repo/SongFormer/src/third_party
export PYTHONPATH=${REPO}/src/SongFormer:${REPO}/src/third_party:${CANONICAL_THIRD_PARTY}:${PYTHONPATH:-}

if [ ! -d "${PRED_DIR}" ]; then
  echo "Missing existing HookTheory prediction directory: ${PRED_DIR}" >&2
  echo "Run eval_hooktheory_partial_alma_fusion_v8.sh first, or pass a checkpoint whose HookTheory predictions already exist." >&2
  exit 1
fi

cd "${REPO}/src/SongFormer"

python evaluation/prepare_hooktheory_partial_labels.py \
  --ann_dir "${RAW_ANN_DIR}" \
  --scp "${HOOK_SCP}" \
  --output_dir "${ANN_DIR}" \
  --prechorus2what verse \
  --composite_policy first \
  --allow_failed

python utils/convert_res2msa_txt.py \
  --input_folder "${PRED_DIR}" \
  --output_folder "${EST_DIR}"

python evaluation/eval_infer_partially_labeled_results.py \
  --ann_dir "${ANN_DIR}/" \
  --est_dir "${EST_DIR}/" \
  --output_dir "${METRICS_DIR}/" \
  --prechorus2what verse
