#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/home/hbli/songformer/cache/hf_cache}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "${SCRIPT_DIR}/../../.." && pwd)
RUN=/mnt/ssd/hbli/songformer/runs/hx_train_hooktheory_v3_manual50_v1
CKPT=${1:-${RUN}/results/train_output_42/model.ckpt-12000.pt}
CKPT_TAG=$(basename "${CKPT}")
CKPT_TAG=${CKPT_TAG%.pt}
RESULT_TAG=${2:-${CKPT_TAG}}
CFG=${REPO}/runs/hx_train_hooktheory_v3_manual50_v1/configs/SongFormer.yaml
PRED_DIR=${RUN}/results/bench_pred/hooktheory_manual_heldout50_${RESULT_TAG}
EST_DIR=${RUN}/results/eval/hooktheory_manual_heldout50_${RESULT_TAG}/est_txt
METRICS_DIR=${RUN}/results/eval/hooktheory_manual_heldout50_${RESULT_TAG}/metrics_per_class
ANN_DIR=${RUN}/results/eval/hooktheory_manual_heldout50_${RESULT_TAG}/ann_txt_normalized
HELDOUT_SCP=$(mktemp --suffix=.scp)
trap 'rm -f "${HELDOUT_SCP}"' EXIT

CANONICAL_THIRD_PARTY=/home/hbli/songformer/repo/SongFormer/src/third_party
export PYTHONPATH=${REPO}/src/SongFormer:${REPO}/src/third_party:${CANONICAL_THIRD_PARTY}:${PYTHONPATH:-}

python "${REPO}/runs/hx_train_hooktheory_v3_manual50_v1/tools/sample_manual_full_train50.py" \
  --split heldout \
  --output "${HELDOUT_SCP}" \
  --normalized_label_dir "${ANN_DIR}"

if [ ! -d "${PRED_DIR}" ]; then
  python "${REPO}/runs/hx_train_hooktheory_v3/scripts/infer_hooktheory.py" \
    -i "${HELDOUT_SCP}" \
    -o "${PRED_DIR}" \
    -gn 1 -tn 1 \
    --model SongFormer \
    --checkpoint "${CKPT}" \
    --config_path "${CFG}"
fi

cd "${REPO}/src/SongFormer"
python utils/convert_res2msa_txt.py \
  --input_folder "${PRED_DIR}" \
  --output_folder "${EST_DIR}"

python evaluation/eval_infer_results.py \
  --ann_dir "${ANN_DIR}/" \
  --est_dir "${EST_DIR}/" \
  --output_dir "${METRICS_DIR}/" \
  --prechorus2what verse
