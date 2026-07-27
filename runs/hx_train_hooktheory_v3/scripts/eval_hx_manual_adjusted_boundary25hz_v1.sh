#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/home/hbli/songformer/cache/hf_cache}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "${SCRIPT_DIR}/../../.." && pwd)
RUN=/mnt/ssd/hbli/songformer/runs/hx_train_hooktheory_v3_boundary25hz_v1
CKPT=${1:-${RUN}/results/train_output_42/model.ckpt-12000.pt}
BOUNDARY_MODE=${2:-highres}

if [[ "${BOUNDARY_MODE}" != "highres" && "${BOUNDARY_MODE}" != "lowres" ]]; then
  echo "boundary mode must be highres or lowres" >&2
  exit 2
fi

CKPT_TAG=$(basename "${CKPT}")
CKPT_TAG=${CKPT_TAG%.pt}
CFG=${REPO}/runs/hx_train_hooktheory_v3/configs/SongFormer_boundary25hz_v1.yaml
SCP=/mnt/ssd/hbli/songformer/runs/hx_retrain_v1/results/hx_bench.scp
ANN_DIR=/mnt/ssd/hbli/datasets/songformer/songformbench/data/labels/HarmonixSet_manual_adjusted
PRED_DIR=${RUN}/results/bench_pred/hx_${BOUNDARY_MODE}_${CKPT_TAG}
EST_DIR=${RUN}/results/eval/hx_${BOUNDARY_MODE}_${CKPT_TAG}/est_txt
METRICS_DIR=${RUN}/results/eval/hx_manual_adjusted_${BOUNDARY_MODE}_${CKPT_TAG}/metrics_per_class

CANONICAL_THIRD_PARTY=/home/hbli/songformer/repo/SongFormer/src/third_party
export PYTHONPATH=${REPO}/src/SongFormer:${REPO}/src/third_party:${CANONICAL_THIRD_PARTY}:${PYTHONPATH:-}

python "${SCRIPT_DIR}/infer_boundary25hz.py" \
  -i "${SCP}" \
  -o "${PRED_DIR}" \
  -gn 1 \
  -tn 1 \
  --model SongFormer \
  --checkpoint "${CKPT}" \
  --config_path "${CFG}" \
  --boundary_mode "${BOUNDARY_MODE}"

cd "${REPO}/src/SongFormer"
python utils/convert_res2msa_txt.py \
  --input_folder "${PRED_DIR}" \
  --output_folder "${EST_DIR}"

python evaluation/eval_infer_results.py \
  --ann_dir "${ANN_DIR}/" \
  --est_dir "${EST_DIR}/" \
  --output_dir "${METRICS_DIR}/" \
  --prechorus2what verse

echo "Metrics: ${METRICS_DIR}"
