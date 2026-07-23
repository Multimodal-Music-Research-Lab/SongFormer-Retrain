#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SRC_DIR="${REPO_ROOT}/src/SongFormer"
export SONGFORMER_REPO_ROOT="${REPO_ROOT}"
CANONICAL_THIRD_PARTY="/home/hbli/songformer/repo/SongFormer/src/third_party"

CHECKPOINT="${1:-/mnt/ssd/hbli/songformer/runs/hx_retrain_v1/results/author_ckpt/SongFormer.safetensors}"
ANN_DIR="${2:-/mnt/ssd/hbli/datasets/songformer/songformbench/data/labels/HarmonixSet}"
EVAL_NAME="${3:-hx_author}"
INPUT_SCP="/mnt/ssd/hbli/songformer/runs/hx_retrain_v1/results/hx_bench.scp"
CONFIG="${REPO_ROOT}/runs/hx_retrain_v1/configs/SongFormer.yaml"
RESULT_ROOT="/mnt/ssd/hbli/songformer/runs/hx_retrain_v1/results"
PRED_DIR="${RESULT_ROOT}/bench_pred/${EVAL_NAME}"
EST_DIR="${RESULT_ROOT}/eval/${EVAL_NAME}/est_txt"
METRICS_DIR="${RESULT_ROOT}/eval/${EVAL_NAME}/metrics"

export PYTHONPATH="${SRC_DIR}:${REPO_ROOT}/src/third_party:${CANONICAL_THIRD_PARTY}:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/home/hbli/songformer/cache/hf_cache}"

cd "${REPO_ROOT}"
python "${SCRIPT_DIR}/infer.py" \
  -i "${INPUT_SCP}" \
  -o "${PRED_DIR}" \
  -gn 1 \
  -tn 1 \
  --model SongFormer \
  --checkpoint "${CHECKPOINT}" \
  --config_path "${CONFIG}"

cd "${SRC_DIR}"
python utils/convert_res2msa_txt.py \
  --input_folder "${PRED_DIR}" \
  --output_folder "${EST_DIR}"

python evaluation/eval_infer_results.py \
  --ann_dir "${ANN_DIR}" \
  --est_dir "${EST_DIR}" \
  --output_dir "${METRICS_DIR}" \
  --prechorus2what verse
