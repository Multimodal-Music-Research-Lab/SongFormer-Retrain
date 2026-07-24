#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: CUDA_VISIBLE_DEVICES=0 $0 /absolute/path/to/model.ckpt-N.pt [config.yaml]"
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SRC_DIR="${REPO_ROOT}/src/SongFormer"
CANONICAL_THIRD_PARTY="/home/hbli/songformer/repo/SongFormer/src/third_party"
export SONGFORMER_REPO_ROOT="${REPO_ROOT}"

CHECKPOINT="$(realpath "$1")"
CONFIG="${2:-${REPO_ROOT}/runs/hx_retrain_v1/configs/SongFormer.yaml}"
CONFIG="$(realpath "${CONFIG}")"
CHECKPOINT_DIR="$(dirname "${CHECKPOINT}")"
RESULTS_DIR="$(dirname "${CHECKPOINT_DIR}")"
CKPT_TAG="$(basename "${CHECKPOINT}")"
CKPT_TAG="${CKPT_TAG%.pt}"
CKPT_TAG="${CKPT_TAG%.safetensors}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${SRC_DIR}:${REPO_ROOT}/src/third_party:${CANONICAL_THIRD_PARTY}:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/home/hbli/songformer/cache/hf_cache}"

run_benchmark() {
  local name="$1"
  local input_scp="$2"
  local ann_dir="$3"
  local pred_dir="${RESULTS_DIR}/bench_pred/${name}_${CKPT_TAG}"
  local est_dir="${RESULTS_DIR}/eval/${name}_${CKPT_TAG}/est_txt"
  local metrics_dir="${RESULTS_DIR}/eval/${name}_${CKPT_TAG}/metrics"

  cd "${REPO_ROOT}"
  python "${SCRIPT_DIR}/infer.py" \
    -i "${input_scp}" \
    -o "${pred_dir}" \
    -gn 1 \
    -tn 1 \
    --model SongFormer \
    --checkpoint "${CHECKPOINT}" \
    --config_path "${CONFIG}"

  cd "${SRC_DIR}"
  python utils/convert_res2msa_txt.py \
    --input_folder "${pred_dir}" \
    --output_folder "${est_dir}"

  python evaluation/eval_infer_results.py \
    --ann_dir "${ann_dir}" \
    --est_dir "${est_dir}" \
    --output_dir "${metrics_dir}" \
    --prechorus2what verse
}

run_benchmark \
  hx_manual_adjusted \
  /mnt/ssd/hbli/songformer/runs/hx_retrain_v1/results/hx_bench.scp \
  /mnt/ssd/hbli/datasets/songformer/songformbench/data/labels/HarmonixSet_manual_adjusted

run_benchmark \
  cn_manual_adjusted \
  /mnt/ssd/hbli/songformer/runs/hx_retrain_v1/results/cn_bench.scp \
  /mnt/ssd/hbli/datasets/songformer/songformbench/data/labels/CN_manual_adjusted
