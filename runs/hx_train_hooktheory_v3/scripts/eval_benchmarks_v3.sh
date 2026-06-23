#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/home/hbli/songformer/cache/hf_cache}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "${SCRIPT_DIR}/../../.." && pwd)
RUN=/mnt/ssd/hbli/songformer/runs/hx_train_hooktheory_v3
CKPT=${1:-${RUN}/results/train_output_42/model.ckpt-12000.pt}
CKPT_TAG=$(basename "${CKPT}")
CKPT_TAG=${CKPT_TAG%.pt}
CFG=${REPO}/runs/hx_train_hooktheory_v3/configs/SongFormer.yaml

export PYTHONPATH=${REPO}/src/SongFormer:${REPO}/src/third_party:${PYTHONPATH:-}

run_full_eval() {
  local name=$1
  local scp=$2
  local ann_dir=$3
  local pred_dir=${RUN}/results/bench_pred/${name}_${CKPT_TAG}
  local est_dir=${RUN}/results/eval/${name}_${CKPT_TAG}/est_txt
  local metrics_dir=${RUN}/results/eval/${name}_${CKPT_TAG}/metrics_per_class

  python ${REPO}/runs/hx_train_hooktheory_v3/scripts/infer.py \
    -i "${scp}" \
    -o "${pred_dir}" \
    -gn 1 -tn 1 \
    --model SongFormer \
    --checkpoint "${CKPT}" \
    --config_path "${CFG}"

  cd ${REPO}/src/SongFormer

  python utils/convert_res2msa_txt.py \
    --input_folder "${pred_dir}" \
    --output_folder "${est_dir}"

  python evaluation/eval_infer_results.py \
    --ann_dir "${ann_dir}/" \
    --est_dir "${est_dir}/" \
    --output_dir "${metrics_dir}/" \
    --prechorus2what verse
}

run_full_eval \
  hx \
  /mnt/ssd/hbli/songformer/runs/hx_retrain_v1/results/hx_bench.scp \
  /mnt/ssd/hbli/datasets/songformer/songformbench/data/labels/HarmonixSet

run_full_eval \
  cn \
  /mnt/ssd/hbli/songformer/runs/hx_retrain_v1/results/cn_bench.scp \
  /mnt/ssd/hbli/datasets/songformer/songformbench/data/labels/CN
