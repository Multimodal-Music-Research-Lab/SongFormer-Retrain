#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export WANDB_MODE=disabled
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/home/hbli/songformer/cache/hf_cache}
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
export MPI_NUM_THREADS=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "${SCRIPT_DIR}/../../.." && pwd)
RUN=/mnt/ssd/hbli/songformer/runs/hx_train_hooktheory_v3_manual50_ext_adjusted_v1
CFG=${REPO}/runs/hx_train_hooktheory_v3_manual50_ext_adjusted_v1/configs/SongFormer.yaml
HOOK_ADJUSTED=/mnt/ssd/hbli/datasets/hooktheory/HookTheory-user-madmom-adjusted-v3-compatible_for_model.jsonl
EXT_ADJUSTED=/mnt/ssd/hbli/datasets/songformer/songformdb/data/Ext/SongFormDB-Ext-adjust-local-restriction_for_model.jsonl

for required_path in "${HOOK_ADJUSTED}" "${EXT_ADJUSTED}"; do
  if [ ! -s "${required_path}" ]; then
    echo "Missing adjusted training labels: ${required_path}" >&2
    exit 1
  fi
done

python "${REPO}/runs/hx_train_hooktheory_v3_manual50_ext_adjusted_v1/tools/prepare_ext_train_split.py"
python "${REPO}/runs/hx_train_hooktheory_v3_manual50_ext_adjusted_v1/tools/sample_manual_full_train50.py" \
  --output "${RUN}/data/hooktheory_manual_full_train50_seed42.scp"

cd "${REPO}/src/SongFormer"
export PYTHONPATH=$(realpath .):${PYTHONPATH:-}

echo "Use GPU: ${CUDA_VISIBLE_DEVICES}"
gpustat --id "${CUDA_VISIBLE_DEVICES}"

accelerate launch --config_file train/accelerate_config/single_gpu.yaml \
  train/train.py \
  --config "${CFG}" \
  --log_interval 5 \
  --init_seed 42
