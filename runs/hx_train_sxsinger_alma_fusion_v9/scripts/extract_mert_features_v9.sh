#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/home/hbli/songformer/cache/hf_cache}

REPO=/home/hbli/songformer/repo/SongFormer
CFG=${REPO}/runs/hx_train_sxsinger_alma_fusion_v9/configs/alma_v9.yaml
export PYTHONPATH=${REPO}/src/SongFormer:${REPO}/src/third_party:${PYTHONPATH:-}

SPLIT=${1:-all}
if [[ "${SPLIT}" == "all" ]]; then
  for item in train val hx_bench; do
    python -m alma_v9.extract_mert_features \
      --config "${CFG}" \
      --split "${item}" \
      --device cuda \
      --skip_existing
  done
else
  python -m alma_v9.extract_mert_features \
    --config "${CFG}" \
    --split "${SPLIT}" \
    --device cuda \
    --skip_existing
fi

