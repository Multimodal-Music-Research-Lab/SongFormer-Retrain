#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/home/hbli/songformer/cache/hf_cache}

REPO=/home/hbli/songformer/repo/SongFormer
export PYTHONPATH=${REPO}/src/SongFormer:${REPO}/src/third_party:${PYTHONPATH:-}

python -m alma_v9.download_models \
  --mert_model_id m-a-p/MERT-v1-95M \
  --mert_dir ${REPO}/src/third_party/mert/MERT-v1-95M \
  --tokenizer_model_id allenai/longformer-base-4096 \
  --tokenizer_dir ${REPO}/src/third_party/text_models/allenai_longformer-base-4096

