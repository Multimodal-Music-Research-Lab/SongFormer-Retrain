#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CKPT=${1:-/mnt/ssd/hbli/songformer/runs/hx_train_hooktheory_v3_boundary25hz_v1/results/train_output_42/model.ckpt-12000.pt}

bash "${SCRIPT_DIR}/eval_hx_manual_adjusted_boundary25hz_v1.sh" "${CKPT}" highres
bash "${SCRIPT_DIR}/eval_hx_manual_adjusted_boundary25hz_v1.sh" "${CKPT}" lowres
