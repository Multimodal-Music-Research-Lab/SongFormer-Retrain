#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/home/hbli/songformer/cache/hf_cache}

REPO=/home/hbli/songformer/repo/SongFormer
RUN=/mnt/ssd/hbli/songformer/runs/hx_train_sxsinger_longformer_fusion_v1
CKPT=${1:-${RUN}/results/train_output_42/model.ckpt-12000.pt}
CFG=${REPO}/runs/hx_train_sxsinger_longformer_fusion_v1/configs/SongFormer.yaml
BENCH_SCP=/mnt/ssd/hbli/songformer/runs/hx_retrain_v1/results/hx_bench.scp
LYRICS_FEATURE_DIR=/mnt/ssd/hbli/songformer/runs/hx_train_sxsinger_longformer_probe_hook_mask_v1/results/songformer_line_features/bench

export PYTHONPATH=${REPO}/src/SongFormer:${REPO}/src/third_party:${PYTHONPATH:-}

python ${REPO}/runs/hx_train_sxsinger_longformer_fusion_v1/scripts/infer.py \
  -i ${BENCH_SCP} \
  -o ${RUN}/results/bench_pred/hx \
  -gn 1 -tn 1 \
  --model SongFormer \
  --checkpoint ${CKPT} \
  --config_path ${CFG} \
  --lyrics_embedding_dir ${LYRICS_FEATURE_DIR}

cd ${REPO}/src/SongFormer

python utils/convert_res2msa_txt.py \
  --input_folder ${RUN}/results/bench_pred/hx \
  --output_folder ${RUN}/results/eval/hx/est_txt

python evaluation/eval_infer_results.py \
  --ann_dir /mnt/ssd/hbli/datasets/songformer/songformbench/data/labels/HarmonixSet/ \
  --est_dir ${RUN}/results/eval/hx/est_txt/ \
  --output_dir ${RUN}/results/eval/hx/metrics/ \
  --prechorus2what verse
