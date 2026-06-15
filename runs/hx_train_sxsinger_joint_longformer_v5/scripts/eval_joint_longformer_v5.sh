#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/home/hbli/songformer/cache/hf_cache}

REPO=/home/hbli/songformer/repo/SongFormer
RUN=/mnt/ssd/hbli/songformer/runs/hx_train_sxsinger_joint_longformer_v5
CKPT=${1:-${RUN}/results/train_output_42/model.ckpt-48000.pt}
CKPT_TAG=$(basename "${CKPT}")
CKPT_TAG=${CKPT_TAG%.pt}
CFG=${REPO}/runs/hx_train_sxsinger_joint_longformer_v5/configs/SongFormer.yaml
BENCH_SCP=/mnt/ssd/hbli/songformer/runs/hx_retrain_v1/results/hx_bench.scp
LYRICS_FEATURE_DIR=/mnt/ssd/hbli/songformer/runs/hx_train_sxsinger_longformer_probe_hook_mask_v1/results/songformer_line_features/bench
LYRICS_JSON_DIR=/mnt/ssd/hbli/datasets/songformer/songformbench/data/soulxsinger_lyrics/HarmonixSet
PRED_DIR=${RUN}/results/bench_pred/hx_${CKPT_TAG}
EST_DIR=${RUN}/results/eval/hx_${CKPT_TAG}/est_txt
METRICS_DIR=${RUN}/results/eval/hx_${CKPT_TAG}/metrics_per_class

export PYTHONPATH=${REPO}/src/SongFormer:${REPO}/src/third_party:${PYTHONPATH:-}

python ${REPO}/runs/hx_train_sxsinger_joint_longformer_v5/scripts/infer.py \
  -i ${BENCH_SCP} \
  -o ${PRED_DIR} \
  -gn 1 -tn 1 \
  --model SongFormer \
  --checkpoint ${CKPT} \
  --config_path ${CFG} \
  --lyrics_embedding_dir ${LYRICS_FEATURE_DIR} \
  --lyrics_json_dir ${LYRICS_JSON_DIR}

cd ${REPO}/src/SongFormer

python utils/convert_res2msa_txt.py \
  --input_folder ${PRED_DIR} \
  --output_folder ${EST_DIR}

python evaluation/eval_infer_results.py \
  --ann_dir /mnt/ssd/hbli/datasets/songformer/songformbench/data/labels/HarmonixSet/ \
  --est_dir ${EST_DIR}/ \
  --output_dir ${METRICS_DIR}/ \
  --prechorus2what verse
