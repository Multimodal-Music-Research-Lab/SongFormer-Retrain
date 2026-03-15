#!/usr/bin/env bash
set -euo pipefail

# 仓库根目录
export REPO_ROOT="/home/hbli/songformer/repo/SongFormer"

# 数据根目录（SSD）
export DATA_ROOT="/mnt/ssd/hbli/datasets/songformer"

# 训练数据（SongFormDB-HX）
export HX_DB_LABELS="/mnt/ssd/hbli/datasets/songformer/songformbench/data/labels"
export HX_DB_MELS="/mnt/ssd/hbli/datasets/songformer/songformbench/data/mels"
export HX_DB_AUDIOS="/mnt/ssd/hbli/datasets/songformer/songformbench/data/audios"

# 验证/评测数据（SongFormBench）
export BENCH_CN_LABELS="/mnt/ssd/hbli/datasets/songformer/songformbench/data/labels/CN"
export BENCH_CN_MELS="/mnt/ssd/hbli/datasets/songformer/songformbench/data/mels/CN"
export BENCH_CN_AUDIOS="/mnt/ssd/hbli/datasets/songformer/songformbench/data/audios/CN"

export BENCH_HX_LABELS="/mnt/ssd/hbli/datasets/songformer/songformbench/data/labels/HarmonixSet"
export BENCH_HX_MELS="/mnt/ssd/hbli/datasets/songformer/songformbench/data/mels/HarmonixSet"
export BENCH_HX_AUDIOS="/mnt/ssd/hbli/datasets/songformer/songformbench/data/audios/HarmonixSet"

# SSL embedding 输出目录（你不重训 SSL，但要生成/缓存 embedding）
export SSL_ROOT="${DATA_ROOT}/ssl/hx"

# 本次实验输出（checkpoint / metrics 等建议放 SSD）
export EXP_NAME="hx_retrain_v1"
export EXP_OUT="${DATA_ROOT}/outputs/${EXP_NAME}"

# 单卡
export CUDA_VISIBLE_DEVICES=0

mkdir -p "${SSL_ROOT}" "${EXP_OUT}"
