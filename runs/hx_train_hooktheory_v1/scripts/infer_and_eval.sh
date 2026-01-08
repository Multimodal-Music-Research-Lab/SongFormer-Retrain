#!/usr/bin/env bash
set -e

# ===== GPU =====
export CUDA_VISIBLE_DEVICES=1
echo "Use GPU: ${CUDA_VISIBLE_DEVICES}"

# ===== CN inference =====
python /home/hbli/songformer/repo/SongFormer/runs/hx_train_hooktheory_v1/scripts/infer.py \
  -i /home/hbli/songformer/repo/SongFormer/runs/hx_retrain_v1/results/cn_bench.scp \
  -o /mnt/ssd/hbli/songformer/runs/hx_train_hooktheory_v1/results/bench_pred/cn \
  -gn 1 -tn 1 \
  --model SongFormer \
  --checkpoint /mnt/ssd/hbli/songformer/runs/hx_train_hooktheory_v1/results/train_output_42/model.ckpt-12000.pt \
  --config_path /home/hbli/songformer/repo/SongFormer/runs/hx_train_hooktheory_v1/configs/SongFormer.yaml

# ===== HX inference =====
python /home/hbli/songformer/repo/SongFormer/runs/hx_train_hooktheory_v1/scripts/infer.py \
  -i /home/hbli/songformer/repo/SongFormer/runs/hx_retrain_v1/results/hx_bench.scp \
  -o /mnt/ssd/hbli/songformer/runs/hx_train_hooktheory_v1/results/bench_pred/hx \
  -gn 1 -tn 1 \
  --model SongFormer \
  --checkpoint /mnt/ssd/hbli/songformer/runs/hx_train_hooktheory_v1/results/train_output_42/model.ckpt-12000.pt \
  --config_path /home/hbli/songformer/repo/SongFormer/runs/hx_train_hooktheory_v1/configs/SongFormer.yaml

# ===== HookTheory inference =====
python /home/hbli/songformer/repo/SongFormer/runs/hx_train_hooktheory_v1/scripts/infer_hooktheory.py \
  -i /home/hbli/songformer/repo/SongFormer/runs/hx_train_hooktheory_v2/results/hooktheory_test.scp \
  -o /mnt/ssd/hbli/songformer/runs/hx_train_hooktheory_v1/results/bench_pred/hooktheory1000_measure \
  -gn 1 -tn 1 \
  --model SongFormer \
  --checkpoint /mnt/ssd/hbli/songformer/runs/hx_train_hooktheory_v1/results/train_output_42/model.ckpt-12000.pt \
  --config_path /home/hbli/songformer/repo/SongFormer/runs/hx_train_hooktheory_v1/configs/SongFormer.yaml

# ===== CN evaluation =====
python utils/convert_res2msa_txt.py \
  --input_folder /mnt/ssd/hbli/songformer/runs/hx_train_hooktheory_v1/results/bench_pred/cn \
  --output_folder /mnt/ssd/hbli/songformer/runs/hx_train_hooktheory_v1/results/eval/cn/est_txt

python /home/hbli/songformer/repo/SongFormer/src/SongFormer/evaluation/eval_infer_results.py \
  --ann_dir /mnt/ssd/hbli/datasets/songformer/songformbench/data/labels/CN/ \
  --est_dir /mnt/ssd/hbli/songformer/runs/hx_train_hooktheory_v1/results/eval/cn/est_txt/ \
  --output_dir /mnt/ssd/hbli/songformer/runs/hx_train_hooktheory_v1/results/eval/cn/metrics/ \
  --prechorus2what verse

# ===== HX evaluation =====
python utils/convert_res2msa_txt.py \
  --input_folder /mnt/ssd/hbli/songformer/runs/hx_train_hooktheory_v1/results/bench_pred/hx \
  --output_folder /mnt/ssd/hbli/songformer/runs/hx_train_hooktheory_v1/results/eval/hx/est_txt

python /home/hbli/songformer/repo/SongFormer/src/SongFormer/evaluation/eval_infer_results.py \
  --ann_dir /mnt/ssd/hbli/datasets/songformer/songformbench/data/labels/HarmonixSet/ \
  --est_dir /mnt/ssd/hbli/songformer/runs/hx_train_hooktheory_v1/results/eval/hx/est_txt/ \
  --output_dir /mnt/ssd/hbli/songformer/runs/hx_train_hooktheory_v1/results/eval/hx/metrics/ \
  --prechorus2what verse