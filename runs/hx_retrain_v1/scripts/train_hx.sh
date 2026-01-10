set -e

# ===== GPU =====
export CUDA_VISIBLE_DEVICES=0
echo "Use GPU: ${CUDA_VISIBLE_DEVICES}"

# DO NOT USE WANDB IN THIS RETRAIN
export WANDB_MODE=disabled
# export WANDB_API_KEY="YOUR_KEY"

# ===== Settings =====
cd /home/hbli/songformer/repo/SongFormer/src/SongFormer
export PYTHONPATH=$(realpath .):$PYTHONPATH

export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
export MPI_NUM_THREADS=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home/hbli/songformer/cache/hf_cache

# export TORCH_LOGS=attention

gpustat --id $CUDA_VISIBLE_DEVICES

CFG=/home/hbli/songformer/repo/SongFormer/runs/hx_retrain_v1/configs/SongFormer.yaml
INIT_SEED=42

# ===== Train =====
accelerate launch --config_file train/accelerate_config/single_gpu.yaml \
  train/train.py \
  --config "${CFG}" \
  --log_interval 5 \
  --init_seed "${INIT_SEED}"

