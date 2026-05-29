#!/bin/bash
#SBATCH --job-name=finetune_sgd_lr_sweep
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err

set -euo pipefail
mkdir -p artifacts/logs

# 1. Setup environment
source ".venv-vl/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs
# Disable trackio: concurrent array tasks hang on the shared NFS SQLite store.
export USE_TRACKIO=0
DATA_DIR="data/vision"
OPENCLIP_DIR="$SCRATCH/openclip"

# 2. Stage datasets to $SLURM_TMPDIR (mirrors finetune.sh)
if [ ! -d "$SLURM_TMPDIR/data" ]; then
  cp downloads/data.tar.gz "$SLURM_TMPDIR/"
  tar -xzf "$SLURM_TMPDIR/data.tar.gz" -C "$SLURM_TMPDIR/"
fi
ln -sfn "$SLURM_TMPDIR/data" data

# 3. SGD learning-rate sweep on Cars only (worst expert gap vs AdamW: -24%).
#    1e-4 under-trains (test 63.55%), 1e-3 diverged — sweet spot is in between.
MODEL="ViT-B-16"
FT_MODE="standard"
OPTIMIZER="sgd"
DATASET="Cars"
LRS=(3e-4 5e-4 7e-4 1e-3)
LR="${LRS[$SLURM_ARRAY_TASK_ID]}"
SAVE_DIR="artifacts/checkpoints-sgd-lr${LR}"
# Unique DDP port per task so co-located array tasks don't collide.
PORT=$((12355 + SLURM_ARRAY_TASK_ID))

echo "[BASH] LR sweep | model: $MODEL | optimizer: $OPTIMIZER | lr: $LR | dataset: $DATASET | save dir: $SAVE_DIR"
python scripts/vision/finetune.py \
  --finetuning-mode="$FT_MODE" \
  --model="$MODEL" \
  --world-size=1 \
  --num-workers=1 \
  --cache-dir="$OPENCLIP_DIR" \
  --data-location="$DATA_DIR" \
  --optimizer="$OPTIMIZER" \
  --lr="$LR" \
  --train-dataset="$DATASET" \
  --port="$PORT" \
  --save="$SAVE_DIR"
