#!/bin/bash
#SBATCH --job-name=finetune_sgd_2x
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err

set -euo pipefail
mkdir -p artifacts/logs

# 1. Setup environment
source ".venv-vl/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs
# Trackio off: concurrent array tasks hang on the shared NFS SQLite store.
export USE_TRACKIO=0
DATA_DIR="data/vision"
OPENCLIP_DIR="$SCRATCH/openclip"

# 2. Stage datasets to $SLURM_TMPDIR (mirrors finetune.sh)
if [ ! -d "$SLURM_TMPDIR/data" ]; then
  cp downloads/data.tar.gz "$SLURM_TMPDIR/"
  tar -xzf "$SLURM_TMPDIR/data.tar.gz" -C "$SLURM_TMPDIR/"
fi
ln -sfn "$SLURM_TMPDIR/data" data

# 3. Re-train the 8 SGD experts at 2x epochs WITH early stopping (save best-val
#    model, not the final one). Same recipe as checkpoints-sgd otherwise:
#    SGD lr=1e-4, wd=0.1. New dir keeps the 1x baseline intact.
MODEL="ViT-B-16"
FT_MODE="standard"
OPTIMIZER="sgd"
LR="1e-4"
SAVE_DIR="artifacts/checkpoints-sgd-2x"
DATASETS=(Cars DTD EuroSAT GTSRB MNIST RESISC45 SUN397 SVHN)
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
# Unique DDP port per task so co-located array tasks don't collide.
PORT=$((12355 + SLURM_ARRAY_TASK_ID))

echo "[BASH] SGD 2x + early-stop | model: $MODEL | optimizer: $OPTIMIZER | lr: $LR | dataset: $DATASET | save dir: $SAVE_DIR"
python scripts/vision/finetune.py \
  --finetuning-mode="$FT_MODE" \
  --model="$MODEL" \
  --world-size=1 \
  --num-workers=1 \
  --cache-dir="$OPENCLIP_DIR" \
  --data-location="$DATA_DIR" \
  --optimizer="$OPTIMIZER" \
  --lr="$LR" \
  --epochs-mult=2.0 \
  --early-stop \
  --patience=10 \
  --train-dataset="$DATASET" \
  --port="$PORT" \
  --save="$SAVE_DIR"
