#!/bin/bash
#SBATCH --job-name=finetune_sgd_lr1e5_wd01_alldata
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-6
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err

set -euo pipefail
mkdir -p artifacts/logs

# 1. Setup environment
source ".venv-vl/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs
# wandb logging is on (USE_WANDB defaults to 1), online mode. Each array task is
# its own run, so there is no shared-store contention like trackio had.
DATA_DIR="data/vision"
OPENCLIP_DIR="$SCRATCH/openclip"

# 2. Stage datasets to $SLURM_TMPDIR (mirrors finetune.sh)
if [ ! -d "$SLURM_TMPDIR/data" ]; then
  cp downloads/data.tar.gz "$SLURM_TMPDIR/"
  tar -xzf "$SLURM_TMPDIR/data.tar.gz" -C "$SLURM_TMPDIR/"
fi
ln -sfn "$SLURM_TMPDIR/data" data

# 3. SGD lr=1e-5 wd=0.1 on the other 7 datasets (Cars already done in this dir).
#    No early stopping / 1x epochs, to match the existing Cars expert there.
MODEL="ViT-B-16"
FT_MODE="standard"
OPTIMIZER="sgd"
LR="1e-5"
WD="0.1"
DATASETS=(DTD EuroSAT GTSRB MNIST RESISC45 SUN397 SVHN)
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
SAVE_DIR="artifacts/checkpoints-sgd-lr1e-5"
# Unique DDP port per task so co-located array tasks don't collide.
PORT=$((12355 + SLURM_ARRAY_TASK_ID))

echo "[BASH] SGD lr1e-5 wd0.1 | model: $MODEL | optimizer: $OPTIMIZER | lr: $LR | wd: $WD | dataset: $DATASET | save dir: $SAVE_DIR"
python scripts/vision/finetune.py \
  --finetuning-mode="$FT_MODE" \
  --model="$MODEL" \
  --world-size=1 \
  --num-workers=1 \
  --cache-dir="$OPENCLIP_DIR" \
  --data-location="$DATA_DIR" \
  --optimizer="$OPTIMIZER" \
  --lr="$LR" \
  --wd="$WD" \
  --train-dataset="$DATASET" \
  --port="$PORT" \
  --save="$SAVE_DIR"
