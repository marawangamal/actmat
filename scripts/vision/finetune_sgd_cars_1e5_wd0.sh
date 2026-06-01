#!/bin/bash
#SBATCH --job-name=finetune_sgd_cars_1e5_wd0
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=artifacts/logs/%x_%j.out
#SBATCH --error=artifacts/logs/%x_%j.err

set -euo pipefail
mkdir -p artifacts/logs

# 1. Setup environment
source ".venv-vl/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs
export USE_TRACKIO=0
DATA_DIR="data/vision"
OPENCLIP_DIR="$SCRATCH/openclip"

# 2. Stage datasets to $SLURM_TMPDIR (mirrors finetune.sh)
if [ ! -d "$SLURM_TMPDIR/data" ]; then
  cp downloads/data.tar.gz "$SLURM_TMPDIR/"
  tar -xzf "$SLURM_TMPDIR/data.tar.gz" -C "$SLURM_TMPDIR/"
fi
ln -sfn "$SLURM_TMPDIR/data" data

# 3. SGD on Cars, lr 1e-5, NO weight decay (wd=0) — vs the wd=0.1 run (63.88% val).
MODEL="ViT-B-16"
FT_MODE="standard"
OPTIMIZER="sgd"
DATASET="Cars"
LR="1e-5"
WD="0.0"
SAVE_DIR="artifacts/checkpoints-sgd-lr1e5-wd0"

echo "[BASH] SGD Cars | model: $MODEL | optimizer: $OPTIMIZER | lr: $LR | wd: $WD | save dir: $SAVE_DIR"
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
  --save="$SAVE_DIR"
