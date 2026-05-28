#!/bin/bash
#SBATCH --job-name=covariance_analysisv2
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=artifacts/logs/%x_%j.out
#SBATCH --error=artifacts/logs/%x_%j.err
# Compute per-layer covariances for every task under a checkpoints-analysisv2-epochs1
# run dir (the layout produced by finetune-analysisv2.sh, no max_samples subdir).
#
# Required env vars (override at sbatch time):
#   MODEL         e.g. ViT-B-16
#
# Optional:
#   SAVE_DIR      default: artifacts/checkpoints-analysisv2-epochs1
#   DATASETS      space-separated, default: full 8-task list
#
# Example:
#   sbatch --export=ALL,MODEL=ViT-B-16 scripts/vision/covariance-analysisv2.sh
set -euo pipefail
mkdir -p artifacts/logs

source ".venv-vl/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs
export HF_HOME=$SCRATCH/huggingface

if [ ! -d "$SLURM_TMPDIR/data" ]; then
  cp downloads/data.tar.gz "$SLURM_TMPDIR/"
  tar -xzf "$SLURM_TMPDIR/data.tar.gz" -C "$SLURM_TMPDIR/"
fi
ln -sfn "$SLURM_TMPDIR/data" data

MODEL="${MODEL:?set MODEL, e.g. ViT-B-16}"
SAVE_DIR="${SAVE_DIR:-artifacts/checkpoints-analysisv2-epochs1}"
DATASETS="${DATASETS:-Cars DTD EuroSAT GTSRB MNIST RESISC45 SUN397 SVHN}"

NUM_BATCHES=10
BATCH_SIZE=32
SPLIT="train"
MHA="split"
COV_TYPE="sm"
COV_ESTIMATOR="full"

RUN_DIR="$SAVE_DIR/$MODEL"
echo "[BASH] Target run dir: $RUN_DIR"
if [ ! -d "$RUN_DIR" ]; then
    echo "ERROR: run dir not found: $RUN_DIR"
    exit 1
fi

for DATASET in $DATASETS; do
    CKPT_DIR="$RUN_DIR/${DATASET}Val"
    if [ ! -d "$CKPT_DIR" ]; then
        echo "WARNING: missing checkpoint dir $CKPT_DIR, skipping $DATASET"
        continue
    fi

    echo "[BASH] covariance.py | model: $MODEL | dataset: $DATASET"
    python scripts/vision/covariance.py \
        --model="$MODEL" \
        --save="$SAVE_DIR" \
        --eval-datasets="$DATASET" \
        --cov-split="$SPLIT" \
        --cov-num-batches="$NUM_BATCHES" \
        --cov-batch-size="$BATCH_SIZE" \
        --mha="$MHA" \
        --cov-type="$COV_TYPE" \
        --cov-estimator="$COV_ESTIMATOR" \
        --cache-dir="$SCRATCH/openclip" \
        --data-location="data/vision"
done
