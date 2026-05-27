#!/bin/bash
#SBATCH --job-name=covariance_analysis
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=artifacts/logs/%x_%j.out
#SBATCH --error=artifacts/logs/%x_%j.err
# Compute per-layer covariances for every task under a checkpoints-analysis
# run dir (the layout produced by finetune-analysis-samples.sh).
#
# Required env vars (override at sbatch time):
#   MODEL         e.g. ViT-B-16
#   MAX_SAMPLES   e.g. 1280   (resolves to <SAVE>/<MODEL>/max_samples_<N>)
#
# Optional:
#   SAVE_DIR      default: artifacts/checkpoints-analysis
#   DATASETS      space-separated, default: full 8-task list
#
# Example:
#   sbatch --export=ALL,MODEL=ViT-B-16,MAX_SAMPLES=1280 scripts/vision/covariance-analysis.sh
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
MAX_SAMPLES="${MAX_SAMPLES:?set MAX_SAMPLES, e.g. 1280}"
SAVE_DIR="${SAVE_DIR:-artifacts/checkpoints-analysis}"
DATASETS="${DATASETS:-Cars DTD EuroSAT GTSRB MNIST RESISC45 SUN397 SVHN}"

NUM_BATCHES=10
BATCH_SIZE=32
SPLIT="train"
MHA="split"
COV_TYPE="sm"
COV_ESTIMATOR="full"

RUN_DIR="$SAVE_DIR/$MODEL/max_samples_$MAX_SAMPLES"
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

    echo "[BASH] covariance.py | model: $MODEL | max_samples: $MAX_SAMPLES | dataset: $DATASET"
    python scripts/vision/covariance.py \
        --model="$MODEL" \
        --save="$SAVE_DIR" \
        --max-samples="$MAX_SAMPLES" \
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
