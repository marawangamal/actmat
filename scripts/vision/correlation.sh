#!/bin/bash
#SBATCH --job-name=correlation
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=artifacts/logs/%x_%j.out
#SBATCH --error=artifacts/logs/%x_%j.err

set -euo pipefail
mkdir -p artifacts/logs

# ─── Configuration ────────────────────────────────────────────────────────────
MODEL="ViT-B-16"
FT_MODE="standard"
DATA_DIR="data/vision"
OPENCLIP_DIR="$SCRATCH/openclip"

DATASETS=(
    "Cars"
    "DTD"
    "EuroSAT"
    "GTSRB"
    "MNIST"
    "RESISC45"
    "SUN397"
    "SVHN"
)
# ──────────────────────────────────────────────────────────────────────────────

source ".venv-vl/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs
export HF_HOME=$SCRATCH/huggingface

if [ ! -d "$SLURM_TMPDIR/data" ]; then
  cp downloads/data.tar.gz "$SLURM_TMPDIR/"
  tar -xzf "$SLURM_TMPDIR/data.tar.gz" -C "$SLURM_TMPDIR/"
fi
ln -sfn "$SLURM_TMPDIR/data" data

EVAL_DATASETS=$(IFS=,; echo "${DATASETS[*]}")

echo "[BASH] correlation.py | model: $MODEL | ft mode: $FT_MODE | datasets: $EVAL_DATASETS"
python scripts/vision/correlation.py \
    --model="$MODEL" \
    --finetuning-mode="$FT_MODE" \
    --eval-datasets="$EVAL_DATASETS" \
    --cache-dir="$OPENCLIP_DIR" \
    --data-location="$DATA_DIR"
