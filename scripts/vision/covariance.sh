#!/bin/bash
#SBATCH --job-name=covariance
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=artifacts/logs/%x_%j.out
#SBATCH --error=artifacts/logs/%x_%j.err

set -euo pipefail
mkdir -p artifacts/logs

# Edit these values for the run.
MODEL="ViT-B-16"
EXPERTS_DIR="artifacts/checkpoints/${MODEL}/group-20/experts"
DATA_LOCATION="data/vision"
CACHE_DIR="$SCRATCH/openclip"
NUM_BATCHES=10
BATCH_SIZE=32
OVERWRITE=0

source ".venv-vl/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs
export HF_HOME=$SCRATCH/huggingface

if [ -n "${SLURM_TMPDIR:-}" ]; then
  if [ ! -d "$SLURM_TMPDIR/data" ]; then
    cp downloads/data.tar.gz "$SLURM_TMPDIR/"
    tar -xzf "$SLURM_TMPDIR/data.tar.gz" -C "$SLURM_TMPDIR/"
  fi
  ln -sfn "$SLURM_TMPDIR/data" data
fi

if [ ! -d "$EXPERTS_DIR" ]; then
  echo "ERROR: experts dir not found: $EXPERTS_DIR"
  exit 1
fi

echo "[BASH] Computing covariances under $EXPERTS_DIR"

find "$EXPERTS_DIR" -mindepth 2 -maxdepth 2 -type f \( \
  -name finetuned.pt -o -name 'checkpoint_*.pt' \
\) | sort | while read -r ckpt; do
  ckpt_dir="$(dirname "$ckpt")"
  ckpt_stem="$(basename "$ckpt" .pt)"
  output="$ckpt_dir/covariance.pt"
  if [ "$ckpt_stem" != "finetuned" ]; then
    output="$ckpt_dir/covariance_${ckpt_stem}.pt"
  fi

  if [ -f "$output" ] && [ "$OVERWRITE" != "1" ]; then
    echo "[BASH] Skipping cached $output"
    continue
  fi

  echo "[BASH] covariance.py | $ckpt"
  python scripts/vision/covariance.py \
    --model="$MODEL" \
    --finetuned-path="$ckpt" \
    --output-path="$output" \
    --data-location="$DATA_LOCATION" \
    --cache-dir="$CACHE_DIR" \
    --cov-num-batches="$NUM_BATCHES" \
    --cov-batch-size="$BATCH_SIZE" \
    --mha=split \
    --cov-type=sm \
    --cov-estimator=full
done
