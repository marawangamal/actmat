#!/bin/bash
#SBATCH --job-name=eval_regmean
#SBATCH --partition=main
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

# 0. Setup environment
source "$SCRATCH/eigcov/.venv/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

DATA_DIR="$SLURM_TMPDIR/datasets"
OPENCLIP_DIR="$SCRATCH/openclip"

if [ ! -d "$DATA_DIR" ]; then
  cp vit_datasets_08.zip "$SLURM_TMPDIR/"
  unzip -q "$SLURM_TMPDIR/vit_datasets_08.zip" -d "$SLURM_TMPDIR/"
fi

# Common parameters
MODELS=(ViT-B-16 ViT-B-32 ViT-L-14)
NUM_BATCHES=10
BATCH_SIZE=32
FT_MODE=standard
RESULTS_DB="results/results-regmean.jsonl"

for MODEL in "${MODELS[@]}"; do

  # 1. Collect covariances (no --mha, defaults to None)
  echo "[BASH] Running covariance.py | model: $MODEL"
  python scripts/vision/covariance.py \
    --model="$MODEL" \
    --cov-split=train \
    --cov-num-batches="$NUM_BATCHES" \
    --cov-batch-size="$BATCH_SIZE" \
    --cov-type=sm \
    --mha=split \
    --cov-estimator=full \
    --openclip-cachedir="$OPENCLIP_DIR" \
    --data-location="$DATA_DIR"

  # 2. Evaluate RegMean
  COV_DIR="checkpoints/$MODEL/_covariances/covariances_strain_n${NUM_BATCHES}_b${BATCH_SIZE}_tsm_attnsplit_efull_ft${FT_MODE}"

  echo "[BASH] Running eval_task_addition.py | model: $MODEL | method: regmean"
  python scripts/vision/eval_task_addition.py \
    --model="$MODEL" \
    --finetuning-mode="$FT_MODE" \
    --merge-func=regmean \
    --cov-dir="$COV_DIR" \
    --data-location="$DATA_DIR" \
    --openclip-cachedir="$OPENCLIP_DIR" \
    --results-db="$RESULTS_DB"

done
