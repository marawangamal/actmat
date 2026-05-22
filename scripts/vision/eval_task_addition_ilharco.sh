#!/bin/bash
#SBATCH --job-name=eval_vision_ilharco
#SBATCH --partition=main
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=artifacts/logs/%x_%j.out
#SBATCH --error=artifacts/logs/%x_%j.err

set -euo pipefail
mkdir -p artifacts/logs

# 0. Setup environment
source "$SCRATCH/actmat/.venv-vl/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

DATA_DIR="data/vision"
CKPT_ROOT="artifacts/checkpoints-ilharco/task_vectors_checkpoints"
RESULTS_DIR="artifacts/results-ilharco"

if [ ! -d "$SLURM_TMPDIR/data" ]; then
  cp downloads/data.tar.gz "$SLURM_TMPDIR/"
  tar -xzf "$SLURM_TMPDIR/data.tar.gz" -C "$SLURM_TMPDIR/"
fi
ln -sfn "$SLURM_TMPDIR/data" data

# Common parameters
NUM_BATCHES=10
BATCH_SIZE=32

# Ilharco released full-FT checkpoints only, so FT_MODES=(standard).
MODELS=(ViT-B-16 ViT-B-32 ViT-L-14)
METHODS=(ties dare sum mean tsv isoc regmean actmat)
FT_MODES=(standard)
MERGE_MODE=d
HPO=''


for FT_MODE in "${FT_MODES[@]}"; do
for MODEL in "${MODELS[@]}"; do
  # Evaluate task addition w/ diff merge methods
  for method in "${METHODS[@]}"; do

    # 2a. Run covariance/fisher script if needed
    if [ "$method" = "regmean" ]; then
      echo "[BASH] Running covariance.py | model: $MODEL | ft mode: $FT_MODE | method: $method"
      python scripts/vision/covariance.py \
        --model="$MODEL" \
        --finetuning-mode="$FT_MODE" \
        --save="$CKPT_ROOT" \
        --mha=split
    elif [ "$method" = "fisher" ]; then
      echo "[BASH] Running fisher.py | model: $MODEL | ft mode: $FT_MODE | method: $method"
      python scripts/vision/fisher.py \
        --model="$MODEL" \
        --finetuning-mode="$FT_MODE" \
        --save="$CKPT_ROOT" \
        --mha=split
    fi

    # 2b. Evaluate task addition
    echo "[BASH] Running eval_task_addition.py | model: $MODEL | ft mode: $FT_MODE | method: $method | mode: $MERGE_MODE"
    python scripts/vision/eval_task_addition.py \
      --model="$MODEL" \
      --finetuning-mode="$FT_MODE" \
      --save="$CKPT_ROOT" \
      --data-location="$DATA_DIR" \
      --merge-func="$method" \
      --merge-mode="$MERGE_MODE" \
      --results-dir="$RESULTS_DIR" \
      --mha=split \
      ${HPO:+--hpo="$HPO"}

  done
done
done
