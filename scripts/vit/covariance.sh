#!/bin/bash
#SBATCH --job-name=cov_vit
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err

set -euo pipefail
mkdir -p artifacts/logs

source ".venv-vl/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

DATA_DIR="${DATA_DIR:-data/vision}"
OPENCLIP_DIR="${OPENCLIP_DIR:-$SCRATCH/openclip}"
CKPT_ROOT="${CKPT_ROOT:-artifacts/checkpoints}"
NUM_TASKS="${NUM_TASKS:-8}"
FT_MODE="${FT_MODE:-fft}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-finetuned.pt}"

DATASETS=(Cars DTD EuroSAT GTSRB MNIST RESISC45 SUN397 SVHN)
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
MODELS=(${MODELS:-ViT-B-16})

for MODEL in "${MODELS[@]}"; do
  EXPERT_DIR="$CKPT_ROOT/$MODEL/group-$FT_MODE-$NUM_TASKS/experts/$DATASET"
  python scripts/vit/covariance.py \
    --model "$MODEL" \
    --expert-dir "$EXPERT_DIR" \
    --dataset "$DATASET" \
    --checkpoint-name "$CHECKPOINT_NAME" \
    --output-path "$EXPERT_DIR/covariance.pt" \
    --data-location "$DATA_DIR" \
    --cache-dir "$OPENCLIP_DIR"
done

