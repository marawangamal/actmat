#!/bin/bash
#SBATCH --job-name=eval_vit_single
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=artifacts/logs/%x_%A.out
#SBATCH --error=artifacts/logs/%x_%A.err

set -euo pipefail
mkdir -p artifacts/logs

source ".venv-vl/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

DATA_DIR="${DATA_DIR:-data/vision}"
OPENCLIP_DIR="${OPENCLIP_DIR:-$SCRATCH/openclip}"
CKPT_ROOT="${CKPT_ROOT:-artifacts/checkpoints}"
RESULTS_ROOT="${RESULTS_ROOT:-artifacts/results}"
MODEL="${MODEL:-ViT-B-16}"
EXPERT_DIR="${EXPERT_DIR:?Set EXPERT_DIR to a directory containing model/head files}"
OUTPUT_DIR="${OUTPUT_DIR:-$RESULTS_ROOT/$MODEL/single}"
EVAL_DATASETS="${EVAL_DATASETS:-Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-finetuned.pt}"
HEADS_DIR_ARGS=()
if [ -n "${HEADS_DIR:-}" ]; then
  HEADS_DIR_ARGS=(--heads-dir "$HEADS_DIR")
fi

python scripts/vit/eval_single.py \
  --model "$MODEL" \
  --expert-dir "$EXPERT_DIR" \
  --eval-datasets "$EVAL_DATASETS" \
  --checkpoint-name "$CHECKPOINT_NAME" \
  --output-dir "$OUTPUT_DIR" \
  --data-location "$DATA_DIR" \
  --cache-dir "$OPENCLIP_DIR" \
  "${HEADS_DIR_ARGS[@]}"
