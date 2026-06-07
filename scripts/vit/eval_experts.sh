#!/bin/bash
#SBATCH --job-name=eval_vit_experts
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
NUM_TASKS="${NUM_TASKS:-8}"
FT_MODE="${FT_MODE:-fft}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-finetuned.pt}"
EVAL_DATASETS="${EVAL_DATASETS:-Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN}"
MODELS=(${MODELS:-ViT-B-16})

for MODEL in "${MODELS[@]}"; do
  python scripts/vit/eval_experts.py \
    --model "$MODEL" \
    --experts-dir "$CKPT_ROOT/$MODEL/group-$FT_MODE-$NUM_TASKS/experts" \
    --eval-datasets "$EVAL_DATASETS" \
    --checkpoint-name "$CHECKPOINT_NAME" \
    --output-dir "$RESULTS_ROOT/$MODEL/group-$FT_MODE-$NUM_TASKS/experts" \
    --data-location "$DATA_DIR" \
    --cache-dir "$OPENCLIP_DIR"
done

