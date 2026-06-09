#!/bin/bash
#SBATCH --job-name=eval_vit_single
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=artifacts/logs/%x_%j.out
#SBATCH --error=artifacts/logs/%x_%j.err

set -euo pipefail
mkdir -p artifacts/logs

source ".venv-vl/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

NUM_TASKS="${NUM_TASKS:-8}"
FT_MODE="${FT_MODE:-fft}"
SINGLE_DIR="${SINGLE_DIR:-pretrained}"
EXPERT_DIR="${EXPERT_DIR:-}"
EXPERTS_DIR="${EXPERTS_DIR:-}"
EVAL_DATASETS="${EVAL_DATASETS:-Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN}"
DATA_DIR="data/vision"
OPENCLIP_DIR="$SCRATCH/openclip"
CKPT_ROOT="artifacts/checkpoints"
RESULTS_ROOT="artifacts/results"
MODEL="${MODEL:-ViT-B-16}"

EXPERT_DIR="${EXPERT_DIR:-$CKPT_ROOT/$MODEL/group-$FT_MODE-$NUM_TASKS/$SINGLE_DIR}"
EXPERTS_DIR="${EXPERTS_DIR:-$CKPT_ROOT/$MODEL/group-$FT_MODE-$NUM_TASKS/experts}"
OUTPUT_DIR="$RESULTS_ROOT/$MODEL/group-$FT_MODE-$NUM_TASKS/$SINGLE_DIR"

# --expert-dir supplies the encoder; --experts-dir supplies per-dataset heads.
python scripts/vit/eval_single.py \
  --model "$MODEL" \
  --expert-dir "$EXPERT_DIR" \
  --experts-dir "$EXPERTS_DIR" \
  --eval-datasets "$EVAL_DATASETS" \
  --output-dir "$OUTPUT_DIR" \
  --data-location "$DATA_DIR" \
  --cache-dir "$OPENCLIP_DIR"
