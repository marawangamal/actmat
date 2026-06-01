#!/bin/bash
#SBATCH --job-name=eval_vision_experts_sgd
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=artifacts/logs/%x_%j.out
#SBATCH --error=artifacts/logs/%x_%j.err

set -euo pipefail
mkdir -p artifacts/logs

# 0. Setup environment
source ".venv-vl/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

CKPT_ROOT="artifacts/checkpoints-sgd"
# Separate results dir so we don't clobber the AdamW experts metrics.json
# (eval_experts.py keys output only by model name).
RESULTS_DIR="artifacts/results-sgd"
DATA_DIR="data/vision"

# 1. Stage datasets to $SLURM_TMPDIR (mirrors finetune.sh)
if [ ! -d "$SLURM_TMPDIR/data" ]; then
  cp downloads/data.tar.gz "$SLURM_TMPDIR/"
  tar -xzf "$SLURM_TMPDIR/data.tar.gz" -C "$SLURM_TMPDIR/"
fi
ln -sfn "$SLURM_TMPDIR/data" data

# 2. Evaluate the SGD-trained ViT-B-16 standard experts on the 8-task suite.
MODEL="ViT-B-16"
FT_MODE="standard"
EVAL_DATASETS="Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN"

echo "[BASH] Running eval_experts.py | model: $MODEL | ft mode: $FT_MODE | ckpt: $CKPT_ROOT"
python scripts/vision/eval_experts.py \
  --model="$MODEL" \
  --finetuning-mode="$FT_MODE" \
  --save="$CKPT_ROOT" \
  --data-location="$DATA_DIR" \
  --results-dir="$RESULTS_DIR" \
  --eval-datasets="$EVAL_DATASETS"
