#!/bin/bash
#SBATCH --job-name=ft_mtl_vision
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=artifacts/logs/%x_%j.out
#SBATCH --error=artifacts/logs/%x_%j.err

set -euo pipefail
mkdir -p artifacts/logs

# 0. Setup environment
source "$SCRATCH/actmat/.venv-vl/bin/activate"
export HF_HOME=$SCRATCH/huggingface
export NLTK_DATA=$SCRATCH/nltk_data
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

# 1. Stage vision datasets to $SLURM_TMPDIR
if [ ! -d "$SLURM_TMPDIR/data" ]; then
  cp downloads/data.tar.gz "$SLURM_TMPDIR/"
  tar -xzf "$SLURM_TMPDIR/data.tar.gz" -C "$SLURM_TMPDIR/"
fi
ln -sfn "$SLURM_TMPDIR/data" data

DATA_DIR="data/vision"
OPENCLIP_DIR="$SCRATCH/openclip"
SAVE_DIR="artifacts/checkpoints"

MODEL=${MODEL:-ViT-B-32}
FT_MODE=${FT_MODE:-standard}

echo "[BASH] Running finetune_mtl.py | model: $MODEL | ft mode: $FT_MODE"
python scripts/vision/finetune_mtl.py \
  --model="$MODEL" \
  --finetuning-mode="$FT_MODE" \
  --data-location="$DATA_DIR" \
  --cache-dir="$OPENCLIP_DIR" \
  --save="$SAVE_DIR" \
  --num-workers=2 \
  --checkpoint-every=200 \
  --patience=10
