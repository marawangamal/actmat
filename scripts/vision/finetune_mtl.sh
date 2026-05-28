#!/bin/bash
#SBATCH --job-name=ft_mtl_vision
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=1-2%1
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err

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

MODELS=(ViT-B-16 ViT-B-32 ViT-L-14)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
FT_MODE=${FT_MODE:-standard}

echo "[BASH] Running finetune_mtl.py | model: $MODEL | ft mode: $FT_MODE | save dir: $SAVE_DIR"
python scripts/vision/finetune_mtl.py \
  --finetuning-mode="$FT_MODE" \
  --model="$MODEL" \
  --world-size=1 \
  --num-workers=1 \
  --cache-dir="$OPENCLIP_DIR" \
  --data-location="$DATA_DIR" \
  --save="$SAVE_DIR" \
  --checkpoint-every=500 \
  --patience=10
