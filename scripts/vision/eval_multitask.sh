#!/bin/bash
#SBATCH --job-name=eval_vision_mtl
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=artifacts/logs/%x_%j.out
#SBATCH --error=artifacts/logs/%x_%j.err

set -euo pipefail
mkdir -p artifacts/logs

source "$SCRATCH/actmat/.venv-vl/bin/activate"
export HF_HOME=$SCRATCH/huggingface
export NLTK_DATA=$SCRATCH/nltk_data
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

if [ ! -d "$SLURM_TMPDIR/data" ]; then
  cp downloads/data.tar.gz "$SLURM_TMPDIR/"
  tar -xzf "$SLURM_TMPDIR/data.tar.gz" -C "$SLURM_TMPDIR/"
fi
ln -sfn "$SLURM_TMPDIR/data" data

DATA_DIR="data/vision"
OPENCLIP_DIR="$SCRATCH/openclip"

MODELS=(${MODELS:-ViT-B-16 ViT-B-32})
FT_MODES=(${FT_MODES:-standard})

for FT_MODE in "${FT_MODES[@]}"; do
for MODEL in "${MODELS[@]}"; do
  echo "[BASH] Running eval_multitask.py | model: $MODEL | ft mode: $FT_MODE"
  python scripts/vision/eval_multitask.py \
    --model="$MODEL" \
    --finetuning-mode="$FT_MODE" \
    --group=20 \
    --cache-dir="$OPENCLIP_DIR" \
    --data-location="$DATA_DIR"
done
done
