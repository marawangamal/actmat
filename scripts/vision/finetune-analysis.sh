#!/bin/bash
#SBATCH --job-name=finetune_vision_models
#SBATCH --partition=main
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

# 1. Setup environment (NOTE: change this to your environment)
source ".venv-vl/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs
DATA_DIR="$SLURM_TMPDIR/datasets"
OPENCLIP_DIR="$SCRATCH/openclip"


# 2. Download datasets (NOTE: change this to your environment)
if [ ! -d "$DATA_DIR" ]; then
  cp vit_datasets_08.zip "$SLURM_TMPDIR/"
  unzip -q "$SLURM_TMPDIR/vit_datasets_08.zip" -d "$SLURM_TMPDIR/"
fi

# 3. Finetune models (using FFT & LoRA)
MODELS=(ViT-B-16)
FT_MODES=(standard)
SAVE_DIR="checkpoints-analysis"

for MODEL in "${MODELS[@]}"; do
  for FT_MODE in "${FT_MODES[@]}"; do

    echo "[BASH] Running finetune.py | model: $MODEL | ft mode: $FT_MODE | save dir: $SAVE_DIR"
    python scripts/vision/finetune.py \
      --finetuning-mode="$FT_MODE" \
      --model="$MODEL" \
      --world-size=1 \
      --num-workers=1 \
      --openclip-cachedir="$OPENCLIP_DIR" \
      --data-location="$DATA_DIR" \
      --save="$SAVE_DIR" \
      --grad-cross-matrix

  done
done
