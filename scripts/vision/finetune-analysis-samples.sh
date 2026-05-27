#!/bin/bash
#SBATCH --job-name=finetune_vision_analysis_samples
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-5
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err

set -euo pipefail
mkdir -p artifacts/logs

# 1. Setup environment (NOTE: change this to your environment)
source ".venv-vl/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs
DATA_DIR="data/vision"
OPENCLIP_DIR="$SCRATCH/openclip"


# 2. Download datasets (NOTE: change this to your environment)
if [ ! -d "$SLURM_TMPDIR/data" ]; then
  cp downloads/data.tar.gz "$SLURM_TMPDIR/"
  tar -xzf "$SLURM_TMPDIR/data.tar.gz" -C "$SLURM_TMPDIR/"
fi
ln -sfn "$SLURM_TMPDIR/data" data

# 3. Finetune models (using FFT & LoRA)
#
# Per-dataset unique training-sample counts (counted from downloads/data.tar.gz).
# Effective = after split_train_into_train_val with val_fraction=0.1,
# max_val_samples=5000 (the *Val variant used by finetune.py).
#
#   Dataset    Raw train    Effective train (xxxVal)
#   Cars         8,144        7,330
#   DTD          3,760        3,384
#   EuroSAT     21,600       19,440
#   GTSRB       26,640       23,976
#   MNIST       60,000       55,000   (val capped at 5000)
#   RESISC45    18,900       17,010
#   SUN397      19,850       17,865
#   SVHN        73,257       68,257   (val capped at 5000)
#
# --max-samples caps unique examples per dataset against the "Effective" column.
# Smallest effective set is DTD (3,384), so MAX_SAMPLES > 3384 is a no-op there.
#
# Job array sweeps MAX_SAMPLES; keep #SBATCH --array in sync with MAX_SAMPLES_LIST length.
MODELS=("${MODEL:-ViT-B-16}")
FT_MODES=(standard)
SAVE_DIR="artifacts/checkpoints-analysis"
MAX_SAMPLES_LIST=(10 100 200 500 1000 10000)

MAX_SAMPLES="${MAX_SAMPLES_LIST[$SLURM_ARRAY_TASK_ID]}"
# Derive a unique DDP port to avoid EADDRINUSE when multiple array tasks share a node.
DDP_PORT=$(( 12000 + (SLURM_ARRAY_JOB_ID % 1000) * 10 + SLURM_ARRAY_TASK_ID ))

for MODEL in "${MODELS[@]}"; do
  for FT_MODE in "${FT_MODES[@]}"; do

    echo "[BASH] Running finetune.py | model: $MODEL | ft mode: $FT_MODE | save dir: $SAVE_DIR | max samples: $MAX_SAMPLES"
    python scripts/vision/finetune.py \
      --finetuning-mode="$FT_MODE" \
      --model="$MODEL" \
      --world-size=1 \
      --num-workers=1 \
      --port="$DDP_PORT" \
      --cache-dir="$OPENCLIP_DIR" \
      --data-location="$DATA_DIR" \
      --save="$SAVE_DIR" \
      --epochs=2 \
      --max-samples="$MAX_SAMPLES" \
      --train-dataset=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN \
      --grad-cross-matrix

  done
done
