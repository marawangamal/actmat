#!/bin/bash
#SBATCH --job-name=finetune_vision_lora_retry
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err

set -euo pipefail
mkdir -p artifacts/logs

# 1. Setup environment
source ".venv-vl/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs
DATA_DIR="data/vision"
OPENCLIP_DIR="$SCRATCH/openclip"

# 2. Stage datasets to $SLURM_TMPDIR.
if [ ! -d "$SLURM_TMPDIR/data" ]; then
  cp downloads/data.tar.gz "$SLURM_TMPDIR/"
  tar -xzf "$SLURM_TMPDIR/data.tar.gz" -C "$SLURM_TMPDIR/"
fi
ln -sfn "$SLURM_TMPDIR/data" data

KMNIST_RAW_DST="$SLURM_TMPDIR/data/vision/KMNIST/KMNIST/raw"
if [ ! -f "$KMNIST_RAW_DST/train-images-idx3-ubyte.gz" ] && [ -d downloads/kmnist ]; then
  mkdir -p "$KMNIST_RAW_DST"
  cp downloads/kmnist/*.gz "$KMNIST_RAW_DST/"
fi

# Real copy for PCAM (h5py + DataLoader workers can't resolve symlink chains).
PCAM_DST="$SLURM_TMPDIR/data/vision/PCAM/pcam"
PCAM_SRC="$PWD/artifacts/data/vision/PCAM/pcam"
if [ ! -f "$PCAM_DST/camelyonpatch_level_2_split_test_y.h5" ] && [ -d "$PCAM_SRC" ]; then
  mkdir -p "$PCAM_DST"
  for f in "$PCAM_SRC"/*.h5; do
    [ -f "$f" ] && cp "$f" "$PCAM_DST/$(basename "$f")"
  done
fi

# 3. Array dispatch over the 6 missing (model, dataset) lora pairs.
PAIRS=(
  "ViT-B-32:PCAM"
  "ViT-B-32:RenderedSST2"
  "ViT-L-14:OxfordIIITPet"
  "ViT-L-14:PCAM"
  "ViT-L-14:EMNIST"
  "ViT-L-14:RenderedSST2"
)
pair="${PAIRS[$SLURM_ARRAY_TASK_ID]}"
MODEL="${pair%:*}"
DATASET="${pair#*:}"
FT_MODE="lora"
SAVE_DIR="artifacts/checkpoints"

PORT=$((12000 + SLURM_JOB_ID % 50000))

echo "[BASH] Running finetune.py | model: $MODEL | ft mode: $FT_MODE | dataset: $DATASET | port: $PORT"
python scripts/vision/finetune.py \
  --finetuning-mode="$FT_MODE" \
  --model="$MODEL" \
  --world-size=1 \
  --num-workers=1 \
  --port="$PORT" \
  --cache-dir="$OPENCLIP_DIR" \
  --data-location="$DATA_DIR" \
  --save="$SAVE_DIR" \
  --train-dataset "$DATASET"
