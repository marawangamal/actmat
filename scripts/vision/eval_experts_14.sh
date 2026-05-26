#!/bin/bash
#SBATCH --job-name=eval_vision_experts_14
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err

set -euo pipefail
mkdir -p artifacts/logs

# 0. Setup environment
source "$SCRATCH/actmat/.venv-vl/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

CKPT_ROOT="artifacts/checkpoints"
RESULTS_DIR="artifacts/results14"
DATA_DIR="$PWD/artifacts/data/vision"

# 1. Stage datasets to $SLURM_TMPDIR
if [ ! -d "$SLURM_TMPDIR/data" ]; then
  cp downloads/data.tar.gz "$SLURM_TMPDIR/"
  tar -xzf "$SLURM_TMPDIR/data.tar.gz" -C "$SLURM_TMPDIR/"
fi
ln -sfn "$SLURM_TMPDIR/data" data

# Stage KMNIST raw files.
KMNIST_RAW_DST="$SLURM_TMPDIR/data/vision/KMNIST/KMNIST/raw"
if [ ! -f "$KMNIST_RAW_DST/train-images-idx3-ubyte.gz" ] && [ -d downloads/kmnist ]; then
  mkdir -p "$KMNIST_RAW_DST"
  cp downloads/kmnist/*.gz "$KMNIST_RAW_DST/"
fi

# Stage PCAM h5 files (real copy — h5py + DataLoader can't resolve symlinks).
PCAM_DST="$SLURM_TMPDIR/data/vision/PCAM/pcam"
PCAM_SRC="$PWD/artifacts/data/vision/PCAM/pcam"
if [ ! -f "$PCAM_DST/camelyonpatch_level_2_split_test_y.h5" ] && [ -d "$PCAM_SRC" ]; then
  mkdir -p "$PCAM_DST"
  for f in "$PCAM_SRC"/*.h5; do
    [ -f "$f" ] && cp "$f" "$PCAM_DST/$(basename "$f")"
  done
fi

# 2. Array dispatch: one task per model.
MODELS=(ViT-B-16 ViT-B-32 ViT-L-14)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

FT_MODES=(none standard)
# 14-task subset (Wang et al. / TALL-masks)
EVAL_DATASETS="Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR100,STL10,Flowers102,OxfordIIITPet,PCAM,FER2013"

for FT_MODE in "${FT_MODES[@]}"; do
  echo "[BASH] Running eval_experts.py | model: $MODEL | ft mode: $FT_MODE"
  python scripts/vision/eval_experts.py \
    --model="$MODEL" \
    --finetuning-mode="$FT_MODE" \
    --save="$CKPT_ROOT" \
    --data-location="$DATA_DIR" \
    --results-dir="$RESULTS_DIR" \
    --eval-datasets="$EVAL_DATASETS"
done
