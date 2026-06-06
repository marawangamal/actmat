#!/bin/bash
#SBATCH --job-name=finetune_vision
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err

# Finetune every model on every task, collecting grad-cross-moment (GCM) stats
# (gbar/sbar/stilde) and saving intermediate drift checkpoints. One array task
# per dataset; each task loops over models. Consolidates the former
# finetune-analysis{,-samples,-drift,v2,v3}.sh scripts onto a single finetune.py.

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

# Stage KMNIST raw files (torchvision's KMNIST mirror is unreliable from compute nodes).
KMNIST_RAW_DST="$SLURM_TMPDIR/data/vision/KMNIST/KMNIST/raw"
if [ ! -f "$KMNIST_RAW_DST/train-images-idx3-ubyte.gz" ] && [ -d downloads/kmnist ]; then
  mkdir -p "$KMNIST_RAW_DST"
  cp downloads/kmnist/*.gz "$KMNIST_RAW_DST/"
fi

# Stage PCAM h5 files (torchvision pulls from Google Drive and easily hits rate-limits).
# COPY rather than symlink — h5py inside DataLoader workers fails to resolve paths
# through symlink chains on network FS ("can't retrieve real path for file").
PCAM_DST="$SLURM_TMPDIR/data/vision/PCAM/pcam"
PCAM_SRC="$PWD/artifacts/data/vision/PCAM/pcam"
if [ ! -f "$PCAM_DST/camelyonpatch_level_2_split_test_y.h5" ] && [ -d "$PCAM_SRC" ]; then
  mkdir -p "$PCAM_DST"
  for f in "$PCAM_SRC"/*.h5; do
    [ -f "$f" ] && cp "$f" "$PCAM_DST/$(basename "$f")"
  done
fi

# 3. Array dispatch over the standard 8-task benchmark (Ilharco et al.); loop over
#    models inside each task. Per-checkpoint skip logic in finetune.py makes
#    already-saved (model, dataset) runs a no-op.
DATASETS=(Cars DTD EuroSAT GTSRB MNIST RESISC45 SUN397 SVHN)
# Full 20-task suite (set #SBATCH --array=0-19 and uncomment):
# DATASETS=(Cars DTD EuroSAT GTSRB MNIST RESISC45 SUN397 SVHN CIFAR10 CIFAR100 STL10 Food101 Flowers102 FER2013 PCAM OxfordIIITPet RenderedSST2 EMNIST FashionMNIST KMNIST)
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

MODELS=(ViT-B-16)
FT_MODE="standard"
SAVE_DIR="artifacts/checkpoints"

# Unique DDP rendezvous port per task so co-located array tasks don't collide.
PORT=$((12355 + SLURM_ARRAY_TASK_ID))

for MODEL in "${MODELS[@]}"; do
  echo "[BASH] Running finetune.py | model: $MODEL | ft mode: $FT_MODE | dataset: $DATASET | save dir: $SAVE_DIR"
  python scripts/vision/finetune.py \
    --finetuning-mode="$FT_MODE" \
    --model="$MODEL" \
    --world-size=1 \
    --num-workers=1 \
    --port="$PORT" \
    --cache-dir="$OPENCLIP_DIR" \
    --data-location="$DATA_DIR" \
    --save="$SAVE_DIR" \
    --train-dataset="$DATASET" \
    --grad-cross-matrix \
    --checkpoint-every=200
done
