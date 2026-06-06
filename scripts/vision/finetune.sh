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

# 3. Array dispatch over the standard 8-task benchmark (Ilharco et al.); loop over
#    models inside each task. Per-checkpoint skip logic in finetune.py makes
#    already-saved (model, dataset) runs a no-op.
DATASETS=(Cars DTD EuroSAT GTSRB MNIST RESISC45 SUN397 SVHN)
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

MODELS=(ViT-B-32 ViT-B-16 ViT-L-14)
FT_MODE="standard"
SAVE_DIR="artifacts/checkpoints-analysis"

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
    --epochs=1 \
    --train-dataset="$DATASET" \
    --grad-cross-matrix \
    --checkpoint-every=200
done
