#!/bin/bash
#SBATCH --job-name=eval_vision_14
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=artifacts/logs/%x_%j.out
#SBATCH --error=artifacts/logs/%x_%j.err

set -euo pipefail
mkdir -p artifacts/logs

# 0. Setup environment
source "$SCRATCH/actmat/.venv-vl/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

CKPT_ROOT="artifacts/checkpoints"
RESULTS_DIR="artifacts/results14"
DATA_DIR="$PWD/artifacts/data/vision"
OPENCLIP_DIR="$SCRATCH/openclip"

# 1. Stage datasets to $SLURM_TMPDIR (mirrors finetune.sh / eval_task_addition.sh)
if [ ! -d "$SLURM_TMPDIR/data" ]; then
  cp downloads/data.tar.gz "$SLURM_TMPDIR/"
  tar -xzf "$SLURM_TMPDIR/data.tar.gz" -C "$SLURM_TMPDIR/"
fi
ln -sfn "$SLURM_TMPDIR/data" data

# Stage KMNIST raw files (not needed for 14-task subset but kept for parity / future flips).
KMNIST_RAW_DST="$SLURM_TMPDIR/data/vision/KMNIST/KMNIST/raw"
if [ ! -f "$KMNIST_RAW_DST/train-images-idx3-ubyte.gz" ] && [ -d downloads/kmnist ]; then
  mkdir -p "$KMNIST_RAW_DST"
  cp downloads/kmnist/*.gz "$KMNIST_RAW_DST/"
fi

# Stage PCAM h5 files (torchvision pulls from Google Drive and easily hits rate-limits).
PCAM_DST="$SLURM_TMPDIR/data/vision/PCAM/pcam"
PCAM_SRC="$PWD/artifacts/data/vision/PCAM/pcam"
if [ ! -f "$PCAM_DST/camelyonpatch_level_2_split_test_y.h5" ] && [ -d "$PCAM_SRC" ]; then
  mkdir -p "$PCAM_DST"
  for f in "$PCAM_SRC"/*.h5 "$PCAM_SRC"/*.h5.gz; do
    [ -f "$f" ] && ln -sfn "$f" "$PCAM_DST/$(basename "$f")"
  done
fi

# Common parameters
NUM_BATCHES=10
BATCH_SIZE=32

# ===== Default experiments (no hyperparameter tuning) =====
# Only ViT-B-16 has the full 20-dataset finetunes available right now.
# ViT-B-32 and ViT-L-14 still have only the original 8; flip the line below
# once their 20-dataset finetunes finish.
# MODELS=(ViT-B-16 ViT-B-32 ViT-L-14)
MODELS=(ViT-B-16)
METHODS=(mean actmat tsv wudi isoc regmean)
FT_MODES=(standard)
MERGE_MODE=d
HPO=''

# Task scenarios (Wang et al. / TALL-masks):
#   8 : Cars, DTD, EuroSAT, GTSRB, MNIST, RESISC45, SUN397, SVHN
#   14: 8 + CIFAR100, STL10, Flowers102, OxfordIIITPet, PCAM, FER2013
#   20: 14 + EMNIST, CIFAR10, Food101, FashionMNIST, RenderedSST2, KMNIST
EVAL_DATASETS="Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR100,STL10,Flowers102,OxfordIIITPet,PCAM,FER2013"


for FT_MODE in "${FT_MODES[@]}"; do
for MODEL in "${MODELS[@]}"; do
  # Reset per-model stats flags so covariance/fisher run at most once per model.
  cov_done=false
  fisher_done=false
  # Evaluate task addition w/ diff merge methods
  for method in "${METHODS[@]}"; do

    # 2a. Run covariance/fisher script if needed (regmean + actmat both consume covariance.pt)
    if { [ "$method" = "regmean" ] || [ "$method" = "actmat" ]; } && [ "$cov_done" = false ]; then
      echo "[BASH] Running covariance.py | model: $MODEL | ft mode: $FT_MODE"
      python scripts/vision/covariance.py \
        --model="$MODEL" \
        --finetuning-mode="$FT_MODE" \
        --save="$CKPT_ROOT" \
        --data-location="$DATA_DIR" \
        --eval-datasets="$EVAL_DATASETS" \
        --mha=split
      cov_done=true
    elif [ "$method" = "fisher" ] && [ "$fisher_done" = false ]; then
      echo "[BASH] Running fisher.py | model: $MODEL | ft mode: $FT_MODE"
      python scripts/vision/fisher.py \
        --model="$MODEL" \
        --finetuning-mode="$FT_MODE" \
        --save="$CKPT_ROOT" \
        --data-location="$DATA_DIR" \
        --eval-datasets="$EVAL_DATASETS" \
        --mha=split
      fisher_done=true
    fi

    # 2b. Evaluate task addition
    echo "[BASH] Running eval_task_addition.py | model: $MODEL | ft mode: $FT_MODE | method: $method | mode: $MERGE_MODE"
    python scripts/vision/eval_task_addition.py \
      --model="$MODEL" \
      --finetuning-mode="$FT_MODE" \
      --save="$CKPT_ROOT" \
      --data-location="$DATA_DIR" \
      --merge-func="$method" \
      --merge-mode="$MERGE_MODE" \
      --results-dir="$RESULTS_DIR" \
      --eval-datasets="$EVAL_DATASETS" \
      --mha=split \
      ${HPO:+--hpo="$HPO"}

  done
done
done
