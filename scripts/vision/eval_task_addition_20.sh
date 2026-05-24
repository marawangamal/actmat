#!/bin/bash
#SBATCH --job-name=eval_vision_20
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
RESULTS_DIR="artifacts/results20"
DATA_DIR="data/vision"
OPENCLIP_DIR="$SCRATCH/openclip"

# 1. Stage datasets to $SLURM_TMPDIR (mirrors finetune.sh / eval_task_addition.sh)
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

# Common parameters
NUM_BATCHES=10
BATCH_SIZE=32

# ===== Default experiments (no hyperparameter tuning) =====
# Only ViT-B-16 has the full 20-dataset finetunes available right now.
# ViT-B-32 and ViT-L-14 still have only the original 8; flip the line below
# once their 20-dataset finetunes finish.
# MODELS=(ViT-B-16 ViT-B-32 ViT-L-14)
MODELS=(ViT-B-16)
METHODS=(sum04)
FT_MODES=(standard lora)
MERGE_MODE=d
HPO=''

# Task scenarios (Wang et al. / TALL-masks):
#   8 : Cars, DTD, EuroSAT, GTSRB, MNIST, RESISC45, SUN397, SVHN
#   14: 8 + CIFAR100, STL10, Flowers102, OxfordIIITPet, PCAM, FER2013
#   20: 14 + EMNIST, CIFAR10, Food101, FashionMNIST, RenderedSST2, KMNIST
EVAL_DATASETS="Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR100,STL10,Flowers102,OxfordIIITPet,PCAM,FER2013,EMNIST,CIFAR10,Food101,FashionMNIST,RenderedSST2,KMNIST"


for FT_MODE in "${FT_MODES[@]}"; do
for MODEL in "${MODELS[@]}"; do
  # Evaluate task addition w/ diff merge methods
  for method in "${METHODS[@]}"; do

    # 2a. Run covariance/fisher script if needed
    if [ "$method" = "regmean" ]; then
      echo "[BASH] Running covariance.py | model: $MODEL | ft mode: $FT_MODE | method: $method"
      python scripts/vision/covariance.py \
        --model="$MODEL" \
        --finetuning-mode="$FT_MODE" \
        --save="$CKPT_ROOT" \
        --eval-datasets="$EVAL_DATASETS" \
        --mha=split
    elif [ "$method" = "fisher" ]; then
      echo "[BASH] Running fisher.py | model: $MODEL | ft mode: $FT_MODE | method: $method"
      python scripts/vision/fisher.py \
        --model="$MODEL" \
        --finetuning-mode="$FT_MODE" \
        --save="$CKPT_ROOT" \
        --eval-datasets="$EVAL_DATASETS" \
        --mha=split
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
