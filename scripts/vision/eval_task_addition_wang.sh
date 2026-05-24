#!/bin/bash
#SBATCH --job-name=eval_vision_wang
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

CKPT_ROOT="artifacts/checkpoints-wang"
RESULTS_DIR="artifacts/results-wang"
# Absolute path: avoids the repo-level `data/` symlink that other concurrent
# eval jobs (eval_task_addition.sh) rewrite to point at their own $SLURM_TMPDIR.
DATA_DIR="$PWD/artifacts/data/vision"

# Common parameters
NUM_BATCHES=10
BATCH_SIZE=32

# ===== Default experiments (no hyperparameter tuning) =====
# Wang released full-FT checkpoints only, so FT_MODES=(standard).
MODELS=(ViT-B-16 ViT-B-32 ViT-L-14)
METHODS=(wudi ace)
FT_MODES=(standard)
MERGE_MODE=d
HPO=''
# Task scenarios (Wang et al. / TALL-masks):
#   8 : Cars, DTD, EuroSAT, GTSRB, MNIST, RESISC45, SUN397, SVHN
#   14: 8 + CIFAR100, STL10, Flowers102, OxfordIIITPet, PCAM, FER2013
#   20: 14 + EMNIST, CIFAR10, Food101, FashionMNIST, RenderedSST2, KMNIST
EVAL_DATASETS="Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR100,STL10,Flowers102,OxfordIIITPet,PCAM,FER2013,EMNIST,CIFAR10,Food101,FashionMNIST,RenderedSST2,KMNIST"

# ===== Hyperparameter-optimized experiments =====
# NOTE: Only evaluate TA (sum) since other methods do not require HP tuning.
# MODELS=(ViT-B-16 ViT-B-32 ViT-L-14)
# METHODS=(sum)
# FT_MODES=(standard)
# HPO='{"alpha": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}'


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
