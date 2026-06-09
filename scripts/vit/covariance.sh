#!/bin/bash
#SBATCH --job-name=cov_vit
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err

set -euo pipefail
mkdir -p artifacts/logs

source ".venv-vl/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

NUM_TASKS="${NUM_TASKS:-8}"
FT_MODE="${FT_MODE:-fft}"
DATA_DIR="data/vision"
OPENCLIP_DIR="$SCRATCH/openclip"
CKPT_ROOT="artifacts/checkpoints"

DATASETS_8=(Cars DTD EuroSAT GTSRB MNIST RESISC45 SUN397 SVHN)
DATASETS_14=(Cars DTD EuroSAT GTSRB MNIST RESISC45 SUN397 SVHN CIFAR100 STL10 Flowers102 OxfordIIITPet PCAM FER2013)
DATASETS_20=(Cars DTD EuroSAT GTSRB MNIST RESISC45 SUN397 SVHN CIFAR100 STL10 Flowers102 OxfordIIITPet PCAM FER2013 EMNIST CIFAR10 Food101 FashionMNIST RenderedSST2 KMNIST)
case "$NUM_TASKS" in
  8) DATASETS=("${DATASETS_8[@]}") ;;
  14) DATASETS=("${DATASETS_14[@]}") ;;
  20) DATASETS=("${DATASETS_20[@]}") ;;
  *) echo "Unsupported NUM_TASKS=$NUM_TASKS"; exit 1 ;;
esac

DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
MODEL="${MODEL:-ViT-B-16}"

EXPERT_DIR="$CKPT_ROOT/$MODEL/group-$FT_MODE-$NUM_TASKS/experts/$DATASET"
python scripts/vit/covariance.py \
  --model "$MODEL" \
  --expert-dir "$EXPERT_DIR" \
  --dataset "$DATASET" \
  --output-path "$EXPERT_DIR/covariance.pt" \
  --data-location "$DATA_DIR" \
  --cache-dir "$OPENCLIP_DIR"
