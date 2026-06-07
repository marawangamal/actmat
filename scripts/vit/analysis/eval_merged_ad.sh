#!/bin/bash
#SBATCH --job-name=eval_vit_merge_ad
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-6
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err

# Sweep regmean_interp over angular distances (covariance -> identity).
# AD is in units of pi (matching generate_error_terms.py): 0 = plain regmean,
# 0.5 = orthogonal; AD >= angle(c, I) saturates at identity (= mean), which
# for the ViT-B-16 fft-8 covariances happens at AD ~ 0.39-0.49.

set -euo pipefail
mkdir -p artifacts/logs

source ".venv-vl/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

NUM_TASKS="${NUM_TASKS:-8}"
FT_MODE="${FT_MODE:-fft}"

# Stage datasets to $SLURM_TMPDIR (mirrors scripts/vit/eval_merged.sh)
if [ ! -d "$SLURM_TMPDIR/data" ]; then
  cp downloads/data.tar.gz "$SLURM_TMPDIR/"
  tar -xzf "$SLURM_TMPDIR/data.tar.gz" -C "$SLURM_TMPDIR/"
fi
DATA_DIR="$SLURM_TMPDIR/data/vision"
OPENCLIP_DIR="$SCRATCH/openclip"
CKPT_ROOT="artifacts/checkpoints"
RESULTS_ROOT="artifacts/results"

DATASETS_8="Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN"
DATASETS_14="${DATASETS_8},CIFAR100,STL10,Flowers102,OxfordIIITPet,PCAM,FER2013"
DATASETS_20="${DATASETS_14},EMNIST,CIFAR10,Food101,FashionMNIST,RenderedSST2,KMNIST"
case "$NUM_TASKS" in
  8) EVAL_DATASETS="$DATASETS_8" ;;
  14) EVAL_DATASETS="$DATASETS_14" ;;
  20) EVAL_DATASETS="$DATASETS_20" ;;
  *) echo "Unsupported NUM_TASKS=$NUM_TASKS"; exit 1 ;;
esac

METHOD="regmean_interp"
ADS=(${ADS:-0.1 0.2 0.3 0.4 0.45 0.5 0.55})
# ViT-B-16 is the only model with a populated group-fft-8 tree (incl. covariance.pt)
MODELS=(${MODELS:-ViT-B-16})

NUM_ADS="${#ADS[@]}"
TOTAL_JOBS=$(("${#MODELS[@]}" * NUM_ADS))
if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL_JOBS" ]; then
  echo "No model/ad pair for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
  exit 0
fi
MODEL_IDX=$((SLURM_ARRAY_TASK_ID / NUM_ADS))
AD_IDX=$((SLURM_ARRAY_TASK_ID % NUM_ADS))
MODEL="${MODELS[$MODEL_IDX]}"
AD="${ADS[$AD_IDX]}"

EXPERTS_DIR="$CKPT_ROOT/$MODEL/group-$FT_MODE-$NUM_TASKS/experts"
OUT="$RESULTS_ROOT/$MODEL/group-$FT_MODE-$NUM_TASKS/merged/$METHOD-ad$AD"
python scripts/vit/eval_merged.py \
  --model "$MODEL" \
  --experts-dir "$EXPERTS_DIR" \
  --eval-datasets "$EVAL_DATASETS" \
  --merge-method "$METHOD" \
  --merge-kwargs "{\"angular_distance\": $AD}" \
  --output-dir "$OUT" \
  --data-location "$DATA_DIR" \
  --cache-dir "$OPENCLIP_DIR"
