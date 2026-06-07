#!/bin/bash
#SBATCH --job-name=eval_vit_merge
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-11
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

METHODS=(${METHODS:-mean isoc tsv actmat})
MODELS=(${MODELS:-ViT-B-16 ViT-B-32 ViT-L-14})

NUM_METHODS="${#METHODS[@]}"
TOTAL_JOBS=$(("${#MODELS[@]}" * NUM_METHODS))
if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL_JOBS" ]; then
  echo "No model/method pair for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
  exit 0
fi
MODEL_IDX=$((SLURM_ARRAY_TASK_ID / NUM_METHODS))
METHOD_IDX=$((SLURM_ARRAY_TASK_ID % NUM_METHODS))
MODEL="${MODELS[$MODEL_IDX]}"
METHOD="${METHODS[$METHOD_IDX]}"

EXPERTS_DIR="$CKPT_ROOT/$MODEL/group-$FT_MODE-$NUM_TASKS/experts"
OUT="$RESULTS_ROOT/$MODEL/group-$FT_MODE-$NUM_TASKS/merged/$METHOD"
python scripts/vit/eval_merged.py \
  --model "$MODEL" \
  --experts-dir "$EXPERTS_DIR" \
  --eval-datasets "$EVAL_DATASETS" \
  --merge-method "$METHOD" \
  --output-dir "$OUT" \
  --data-location "$DATA_DIR" \
  --cache-dir "$OPENCLIP_DIR"
