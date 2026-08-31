#!/bin/bash
#SBATCH --job-name=eval_vit_merge_intermediate
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-4
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err

set -euo pipefail
mkdir -p artifacts/logs

source ".venv-vl/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

NUM_TASKS="${NUM_TASKS:-8}"
FT_MODE="${FT_MODE:-fft}"
MODEL="${MODEL:-ViT-B-16}"
CHECKPOINTS="${CHECKPOINTS:-0 200 400 600 800 1000 1200 1400}"
GROUP="${GROUP:-$FT_MODE-$NUM_TASKS}"

# Stage datasets to the node-local disk.
if [ ! -d "$SLURM_TMPDIR/data" ]; then
  cp downloads/data.tar.gz "$SLURM_TMPDIR/"
  tar -xzf "$SLURM_TMPDIR/data.tar.gz" -C "$SLURM_TMPDIR/"
fi
DATA_DIR="$SLURM_TMPDIR/data/vision"
OPENCLIP_DIR="$SCRATCH/openclip"
CKPT_ROOT="artifacts/checkpoints"
RESULTS_ROOT="artifacts/results"

DATASETS_8=(Cars DTD EuroSAT GTSRB MNIST RESISC45 SUN397 SVHN)
DATASETS_14=(Cars DTD EuroSAT GTSRB MNIST RESISC45 SUN397 SVHN CIFAR100 STL10 Flowers102 OxfordIIITPet PCAM FER2013)
DATASETS_20=(Cars DTD EuroSAT GTSRB MNIST RESISC45 SUN397 SVHN CIFAR100 STL10 Flowers102 OxfordIIITPet PCAM FER2013 EMNIST CIFAR10 Food101 FashionMNIST RenderedSST2 KMNIST)
case "$NUM_TASKS" in
  8) DATASETS=("${DATASETS_8[@]}") ;;
  14) DATASETS=("${DATASETS_14[@]}") ;;
  20) DATASETS=("${DATASETS_20[@]}") ;;
  *) echo "Unsupported NUM_TASKS=$NUM_TASKS" >&2; exit 1 ;;
esac
EVAL_DATASETS="$(IFS=,; echo "${DATASETS[*]}")"

read -r -a METHODS_ARRAY <<< "${METHODS:-mean isoc tsv actmat wudi}"
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#METHODS_ARRAY[@]}" ]; then
  echo "No method for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
  exit 0
fi
METHOD="${METHODS_ARRAY[$SLURM_ARRAY_TASK_ID]}"

# Accept either CHECKPOINTS="0 200 400" or CHECKPOINTS="0,200,400".
read -r -a CHECKPOINT_ARRAY <<< "${CHECKPOINTS//,/ }"
if [ "${#CHECKPOINT_ARRAY[@]}" -eq 0 ]; then
  echo "CHECKPOINTS must contain at least one step" >&2
  exit 1
fi

EXPERTS_DIR="$CKPT_ROOT/$MODEL/group-$GROUP/experts"
OVERWRITE_ARGS=()
if [ "${OVERWRITE:-0}" = "1" ]; then
  OVERWRITE_ARGS=(--overwrite)
fi

for STEP in "${CHECKPOINT_ARRAY[@]}"; do
  if [[ ! "$STEP" =~ ^[0-9]+$ ]]; then
    echo "Invalid checkpoint step '$STEP'; expected a non-negative integer" >&2
    exit 1
  fi

  CHECKPOINT_NAME="checkpoint_${STEP}.pt"
  MISSING=()
  for DATASET in "${DATASETS[@]}"; do
    if [ ! -f "$EXPERTS_DIR/$DATASET/$CHECKPOINT_NAME" ]; then
      MISSING+=("$DATASET")
    fi
  done
  if [ "${#MISSING[@]}" -gt 0 ]; then
    echo "$CHECKPOINT_NAME is missing for: ${MISSING[*]}" >&2
    exit 1
  fi

  OUT="$RESULTS_ROOT/$MODEL/group-$GROUP/intermediate/checkpoint_$STEP/$METHOD"
  python scripts/vit/eval_merged.py \
    --model "$MODEL" \
    --experts-dir "$EXPERTS_DIR" \
    --eval-datasets "$EVAL_DATASETS" \
    --merge-method "$METHOD" \
    --checkpoint-name "$CHECKPOINT_NAME" \
    --output-dir "$OUT" \
    --data-location "$DATA_DIR" \
    --cache-dir "$OPENCLIP_DIR" \
    "${OVERWRITE_ARGS[@]}"
done
