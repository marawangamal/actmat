#!/bin/bash
#SBATCH --job-name=eval_t5_experts_intermediate
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=artifacts/logs/%x_%j.out
#SBATCH --error=artifacts/logs/%x_%j.err

set -euo pipefail
mkdir -p artifacts/logs

source ".venv-vl/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

NUM_TASKS="${NUM_TASKS:-7}"
FT_MODE="${FT_MODE:-fft}"
MODEL="${MODEL:-t5-base}"
GROUP="${GROUP:-$FT_MODE-$NUM_TASKS}"
CHECKPOINTS="${CHECKPOINTS:-auto}"
DATA_DIR="data"
CACHE_DIR="$SCRATCH/huggingface"
CKPT_ROOT="artifacts/checkpoints"
RESULTS_ROOT="artifacts/results"

DATASETS_7=(qasc wiki_qa quartz paws story_cloze winogrande wsc)
case "$NUM_TASKS" in
  7) DATASETS=("${DATASETS_7[@]}") ;;
  *) echo "Unsupported NUM_TASKS=$NUM_TASKS" >&2; exit 1 ;;
esac
EVAL_DATASETS="$(IFS=,; echo "${DATASETS[*]}")"
EXPERTS_DIR="$CKPT_ROOT/$MODEL/group-$GROUP/experts"

if [ "$CHECKPOINTS" = "auto" ]; then
  CHECKPOINT_ARRAY=()
  while IFS= read -r STEP; do
    SHARED=1
    for DATASET in "${DATASETS[@]}"; do
      if [ ! -f "$EXPERTS_DIR/$DATASET/checkpoint_$STEP.pt" ]; then
        SHARED=0
        break
      fi
    done
    if [ "$SHARED" = "1" ]; then
      CHECKPOINT_ARRAY+=("$STEP")
    fi
  done < <(
    find "$EXPERTS_DIR/${DATASETS[0]}" -maxdepth 1 -type f \
      -name 'checkpoint_*.pt' -printf '%f\n' \
      | sed -E 's/checkpoint_([0-9]+)\.pt/\1/' \
      | sort -n
  )
else
  read -r -a CHECKPOINT_ARRAY <<< "${CHECKPOINTS//,/ }"
fi

if [ "${#CHECKPOINT_ARRAY[@]}" -eq 0 ]; then
  echo "No checkpoints are shared by all experts in $EXPERTS_DIR" >&2
  exit 1
fi
echo "Shared checkpoint steps: ${CHECKPOINT_ARRAY[*]}"

OVERWRITE_ARGS=()
if [ "${OVERWRITE:-0}" = "1" ]; then
  OVERWRITE_ARGS=(--overwrite)
fi

for STEP in "${CHECKPOINT_ARRAY[@]}"; do
  if [[ ! "$STEP" =~ ^[0-9]+$ ]]; then
    echo "Invalid checkpoint step '$STEP'; expected a non-negative integer" >&2
    exit 1
  fi
  CHECKPOINT_NAME="checkpoint_$STEP.pt"
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

  OUT="$RESULTS_ROOT/$MODEL/group-$GROUP/intermediate/checkpoint_$STEP/experts"
  python scripts/t5/eval_experts.py \
    --model "$MODEL" \
    --experts-dir "$EXPERTS_DIR" \
    --eval-datasets "$EVAL_DATASETS" \
    --checkpoint-name "$CHECKPOINT_NAME" \
    --output-dir "$OUT" \
    --data-location "$DATA_DIR" \
    --cache-dir "$CACHE_DIR" \
    "${OVERWRITE_ARGS[@]}"
done
