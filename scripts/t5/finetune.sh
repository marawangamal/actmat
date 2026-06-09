#!/bin/bash
#SBATCH --job-name=finetune_t5
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-6
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err

set -euo pipefail
mkdir -p artifacts/logs

source ".venv-vl/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

NUM_TASKS="${NUM_TASKS:-7}"
FT_MODE="${FT_MODE:-fft}"
DATA_DIR="data"
CACHE_DIR="$SCRATCH/huggingface"
CKPT_ROOT="artifacts/checkpoints"

DATASETS_7=(qasc wiki_qa quartz paws story_cloze winogrande wsc)
case "$NUM_TASKS" in
  7) DATASETS=("${DATASETS_7[@]}") ;;
  *) echo "Unsupported NUM_TASKS=$NUM_TASKS"; exit 1 ;;
esac

DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
MODEL="${MODEL:-t5-base}"
PORT=$((12355 + SLURM_ARRAY_TASK_ID))

if [ "$DATASET" = "story_cloze" ] && [ ! -f data/language/story_cloze/cloze_validation_2016.csv ]; then
  if [ -L data ] && [ ! -e data ]; then
    rm data
  fi
  mkdir -p data/language/story_cloze
  tar -xzf downloads/data.tar.gz -C . \
    data/language/story_cloze/cloze_test_2016.csv \
    data/language/story_cloze/cloze_validation_2016.csv
fi

OVERWRITE_ARGS=()
if [ "${OVERWRITE:-0}" = "1" ]; then
  OVERWRITE_ARGS=(--overwrite)
fi

OUT="$CKPT_ROOT/$MODEL/group-$FT_MODE-$NUM_TASKS/experts/$DATASET"
python scripts/t5/finetune.py \
  --model "$MODEL" \
  --train-dataset "$DATASET" \
  --finetuning-mode "$FT_MODE" \
  --output-dir "$OUT" \
  --world-size 1 \
  --num-workers 1 \
  --port "$PORT" \
  --cache-dir "$CACHE_DIR" \
  --data-location "$DATA_DIR" \
  --wandb \
  --early-stop \
  --grad-cross-matrix \
  --checkpoint-every 200 \
  "${OVERWRITE_ARGS[@]}"
