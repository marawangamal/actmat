#!/bin/bash
#SBATCH --job-name=eval_t5_merge
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
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
RESULTS_ROOT="artifacts/results"

DATASETS_7="qasc,wiki_qa,quartz,paws,story_cloze,winogrande,wsc"
case "$NUM_TASKS" in
  7) EVAL_DATASETS="$DATASETS_7" ;;
  *) echo "Unsupported NUM_TASKS=$NUM_TASKS"; exit 1 ;;
esac

METHODS=(${METHODS:-mean isoc tsv actmat})
MODEL="${MODEL:-t5-base}"

NUM_METHODS="${#METHODS[@]}"
if [ "$SLURM_ARRAY_TASK_ID" -ge "$NUM_METHODS" ]; then
  echo "No method for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
  exit 0
fi
METHOD="${METHODS[$SLURM_ARRAY_TASK_ID]}"

# Standard regenerated checkpoints.
# EXPERTS_DIR="$CKPT_ROOT/$MODEL/group-$FT_MODE-$NUM_TASKS/experts"
# OUT="$RESULTS_ROOT/$MODEL/group-$FT_MODE-$NUM_TASKS/merged/$METHOD"

# Legacy FFT checkpoint sweep.
EXPERTS_DIR="$CKPT_ROOT/$MODEL/group-legacy-$FT_MODE-$NUM_TASKS/experts"
OUT="$RESULTS_ROOT/$MODEL/group-legacy-$FT_MODE-$NUM_TASKS/merged/$METHOD"

# Legacy LoRA checkpoint sweep.
# EXPERTS_DIR="$CKPT_ROOT/$MODEL/group-legacy-lora-$NUM_TASKS/experts"
# OUT="$RESULTS_ROOT/$MODEL/group-legacy-lora-$NUM_TASKS/merged/$METHOD"
python scripts/t5/eval_merged.py \
  --model "$MODEL" \
  --experts-dir "$EXPERTS_DIR" \
  --eval-datasets "$EVAL_DATASETS" \
  --merge-method "$METHOD" \
  --output-dir "$OUT" \
  --data-location "$DATA_DIR" \
  --cache-dir "$CACHE_DIR"
