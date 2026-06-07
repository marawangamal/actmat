#!/bin/bash
#SBATCH --job-name=eval_t5_single
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err

set -euo pipefail
mkdir -p artifacts/logs

source ".venv-vl/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

NUM_TASKS="${NUM_TASKS:-7}"
FT_MODE="${FT_MODE:-fft}"
SINGLE_DIR="${SINGLE_DIR:-pretrained}"
EXPERT_DIR="${EXPERT_DIR:-}"
EVAL_DATASETS="${EVAL_DATASETS:-qasc,wiki_qa,quartz,paws,story_cloze,winogrande,wsc}"
DATA_DIR="data"
CACHE_DIR="$SCRATCH/huggingface"
CKPT_ROOT="artifacts/checkpoints"
RESULTS_ROOT="artifacts/results"
MODELS=(${MODELS:-t5-base t5-large})

if [ "$SLURM_ARRAY_TASK_ID" -ge "${#MODELS[@]}" ]; then
  echo "No model for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
  exit 0
fi
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
EXPERT_DIR="${EXPERT_DIR:-$CKPT_ROOT/$MODEL/group-$FT_MODE-$NUM_TASKS/$SINGLE_DIR}"
OUTPUT_DIR="$RESULTS_ROOT/$MODEL/group-$FT_MODE-$NUM_TASKS/$SINGLE_DIR"

python scripts/t5/eval_single.py \
  --model "$MODEL" \
  --expert-dir "$EXPERT_DIR" \
  --eval-datasets "$EVAL_DATASETS" \
  --output-dir "$OUTPUT_DIR" \
  --data-location "$DATA_DIR" \
  --cache-dir "$CACHE_DIR"
