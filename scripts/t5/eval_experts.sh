#!/bin/bash
#SBATCH --job-name=eval_t5_experts
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
DATA_DIR="data"
CACHE_DIR="$SCRATCH/huggingface"
CKPT_ROOT="artifacts/checkpoints"
RESULTS_ROOT="artifacts/results"

DATASETS_7="qasc,wiki_qa,quartz,paws,story_cloze,winogrande,wsc"
case "$NUM_TASKS" in
  7) EVAL_DATASETS="$DATASETS_7" ;;
  *) echo "Unsupported NUM_TASKS=$NUM_TASKS"; exit 1 ;;
esac

MODEL="${MODEL:-t5-base}"

EXPERTS_DIR="$CKPT_ROOT/$MODEL/group-$FT_MODE-$NUM_TASKS/experts"
OUT="$RESULTS_ROOT/$MODEL/group-$FT_MODE-$NUM_TASKS/experts"
python scripts/t5/eval_experts.py \
  --model "$MODEL" \
  --experts-dir "$EXPERTS_DIR" \
  --eval-datasets "$EVAL_DATASETS" \
  --output-dir "$OUT" \
  --data-location "$DATA_DIR" \
  --cache-dir "$CACHE_DIR"
