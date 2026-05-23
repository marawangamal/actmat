#!/bin/bash
#SBATCH --job-name=eval_roberta_models
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=artifacts/logs/%x_%j.out
#SBATCH --error=artifacts/logs/%x_%j.err

set -euo pipefail
mkdir -p artifacts/logs

# 0. Setup environment
source "$SCRATCH/actmat/.venv-vl/bin/activate"
export HF_HOME=$SCRATCH/huggingface
export NLTK_DATA=$SCRATCH/nltk_data
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

# ===== Default experiments (no hyperparameter tuning) =====
# roberta-base from lu-vae/roberta-glue, roberta-large from WUDI's Drive folder.
# Override via env: MODELS="roberta-large" sbatch scripts/roberta/eval_task_addition.sh
MODELS=(${MODELS:-roberta-base})
# Data-free merges. Skip regmean / fisher — they require per-task statistics
# collection, which is out of scope for this driver.
METHODS=(ace)
MERGE_MODE=d
HPO=''

for MODEL in "${MODELS[@]}"; do
  for method in "${METHODS[@]}"; do

    echo "[BASH] Running eval_task_addition.py | model: $MODEL | method: $method | mode: $MERGE_MODE"
    python scripts/roberta/eval_task_addition.py \
      --model="$MODEL" \
      --merge-func="$method" \
      --merge-mode="$MERGE_MODE" \
      ${HPO:+--hpo="$HPO"}

  done
done
