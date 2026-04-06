#!/bin/bash
#SBATCH --job-name=eval_lang_covs
#SBATCH --partition=main
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

# 0. Setup environment
source "$SCRATCH/eigcov/.venv/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

HF_CACHE_DIR="$SCRATCH/hf_cache"

# ── Configuration ────────────────────────────────────────────────────────
MODELS=(t5-base)
FT_MODES=(standard)
BATCH_SIZE=1
NUM_BATCHES_LIST=(1 10 100 500 1000)
RESULTS_DB="results/results.jsonl"
# ─────────────────────────────────────────────────────────────────────────

for MODEL in "${MODELS[@]}"; do
  for FT_MODE in "${FT_MODES[@]}"; do

    # 1. EigCov (data-free, run once)
    echo "[BASH] Running eval_task_addition.py | model: $MODEL | ft mode: $FT_MODE | method: eigcov"
    python scripts/language/eval_task_addition.py \
      --model="$MODEL" \
      --finetuning-mode="$FT_MODE" \
      --merge-func=eigcov \
      --results-db="$RESULTS_DB" \
      --hf-cache-dir="$HF_CACHE_DIR"

    # 2. Collect covariances for all sample counts in one pass
    NUM_BATCHES_CSV=$(IFS=,; echo "${NUM_BATCHES_LIST[*]}")
    echo "[BASH] Running covariance.py | model: $MODEL | n: $NUM_BATCHES_CSV | b: $BATCH_SIZE"
    python scripts/language/covariance.py \
      --model="$MODEL" \
      --finetuning-mode="$FT_MODE" \
      --cov-split=train \
      --cov-num-batches="$NUM_BATCHES_CSV" \
      --cov-batch-size="$BATCH_SIZE" \
      --cov-type=sm \
      --cov-estimator=full

    # 3. Evaluate RegMean at each sample count
    # NOTE: Covariances are now auto-discovered from checkpoint_dir/covariance.pt.
    # To ablate over different sample counts, re-collect covariance with the desired
    # settings before running eval.
    echo "[BASH] Running eval_task_addition.py | model: $MODEL | ft mode: $FT_MODE | method: regmean"
    python scripts/language/eval_task_addition.py \
      --model="$MODEL" \
      --finetuning-mode="$FT_MODE" \
      --merge-func=regmean \
      --results-db="$RESULTS_DB" \
      --hf-cache-dir="$HF_CACHE_DIR"
  done
done
