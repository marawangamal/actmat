#!/bin/bash
#SBATCH --job-name=eval_wizardlm
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=artifacts/logs/%x_%j.out
#SBATCH --error=artifacts/logs/%x_%j.err
# Merge + evaluate WizardLM-13B / WizardMath-13B / llama-2-13b-code-alpaca via
# olmes, then collect results. Reproduces the DARE paper Fig. 1 (right).
#
# Prerequisites:
#   - bash scripts/wizardlm/download_models.sh (HF_TOKEN required for Llama-2)
#   - OPENAI_API_KEY exported for alpaca_eval_v2 (GPT-4 judge)
#
# Usage:
#   sbatch scripts/wizardlm/eval_task_addition.sh
set -euo pipefail
mkdir -p artifacts/logs

# 0. Setup environment
source "$SCRATCH/actmat/.venv-olmo/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL="Llama-2-13b-wizardlm"
METHODS=(sum mean ties dare tsv isoc regmean actmat)

# Per-method extra merge kwargs (JSON). Empty string for none.
declare -A MERGE_KWARGS=(
  ["dare"]='{"drop_rate": 0.5, "seed": 0, "base_merge": "sum"}'
)

STATS_METHODS=(regmean actmat)

# ── OLMES ─────────────────────────────────────────────────────────────────────
OLMES_TASKS=(
  "gsm8k::tulu"
  "codex_humaneval::tulu"
  "mbpp:3shot::none"
  "alpaca_eval_v2::tulu"
)
OLMES_MODEL_ARGS='{"gpu_memory_utilization": 0.8, "trust_remote_code": false, "max_length": 4096}'
GPUS=2
BATCH_SIZE=32
NUM_WORKERS=1

# ── Statistics collection (regmean / actmat) ─────────────────────────────────
need_stats=0
for m in "${METHODS[@]}"; do
  for sm in "${STATS_METHODS[@]}"; do
    if [[ "$m" == "$sm" ]]; then need_stats=1; fi
  done
done
if [[ $need_stats -eq 1 ]]; then
  echo ">>> Collecting covariances for ${MODEL}"
  python scripts/wizardlm/covariance.py \
    --capability all \
    --save "artifacts/checkpoints/${MODEL}"
fi

# ── Merge + Evaluate ────────────────────────────────────────────────────────
for method in "${METHODS[@]}"; do
  MERGED_DIR="artifacts/checkpoints/${MODEL}/${method}"
  RESULTS_DIR="artifacts/results/${MODEL}-${method}"

  echo "============================================================"
  echo "Method: ${method}"
  echo "Merged: ${MERGED_DIR}"
  echo "Results: ${RESULTS_DIR}"
  echo "============================================================"

  # 1. Merge (skip if already done)
  if [[ -d "$MERGED_DIR" ]]; then
    echo ">>> Skipping merge: ${MERGED_DIR} already exists"
  else
    extra_args=()
    if [[ -n "${MERGE_KWARGS[$method]:-}" ]]; then
      extra_args+=(--merge-kwargs "${MERGE_KWARGS[$method]}")
    fi
    python scripts/wizardlm/merge.py \
      --save "artifacts/checkpoints/${MODEL}" \
      --merge-func "$method" \
      --output-dir "$MERGED_DIR" \
      "${extra_args[@]}"
  fi

  # 2. Evaluate (skip if metrics.json already exists)
  if [[ -f "$RESULTS_DIR/metrics.json" ]]; then
    echo ">>> Skipping eval: ${RESULTS_DIR}/metrics.json already exists"
  else
    echo ">>> Evaluating: Batch size = $BATCH_SIZE, Workers = $NUM_WORKERS, GPUs = $GPUS"
    echo ">>> Model: $MERGED_DIR, tasks: ${OLMES_TASKS[@]}"
    olmes \
      --model "$MERGED_DIR" \
      --task "${OLMES_TASKS[@]}" \
      --output-dir "$RESULTS_DIR" \
      --gpus "$GPUS" \
      --model-type vllm \
      --model-args "$OLMES_MODEL_ARGS" \
      --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS"
  fi
done
