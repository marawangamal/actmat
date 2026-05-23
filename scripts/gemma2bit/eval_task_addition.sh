#!/bin/bash
#SBATCH --job-name=eval_gemma2bit
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=artifacts/logs/%x_%j.out
#SBATCH --error=artifacts/logs/%x_%j.err
# Merge + evaluate Gemma-2-2B-IT (MergeBench) models via olmes (instruction/math/code)
# and lm-eval (multilingual).
#
# Prerequisites:
#   UV_PROJECT_ENVIRONMENT=.venv-gemma uv sync --group gemma
#   bash scripts/gemma2bit/download_models.sh
#
# Usage:
#   sbatch scripts/gemma2bit/eval_task_addition.sh
set -euo pipefail
mkdir -p artifacts/logs

# 0. Setup environment
source "$SCRATCH/actmat/.venv-gemma/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL="gemma-2-2b-it"
METHODS=(mean tsv isoc actmat)

# ── OLMES (instruction, math, code) ───────────────────────────────────────────
OLMES_TASKS=(
  "ifeval::tulu"
  "gsm8k::tulu"
  "codex_humanevalplus::tulu"
  "mbppplus:0-shot-chat"
)
OLMES_MODEL_ARGS='{"gpu_memory_utilization": 0.8, "trust_remote_code": false, "max_length": 8192}'
GPUS=1
BATCH_SIZE=64
NUM_WORKERS=1

# ── lm-eval (multilingual) ────────────────────────────────────────────────────
LM_EVAL_TASKS="m_mmlu_fr,arc_fr,hellaswag_fr,m_mmlu_es,arc_es,hellaswag_es,m_mmlu_de,arc_de,hellaswag_de,m_mmlu_ru,arc_ru,hellaswag_ru"
LM_EVAL_BATCH_SIZE=8

# ── Merge + Evaluate ────────────────────────────────────────────────────────
for method in "${METHODS[@]}"; do
  MERGED_DIR="artifacts/checkpoints/${MODEL}/${method}"
  RESULTS_DIR="artifacts/results/${MODEL}-${method}"
  OLMES_RESULTS_DIR="${RESULTS_DIR}/olmes"
  LM_EVAL_RESULTS_DIR="${RESULTS_DIR}/lm-eval"

  echo "============================================================"
  echo "Method: ${method}"
  echo "Merged: ${MERGED_DIR}"
  echo "Results: ${RESULTS_DIR}"
  echo "============================================================"

  # 1. Merge (skip if already done)
  if [[ -d "$MERGED_DIR" ]]; then
    echo ">>> Skipping merge: ${MERGED_DIR} already exists"
  else
    python scripts/gemma2bit/merge.py \
      --save "artifacts/checkpoints/${MODEL}" \
      --merge-func "$method" \
      --output-dir "$MERGED_DIR"
  fi

  # 2a. olmes — instruction, math, code (skip if metrics.json exists)
  if [[ -f "$OLMES_RESULTS_DIR/metrics.json" ]]; then
    echo ">>> Skipping olmes: ${OLMES_RESULTS_DIR}/metrics.json already exists"
  else
    echo ">>> olmes eval: tasks = ${OLMES_TASKS[*]}"
    olmes \
      --model "$MERGED_DIR" \
      --task "${OLMES_TASKS[@]}" \
      --output-dir "$OLMES_RESULTS_DIR" \
      --gpus "$GPUS" \
      --model-type vllm \
      --model-args "$OLMES_MODEL_ARGS" \
      --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS"
  fi

  # 2b. lm-eval — multilingual (skip if results files exist)
  if compgen -G "${LM_EVAL_RESULTS_DIR}/results_*.json" > /dev/null; then
    echo ">>> Skipping lm-eval: ${LM_EVAL_RESULTS_DIR}/results_*.json already exists"
  else
    mkdir -p "$LM_EVAL_RESULTS_DIR"
    echo ">>> lm-eval: tasks = ${LM_EVAL_TASKS}"
    lm_eval \
      --model hf \
      --model_args "pretrained=${MERGED_DIR},dtype=bfloat16" \
      --tasks "$LM_EVAL_TASKS" \
      --batch_size "$LM_EVAL_BATCH_SIZE" \
      --output_path "$LM_EVAL_RESULTS_DIR"
  fi
done
