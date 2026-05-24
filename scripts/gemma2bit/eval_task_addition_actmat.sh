#!/bin/bash
#SBATCH --job-name=eval_gemma2bit_actmat
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=artifacts/logs/%x_%j.out
#SBATCH --error=artifacts/logs/%x_%j.err
# Parallel companion to eval_task_addition.sh — runs ONLY the actmat method
# and writes results to artifacts/results-collision-avoid/ so it does not
# collide with the main driver should it also reach actmat later.
#
# The merged checkpoint is shared at artifacts/checkpoints/gemma-2-2b-it/actmat/
# (whichever job creates it first wins; the other skips merge).
set -euo pipefail
mkdir -p artifacts/logs

source "$SCRATCH/actmat/.venv-gemma/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="gemma-2-2b-it"
METHODS=(actmat)

OLMES_TASKS=(
  "ifeval::tulu"
  "gsm8k::tulu"
  "codex_humanevalplus::tulu"
  "mbppplus:0-shot-chat"
)
OLMES_MODEL_ARGS='{"trust_remote_code": false, "max_length": 8192, "dtype": "bfloat16"}'
GPUS=1
BATCH_SIZE=2
NUM_WORKERS=1

LM_EVAL_TASKS="m_mmlu_fr,arc_fr,hellaswag_fr,m_mmlu_es,arc_es,hellaswag_es,m_mmlu_de,arc_de,hellaswag_de,m_mmlu_ru,arc_ru,hellaswag_ru"
LM_EVAL_BATCH_SIZE=8

RESULTS_BASE="artifacts/results-collision-avoid"

for method in "${METHODS[@]}"; do
  MERGED_DIR="artifacts/checkpoints/${MODEL}/${method}"
  RESULTS_DIR="${RESULTS_BASE}/${MODEL}-${method}"
  OLMES_RESULTS_DIR="${RESULTS_DIR}/olmes"
  LM_EVAL_RESULTS_DIR="${RESULTS_DIR}/lm-eval"

  echo "============================================================"
  echo "Method: ${method}"
  echo "Merged: ${MERGED_DIR}"
  echo "Results: ${RESULTS_DIR}"
  echo "============================================================"

  if [[ -d "$MERGED_DIR" ]]; then
    echo ">>> Skipping merge: ${MERGED_DIR} already exists"
  else
    python scripts/gemma2bit/merge.py \
      --save "artifacts/checkpoints/${MODEL}" \
      --merge-func "$method" \
      --output-dir "$MERGED_DIR"
  fi

  if [[ -f "$OLMES_RESULTS_DIR/metrics.json" ]]; then
    echo ">>> Skipping olmes: ${OLMES_RESULTS_DIR}/metrics.json already exists"
  else
    echo ">>> olmes eval (hf backend): tasks = ${OLMES_TASKS[*]}"
    olmes \
      --model "$MERGED_DIR" \
      --task "${OLMES_TASKS[@]}" \
      --output-dir "$OLMES_RESULTS_DIR" \
      --gpus "$GPUS" \
      --model-type hf \
      --model-args "$OLMES_MODEL_ARGS" \
      --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS"
  fi

  if [[ -n "$(find "$LM_EVAL_RESULTS_DIR" -name 'results_*.json' -print -quit 2>/dev/null)" ]]; then
    echo ">>> Skipping lm-eval: ${LM_EVAL_RESULTS_DIR} already has results"
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
