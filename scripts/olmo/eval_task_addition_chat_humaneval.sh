#!/bin/bash
#SBATCH --job-name=eval_olmo_chat_humaneval
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err
# Reduced variant of eval_task_addition_chat.sh as a SLURM job array: HumanEval
# only, one array task per ignore-keys variant
# (configs/olmo/ignore_top{10,25,50}pct.txt). Math view skipped -- HumanEval
# uses the Code chat template.
#
# Usage:
#   sbatch scripts/olmo/eval_task_addition_chat_humaneval.sh
set -euo pipefail
mkdir -p artifacts/logs

source "$SCRATCH/actmat/.venv-olmo/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL="Olmo-3-7b"
METHOD="actmat"
IGNORE_VARIANTS=(top10pct top25pct top50pct)

VARIANT="${IGNORE_VARIANTS[$SLURM_ARRAY_TASK_ID]}"

CODE_TASKS=("codex_humaneval::tulu")
OLMES_MODEL_ARGS='{"gpu_memory_utilization": 0.8, "trust_remote_code": false, "max_length": 16384}'
GPUS=1
BATCH_SIZE=64
NUM_WORKERS=1

CODE_CHAT_TEMPLATE="artifacts/checkpoints/${MODEL}/Code/finetuned/chat_template.jinja"

# ── Helpers ──────────────────────────────────────────────────────────────────
make_view() {
  local canonical="$1"
  local view="$2"
  local chat_template="$3"

  mkdir -p "$view"
  for f in "$canonical"/*; do
    ln -sfn "$(realpath "$f")" "${view}/$(basename "$f")"
  done
  ln -sfn "$(realpath "$chat_template")" "${view}/chat_template.jinja"
}

run_olmes() {
  local model_dir="$1"
  local results_dir="$2"
  shift 2
  local tasks=("$@")

  if [[ -f "${results_dir}/metrics.json" ]]; then
    echo ">>> Skipping eval: ${results_dir}/metrics.json already exists"
    return
  fi
  echo ">>> Evaluating: Batch size = $BATCH_SIZE, Number of workers = $NUM_WORKERS, GPUs = $GPUS"
  echo ">>> Model: $model_dir, tasks: ${tasks[*]}"
  olmes \
    --model "$model_dir" \
    --task "${tasks[@]}" \
    --output-dir "$results_dir" \
    --gpus "$GPUS" \
    --model-type vllm \
    --model-args "$OLMES_MODEL_ARGS" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS"
}

# ── Merge + Evaluate (one (method, variant) per array task) ─────────────────
IGNORE_FILE="configs/olmo/ignore_${VARIANT}.txt"
SUFFIX="${METHOD}-ignore-${VARIANT}"
MERGED_DIR="artifacts/checkpoints/${MODEL}/${SUFFIX}"
RESULTS_DIR="artifacts/results/${MODEL}-${SUFFIX}-chat"

echo "============================================================"
echo "Array task ${SLURM_ARRAY_TASK_ID}: Method=${METHOD} | Ignore=${IGNORE_FILE}"
echo "Merged: ${MERGED_DIR}"
echo "Results: ${RESULTS_DIR}"
echo "============================================================"

# 1. Merge (skip if already done)
if [[ -d "$MERGED_DIR" ]]; then
  echo ">>> Skipping merge: ${MERGED_DIR} already exists"
else
  python scripts/olmo/merge.py \
    --save "artifacts/checkpoints/${MODEL}" \
    --merge-func "$METHOD" \
    --ignore-keys-file "$IGNORE_FILE" \
    --output-dir "$MERGED_DIR"
fi

# 2. Materialize Code chat-template view
CODE_VIEW="${MERGED_DIR}-chat-code"
make_view "$MERGED_DIR" "$CODE_VIEW" "$CODE_CHAT_TEMPLATE"
echo ">>> View: ${CODE_VIEW} (Code chat template)"

# 3. Evaluate
run_olmes "$CODE_VIEW" "${RESULTS_DIR}/chat-code" "${CODE_TASKS[@]}"
