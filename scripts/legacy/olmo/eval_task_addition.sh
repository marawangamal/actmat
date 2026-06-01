#!/bin/bash
#SBATCH --job-name=eval_olmo_chat
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=artifacts/logs/%x_%j.out
#SBATCH --error=artifacts/logs/%x_%j.err
# Merge + evaluate OLMo models via olmes, using the *matching* expert's chat
# template per task (Math template for AIME, Code/IF template for HumanEval &
# IFEval). The canonical merged dir today carries the Math chat template (see
# merge.py: tokenizer_dir defaults to OLMO_TASKS[0] == "Math"), which silently
# wraps every Code/IF prompt in "Solve the following math problem step by
# step..." -- this script fixes that by routing each task to a "view" of the
# merged model whose chat_template.jinja points at the right expert's.
#
# Per merged model we materialize two view directories:
#   ${MERGED_DIR}-chat-code   -- weights/tokenizer symlinked from ${MERGED_DIR},
#                                chat_template.jinja -> Code/finetuned
#   ${MERGED_DIR}-chat-math   -- same, but chat_template.jinja -> Math/finetuned
# (Code and IF share the same chat template, so two views cover all 3 experts.)
# Weights are NOT duplicated -- only chat_template.jinja is per-view.
#
# Prerequisites: run scripts/olmo/download_models.sh first.
#
# Usage:
#   sbatch scripts/olmo/eval_task_addition_chat.sh
set -euo pipefail
mkdir -p artifacts/logs

# 0. Setup environment
source "$SCRATCH/actmat/.venv-olmo/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL="Olmo-3-7b"
METHODS=(sum04)

# ── OLMES tasks, split by which expert's chat template they need ─────────────
CODE_TASKS=(
  "codex_humaneval::tulu"
  "codex_humanevalplus::tulu"
  "ifeval::tulu"
)
MATH_TASKS=(
  "aime:zs_cot_r1::pass_at_32_2024_deepseek"
  "aime:zs_cot_r1::pass_at_32_2025_deepseek"
)
OLMES_MODEL_ARGS='{"gpu_memory_utilization": 0.8, "trust_remote_code": false, "max_length": 16384}'
GPUS=1
BATCH_SIZE=64
NUM_WORKERS=1

# ── Chat-template sources (per-expert finetuned tokenizer dirs) ─────────────
CODE_CHAT_TEMPLATE="artifacts/checkpoints/${MODEL}/Code/finetuned/chat_template.jinja"
MATH_CHAT_TEMPLATE="artifacts/checkpoints/${MODEL}/Math/finetuned/chat_template.jinja"

# ── Helpers ──────────────────────────────────────────────────────────────────
# Materialize a "view" of a merged checkpoint that uses a specific chat template.
# Symlinks every file from the canonical merged dir, then overrides
# chat_template.jinja with a symlink to the supplied expert template.
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

# Run olmes for a (view, task-list, results-subdir) triple, skipping if done.
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

# ── Covariance (regmean only; script self-skips if covariance.pt exists) ────
if [[ " ${METHODS[*]} " =~ " regmean " ]]; then
  python scripts/olmo/covariance.py \
    --capability all \
    --save "artifacts/checkpoints/${MODEL}"
fi

# ── Merge + Evaluate ────────────────────────────────────────────────────────
for method in "${METHODS[@]}"; do
  MERGED_DIR="artifacts/checkpoints/${MODEL}/${method}"
  RESULTS_DIR="artifacts/results/${MODEL}-${method}-chat"

  echo "============================================================"
  echo "Method: ${method}"
  echo "Merged: ${MERGED_DIR}"
  echo "Results: ${RESULTS_DIR}"
  echo "============================================================"

  # 1. Merge (skip if already done)
  if [[ -d "$MERGED_DIR" ]]; then
    echo ">>> Skipping merge: ${MERGED_DIR} already exists"
  else
    python scripts/olmo/merge.py \
      --save "artifacts/checkpoints/${MODEL}" \
      --merge-func "$method" \
      --output-dir "$MERGED_DIR"
  fi

  # 2. Materialize per-template views of the merged checkpoint
  CODE_VIEW="${MERGED_DIR}-chat-code"
  MATH_VIEW="${MERGED_DIR}-chat-math"
  make_view "$MERGED_DIR" "$CODE_VIEW" "$CODE_CHAT_TEMPLATE"
  make_view "$MERGED_DIR" "$MATH_VIEW" "$MATH_CHAT_TEMPLATE"
  echo ">>> Views: ${CODE_VIEW} (Code/IF template), ${MATH_VIEW} (Math template)"

  # 3. Evaluate each task group against its matching view
  run_olmes "$CODE_VIEW" "${RESULTS_DIR}/chat-code" "${CODE_TASKS[@]}"
  # AIME/math skipped for sum04: paper-snapshot AIME already in *-reb-chat.
  # run_olmes "$MATH_VIEW" "${RESULTS_DIR}/chat-math" "${MATH_TASKS[@]}"
done
