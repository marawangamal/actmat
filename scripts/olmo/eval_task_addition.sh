#!/bin/bash
#SBATCH --job-name=eval_olmo
#SBATCH --partition=long
#SBATCH --array=0
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err
# Merge + evaluate OLMo task vectors via olmes, with per-task chat templates.
# Tasks: HumanEval, HumanEval+, IFEval (Code chat template) and GSM8K +
# MATH-500 (Math chat template). AIME is NOT included here -- see
# eval_task_addition_old.sh / eval_task_addition_chat.sh for the AIME variant.
#
# Job array: one task per method in METHODS (index by $SLURM_ARRAY_TASK_ID).
# Adjust #SBATCH --array=0-N to match len(METHODS)-1.
#
# Per merged model, two chat-template views are materialized:
#   ${MERGED_DIR}-chat-code   -- Code/IF chat template (HumanEval/+/IFEval)
#   ${MERGED_DIR}-chat-math   -- Math chat template (GSM8K/MATH-500)
# Code/IF results: ${RESULTS_DIR}/chat-code
# GSM8K + MATH-500 results: ${RESULTS_DIR}/chat-gsm8k-math
#
# Usage:
#   sbatch scripts/olmo/eval_task_addition.sh
set -euo pipefail
mkdir -p artifacts/logs

# 0. Setup environment
source "$SCRATCH/actmat/.venv-olmo/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL="Olmo-3-7b"
METHODS=(regmean)
method="${METHODS[${SLURM_ARRAY_TASK_ID:-0}]}"

# ── OLMES tasks, split by which expert's chat template they need ─────────────
CODE_TASKS=(
  "codex_humaneval::tulu"
  "codex_humanevalplus::tulu"
  "ifeval::tulu"
)
# GSM8K::tulu: 8-shot CoT, greedy, max_gen_toks=512, ~1319 problems.
# MATH-500::tulu: 4-shot, greedy, max_gen_toks=1024, 500-problem subset of
# Hendrycks MATH (Lightman et al., OpenAI PRM paper). ~10x faster than full
# minerva_math (5000 problems) and the standard reported by R1/o1/etc.
MATH_TASKS=(
  "gsm8k::tulu"
  "minerva_math_500::tulu"
)
OLMES_MODEL_ARGS='{"gpu_memory_utilization": 0.8, "trust_remote_code": false, "max_length": 16384}'
# Code tasks override: HumanEval/+ default to max_gen_toks=999999 (capped by
# max_length). Empirically generations never exceed 604 tokens, so 2048 is
# ample. IFEval already defaults to 2048, so this is a no-op for it. Applied
# uniformly to all CODE_TASKS via --task-args.
CODE_TASK_ARGS='{"generation_kwargs": {"max_gen_toks": 2048}}'
GPUS=1
BATCH_SIZE=64
NUM_WORKERS=1

# ── Chat-template sources (per-expert finetuned tokenizer dirs) ─────────────
CODE_CHAT_TEMPLATE="artifacts/checkpoints/${MODEL}/Code/finetuned/chat_template.jinja"
MATH_CHAT_TEMPLATE="artifacts/checkpoints/${MODEL}/Math/finetuned/chat_template.jinja"

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
  local task_args="$3"   # JSON dict, "" for no override
  shift 3
  local tasks=("$@")

  if [[ -f "${results_dir}/metrics.json" ]]; then
    echo ">>> Skipping eval: ${results_dir}/metrics.json already exists"
    return
  fi
  echo ">>> Evaluating: Batch size = $BATCH_SIZE, Number of workers = $NUM_WORKERS, GPUs = $GPUS"
  echo ">>> Model: $model_dir, tasks: ${tasks[*]}"
  local extra=()
  if [[ -n "$task_args" ]]; then
    extra=(--task-args "$task_args")
    echo ">>> Task args override: $task_args"
  fi
  olmes \
    --model "$model_dir" \
    --task "${tasks[@]}" \
    --output-dir "$results_dir" \
    --gpus "$GPUS" \
    --model-type vllm \
    --model-args "$OLMES_MODEL_ARGS" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    "${extra[@]}"
}

# ── Covariance (regmean only; script self-skips if covariance.pt exists) ────
if [[ "$method" == "regmean" ]]; then
  python scripts/olmo/covariance.py \
    --capability all \
    --save "artifacts/checkpoints/${MODEL}"
fi

# ── Merge + Evaluate (single method per array task) ─────────────────────────
MERGED_DIR="artifacts/checkpoints/${MODEL}/${method}"
RESULTS_DIR="artifacts/results/${MODEL}-${method}-chat"

echo "============================================================"
echo "Array task: ${SLURM_ARRAY_TASK_ID:-0} / ${#METHODS[@]}"
echo "Method:     ${method}"
echo "Merged:     ${MERGED_DIR}"
echo "Results:    ${RESULTS_DIR}"
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
run_olmes "$CODE_VIEW" "${RESULTS_DIR}/chat-code"      "$CODE_TASK_ARGS" "${CODE_TASKS[@]}"
run_olmes "$MATH_VIEW" "${RESULTS_DIR}/chat-gsm8k-math" ""                "${MATH_TASKS[@]}"
