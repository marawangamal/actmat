#!/bin/bash
# Merge + evaluate OLMo models.
#
# For each merge method: runs merge.py, optionally uploads to HF Hub,
# evaluates via olmes, then collects all results into a summary table.
#
# Usage:
#   bash scripts/olmo/eval_task_addition.sh

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
PRETRAINED_DIR="checkpoints/olmo/meta-llama-Meta-Llama-3.1-8B"
FINETUNED_DIRS=(
  checkpoints/olmo/pmahdavi-Llama-3.1-8B-math-reasoning
  checkpoints/olmo/pmahdavi-Llama-3.1-8B-coding
  checkpoints/olmo/pmahdavi-Llama-3.1-8B-precise-if
  checkpoints/olmo/pmahdavi-Llama-3.1-8B-general
  checkpoints/olmo/pmahdavi-Llama-3.1-8B-knowledge-recall
)
METHODS=(eigcov tsv mean isoc)
OUTPUT_BASE="checkpoints/olmo"
RESULTS_BASE="results"
GPUS=4
BATCH_SIZE=128

# ── Olmes config ─────────────────────────────────────────────────────────────
OLMES_TASKS=(
  "codex_humaneval::tulu"
  "codex_humanevalplus::tulu"
  "ifeval::tulu"
  "aime:zs_cot_r1::pass_at_32_2024_deepseek"
  "aime:zs_cot_r1::pass_at_32_2025_deepseek"
)
OLMES_MODEL_ARGS='{"gpu_memory_utilization": 0.8, "trust_remote_code": false, "max_length": 16384}'

# ── Auto-convert HF model IDs to param-folder format ────────────────────────
ensure_param_folder() {
  local dir="$1"
  if [[ -d "$dir" ]]; then
    PARAM_FOLDER_RESULT="$dir"
    return
  fi
  local local_dir="${OUTPUT_BASE}/$(echo "$dir" | tr '/' '-')"
  if [[ ! -d "$local_dir" ]]; then
    echo ">>> Downloading ${dir} to ${local_dir} ..."
    python scripts/olmo/save_model_param_folder.py --model "$dir" --output-dir "$local_dir"
  else
    echo ">>> Using cached ${local_dir} for ${dir}"
  fi
  PARAM_FOLDER_RESULT="$local_dir"
}

ensure_param_folder "$PRETRAINED_DIR"
PRETRAINED_DIR="$PARAM_FOLDER_RESULT"
NEW_FT_DIRS=()
for ft_dir in "${FINETUNED_DIRS[@]}"; do
  ensure_param_folder "$ft_dir"
  NEW_FT_DIRS+=("$PARAM_FOLDER_RESULT")
done
FINETUNED_DIRS=("${NEW_FT_DIRS[@]}")

# ── Run ──────────────────────────────────────────────────────────────────────
MODEL_PREFIX="$(basename "$PRETRAINED_DIR")"
RESULT_DIRS=()

for method in "${METHODS[@]}"; do
  RUN_NAME="${MODEL_PREFIX}-${method}"
  OUTPUT_DIR="${OUTPUT_BASE}/${RUN_NAME}"
  RESULTS_DIR="${RESULTS_BASE}/${RUN_NAME}"

  echo "============================================================"
  echo "Method     : ${method}"
  echo "Output     : ${OUTPUT_DIR}"
  echo "Results    : ${RESULTS_DIR}"
  echo "============================================================"

  # 1. Merge (skip if output dir already exists)
  if [[ -d "$OUTPUT_DIR" ]]; then
    echo ">>> Skipping merge: ${OUTPUT_DIR} already exists"
  else
    echo ">>> Merging with ${method} ..."
    python scripts/olmo/merge.py \
      --pretrained-dir "$PRETRAINED_DIR" \
      --finetuned-dirs "${FINETUNED_DIRS[@]}" \
      --merge-func "$method" \
      --output-dir "$OUTPUT_DIR"
  fi

  # 2. Evaluate via olmes (skip if results already exist)
  if ls "$RESULTS_DIR"/*-metrics-all.json &>/dev/null; then
    echo ">>> Skipping eval: ${RESULTS_DIR} already has results"
  else
    echo ">>> Evaluating with olmes ..."
    olmes \
      --model "$OUTPUT_DIR" \
      --task "${OLMES_TASKS[@]}" \
      --output-dir "$RESULTS_DIR" \
      --gpus "$GPUS" \
      --model-type vllm \
      --model-args "$OLMES_MODEL_ARGS" \
      --batch-size "$BATCH_SIZE"
  fi

  RESULT_DIRS+=("$RESULTS_DIR")
  echo ""
done

# 3. Collect results across all methods
echo "============================================================"
echo "Collecting results ..."
echo "============================================================"
python scripts/olmo/collect_results.py --dirs "${RESULT_DIRS[@]}"
