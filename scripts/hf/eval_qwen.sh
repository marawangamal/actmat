#!/bin/bash
#SBATCH --job-name=hf_eval_qwen
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err
# Merge Qwen2.5-Coder-1.5B + Qwen2.5-Math-1.5B (base: Qwen2.5-1.5B), then eval
# math (gsm8k + MATH-500) + code (humaneval) with the OLMES harness.
#
# These are BASE (non-instruct) experts, so we use the base-model eval protocol:
# few-shot in-context completion with NO chat template, matching how Qwen2.5
# (§5.1) and AI2's OlmoBaseEval evaluate base models, and matching the
# eval_qwen_base.sh reference run. Applying a chat template (::tulu) to base
# models produces garbage, so no per-template views are needed — we eval the
# merged checkpoint directly.
#
# Submit with: sbatch --array=0-$((N-1)) scripts/hf/eval_qwen.sh
set -euo pipefail

source "$SCRATCH/actmat/.venv-olmo/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export HF_HOME="$SCRATCH/huggingface"
export SSL_CERT_DIR=/etc/ssl/certs

BASE_MODEL="Qwen/Qwen2.5-1.5B"
MATH_EXPERT="Qwen/Qwen2.5-Math-1.5B"
CODE_EXPERT="Qwen/Qwen2.5-Coder-1.5B"
METHODS=(mean tsv actmat_herm_10ki)
METHOD="${METHODS[$SLURM_ARRAY_TASK_ID]}"
MERGED_DIR="artifacts/checkpoints/Qwen2.5-1.5B/group-main/merged/${METHOD}"
RESULTS_BASE="artifacts/results/Qwen2.5-1.5B/group-main/merged/${METHOD}"

# Base-model (non-chat) eval recipes (same as eval_qwen_base.sh): gsm8k/MATH-500
# are 8/4-shot greedy exact_match; humaneval uses starcoder_pass@1 (0-shot
# raw-completion generative pass@1, temp 0.2) — the standard base-code-model
# setup. (The ::none humaneval variant is a bits-per-byte prompt, not clean
# generative pass@1, so it is NOT used.)
TASKS=(
  "gsm8k::olmes"
  "minerva_math_500::olmes"
  "codex_humaneval::starcoder_pass@1"
)
OLMES_MODEL_ARGS='{"gpu_memory_utilization": 0.8, "trust_remote_code": false, "max_length": 4096}'

# 1. Merge (skip if the merged checkpoint already exists; rm -rf "$MERGED_DIR"
# to force a re-merge). --chat-template-name-or-path is required by merge.py but
# unused at eval time (no chat template is applied for base-model eval).
if [[ -d "$MERGED_DIR" ]]; then
  echo ">>> Skipping merge: $MERGED_DIR already exists"
else
  python src/hf/merge.py \
    --base-model-name-or-path "$BASE_MODEL" \
    --chat-template-name-or-path "$MATH_EXPERT" \
    --expert-model-names-or-paths "$MATH_EXPERT" "$CODE_EXPERT" \
    --merge-method "$METHOD" \
    --output-dir "$MERGED_DIR"
fi

# 2. Evaluate the merged checkpoint directly (no chat template, no views)
olmes --model "$MERGED_DIR" --task "${TASKS[@]}" \
  --output-dir "$RESULTS_BASE" \
  --gpus 1 --model-type vllm --model-args "$OLMES_MODEL_ARGS" \
  --batch-size 64 --num-workers 1
