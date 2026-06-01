#!/bin/bash
#SBATCH --job-name=hf_eval_qwen
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err
# Merge Qwen2.5-Coder-1.5B + Qwen2.5-Math-1.5B (base: Qwen2.5-1.5B), then eval
# math (gsm8k) + code (humaneval) with lm-eval-harness.
# Submit with: sbatch --array=0-$((N-1)) scripts/hf/eval_qwen.sh
set -euo pipefail

source "$SCRATCH/actmat/.venv-olmo/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export HF_HOME="$SCRATCH/huggingface"

BASE_MODEL="Qwen/Qwen2.5-1.5B"
MATH_EXPERT="Qwen/Qwen2.5-Math-1.5B"
CODER_EXPERT="Qwen/Qwen2.5-Coder-1.5B"
METHODS=(sum mean actmat tsv)
METHOD="${METHODS[$SLURM_ARRAY_TASK_ID]}"
MERGED_DIR="artifacts/checkpoints/Qwen2.5-1.5B/group-main/merged/${METHOD}"

# 1. Merge (chat template from the math expert)
python src/hf/merge.py \
  --base-model-name-or-path "$BASE_MODEL" \
  --chat-template-name-or-path "$MATH_EXPERT" \
  --expert-model-names-or-paths "$MATH_EXPERT" "$CODER_EXPERT" \
  --merge-method "$METHOD" \
  --output-dir "$MERGED_DIR"

# 2. Evaluate (humaneval executes generated code -> --confirm_run_unsafe_code)
lm_eval --model hf \
  --model_args "pretrained=$MERGED_DIR" \
  --tasks gsm8k,humaneval \
  --batch_size auto \
  --confirm_run_unsafe_code \
  --output_path "artifacts/results/Qwen2.5-1.5B/group-main/merged/$METHOD"
