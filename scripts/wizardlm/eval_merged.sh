#!/bin/bash
#SBATCH --job-name=hf_eval_wizardlm
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err
# Submit with: sbatch --array=0-$((N-1)) scripts/wizardlm/eval_merged.sh
set -euo pipefail

source "$SCRATCH/actmat/.venv-olmo/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export HF_HOME="$SCRATCH/huggingface"

BASE_MODEL="meta-llama/Llama-2-13b-hf"
CHAT_TEMPLATE="vanillaOVO/WizardMath-13B-V1.0"
EXPERTS=(
  WizardLMTeam/WizardLM-13B-V1.2
  vanillaOVO/WizardMath-13B-V1.0
  layoric/llama-2-13b-code-alpaca
)
METHODS=(sum mean actmat tsv)
NUM_METHODS="${#METHODS[@]}"
if [ "$SLURM_ARRAY_TASK_ID" -ge "$NUM_METHODS" ]; then
  echo "No method for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
  exit 0
fi
METHOD="${METHODS[$SLURM_ARRAY_TASK_ID]}"
MERGED_DIR="artifacts/checkpoints/WizardLM/group-main/merged/${METHOD}"
RESULTS_BASE="artifacts/results/WizardLM/group-main/merged/${METHOD}"
EXPERT_STATS_DIR="artifacts/checkpoints/WizardLM/group-main/experts"

# 1. Merge (chat template from the math expert)
if [[ -f "$MERGED_DIR/model.safetensors.index.json" ]]; then
  echo ">>> Skipping merge: $MERGED_DIR already exists"
else
  python src/hf2/merge.py \
    --base-model-name-or-path "$BASE_MODEL" \
    --chat-template-name-or-path "$CHAT_TEMPLATE" \
    --expert-model-names-or-paths "${EXPERTS[@]}" \
    --merge-method "$METHOD" \
    --ignore-keep-pt 'embed_tokens|lm_head' \
    --expert-stats-dir "$EXPERT_STATS_DIR" \
    --output-dir "$MERGED_DIR"
fi

# 2. Evaluate
lm_eval --model hf \
  --model_args "pretrained=$MERGED_DIR" \
  --tasks gsm8k \
  --batch_size auto \
  --output_path "$RESULTS_BASE"
