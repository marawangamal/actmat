#!/bin/bash
#SBATCH --job-name=hf_eval_wizardlm
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err
# Submit with: sbatch --array=0-$((N-1)) scripts/wizardlm/eval_wizardlm.sh
set -euo pipefail

source "$SCRATCH/actmat/.venv-olmo/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export HF_HOME="$SCRATCH/huggingface"

BASE_MODEL="meta-llama/Llama-2-13b-hf"
CHAT_TEMPLATE="vanillaOVO/WizardMath-13B-V1.0"
METHODS=(sum mean actmat tsv)
METHOD="${METHODS[$SLURM_ARRAY_TASK_ID]}"
MERGED_DIR="artifacts/checkpoints/WizardLM/group-main/merged/${METHOD}"

# 1. Merge (chat template from the math expert)
python src/hf/merge.py \
  --base-model-name-or-path "$BASE_MODEL" \
  --chat-template-name-or-path "$CHAT_TEMPLATE" \
  --expert-model-names-or-paths "WizardLMTeam/WizardLM-13B-V1.2" "vanillaOVO/WizardMath-13B-V1.0" "layoric/llama-2-13b-code-alpaca" \
  --merge-method "$METHOD" \
  --output-dir "$MERGED_DIR" \
  --ignore-keep-pt 'embed_tokens|lm_head'

# 2. Evaluate
lm_eval --model hf \
  --model_args "pretrained=$MERGED_DIR" \
  --tasks gsm8k \
  --batch_size auto \
  --output_path "artifacts/results/WizardLM/group-main/merged/$METHOD"
