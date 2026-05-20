#!/bin/bash
# Download WizardLM-13B + WizardMath-13B + llama-2-13b-code-alpaca and organize
# into checkpoint structure mirroring the OLMo layout:
#
#   artifacts/checkpoints/Llama-2-13b-wizardlm/
#     pretrained/          (param folder — shared, meta-llama/Llama-2-13b-hf)
#     LM/
#       zeroshot/          (symlink → pretrained)
#       finetuned/         (WizardLM-13B-V1.2 param folder)
#     Math/
#       zeroshot/          (symlink → pretrained)
#       finetuned/         (WizardMath-13B-V1.0 param folder)
#     Code/
#       zeroshot/          (symlink → pretrained)
#       finetuned/         (llama-2-13b-code-alpaca param folder)
#
# Reproduces the DARE paper Fig. 1 (right) merging setup
# (Yu et al. 2024, https://arxiv.org/abs/2311.03099).
#
# Prerequisites:
#   - HF_TOKEN exported (meta-llama/Llama-2-13b-hf is gated).
#
# Usage:
#   bash scripts/wizardlm/download_models.sh
set -euo pipefail

PRETRAINED_ID="meta-llama/Llama-2-13b-hf"

# Map: capability_name -> HF model id
FINETUNED_TASKS=(
  "LM:WizardLMTeam/WizardLM-13B-V1.2"
  "Math:WizardLMTeam/WizardMath-13B-V1.0"
  "Code:layoric/llama-2-13b-code-alpaca"
)

BASE="artifacts/checkpoints/Llama-2-13b-wizardlm"
PRETRAINED_DIR="${BASE}/pretrained"

# 1. Download pretrained base
if [[ -d "$PRETRAINED_DIR" ]]; then
  echo ">>> Skipping pretrained: ${PRETRAINED_DIR} already exists"
else
  echo ">>> Downloading pretrained: ${PRETRAINED_ID}"
  python scripts/wizardlm/save_model_param_folder.py \
    --model "$PRETRAINED_ID" --output-dir "$PRETRAINED_DIR" --dtype bfloat16
fi

# 2. Download finetuned experts and symlink zeroshot -> pretrained
for entry in "${FINETUNED_TASKS[@]}"; do
  task="${entry%%:*}"     # LM, Math, Code
  hf_id="${entry#*:}"     # HF model id

  ft_dir="${BASE}/${task}/finetuned"
  if [[ -d "$ft_dir" ]]; then
    echo ">>> Skipping ${task}/finetuned: already exists"
  else
    echo ">>> Downloading ${task}/finetuned: ${hf_id}"
    python scripts/wizardlm/save_model_param_folder.py \
      --model "$hf_id" --output-dir "$ft_dir" --dtype bfloat16
  fi

  zs_link="${BASE}/${task}/zeroshot"
  rm -f "$zs_link"
  ln -s "$(realpath "$PRETRAINED_DIR")" "$zs_link"
  echo ">>> Symlinked ${task}/zeroshot -> $(realpath "$PRETRAINED_DIR")"
done
