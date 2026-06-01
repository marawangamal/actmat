#!/bin/bash
# Download Llama-3.1-8B-Instruct MergeBench models and organize into checkpoint structure:
#
#   artifacts/checkpoints/Llama-3.1-8B-Instruct/
#     pretrained/              (param folder — shared)
#     instruction/
#       pretrained/            (symlink → ../pretrained)
#       finetuned/             (param folder)
#     math/
#       pretrained/            (symlink → ../pretrained)
#       finetuned/             (param folder)
#     coding/
#       pretrained/            (symlink → ../pretrained)
#       finetuned/             (param folder)
#     multilingual/
#       pretrained/            (symlink → ../pretrained)
#       finetuned/             (param folder)
#
# Pretrained is Meta-gated; run `huggingface-cli login` first.
#
# Usage:
#   bash scripts/llama8b/download_models.sh
set -euo pipefail

PRETRAINED_ID="meta-llama/Llama-3.1-8B-Instruct"
FINETUNED_IDS=(
  "MergeBench/Llama-3.1-8B-Instruct_instruction"
  "MergeBench/Llama-3.1-8B-Instruct_math"
  "MergeBench/Llama-3.1-8B-Instruct_coding"
  "MergeBench/Llama-3.1-8B-Instruct_multilingual"
)

BASE="artifacts/checkpoints/Llama-3.1-8B-Instruct"
PRETRAINED_DIR="${BASE}/pretrained"

# 1. Download pretrained model
if [[ -d "$PRETRAINED_DIR/params" ]] && [[ -n "$(ls -A "$PRETRAINED_DIR/params" 2>/dev/null)" ]]; then
  echo ">>> Skipping pretrained: ${PRETRAINED_DIR} already exists with params"
else
  rm -rf "$PRETRAINED_DIR"
  echo ">>> Downloading pretrained: ${PRETRAINED_ID}"
  python scripts/llama8b/save_model_param_folder.py --model "$PRETRAINED_ID" --output-dir "$PRETRAINED_DIR"
fi

# 2. Download finetuned models and symlink per-task pretrained → shared pretrained
for hf_id in "${FINETUNED_IDS[@]}"; do
  task="${hf_id##*_}"  # extract suffix after final underscore: instruction, math, coding, multilingual

  # Download finetuned — verify params/ is non-empty (timed-out downloads can
  # leave config/tokenizer behind with no params files).
  ft_dir="${BASE}/${task}/finetuned"
  if [[ -d "$ft_dir/params" ]] && [[ -n "$(ls -A "$ft_dir/params" 2>/dev/null)" ]]; then
    echo ">>> Skipping ${task}/finetuned: already exists with params"
  else
    rm -rf "$ft_dir"
    echo ">>> Downloading ${task}/finetuned: ${hf_id}"
    python scripts/llama8b/save_model_param_folder.py --model "$hf_id" --output-dir "$ft_dir"
  fi

  # Symlink per-task pretrained → shared pretrained (always refresh to handle re-downloads)
  pre_link="${BASE}/${task}/pretrained"
  rm -f "$pre_link"
  ln -s "$(realpath "$PRETRAINED_DIR")" "$pre_link"
  echo ">>> Symlinked ${task}/pretrained → $(realpath "$PRETRAINED_DIR")"
done
