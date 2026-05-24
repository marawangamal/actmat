#!/bin/bash
# Download Gemma-2-2B-IT MergeBench models and organize into checkpoint structure:
#
#   artifacts/checkpoints/gemma-2-9b-it/
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
# Usage:
#   bash scripts/gemma9bit/download_models.sh
set -euo pipefail

PRETRAINED_ID="google/gemma-2-9b-it"
FINETUNED_IDS=(
  "MergeBench/gemma-2-9b-it_instruction"
  "MergeBench/gemma-2-9b-it_math"
  "MergeBench/gemma-2-9b-it_coding"
  "MergeBench/gemma-2-9b-it_multilingual"
)

BASE="artifacts/checkpoints/gemma-2-9b-it"
PRETRAINED_DIR="${BASE}/pretrained"

# 1. Download pretrained model
if [[ -d "$PRETRAINED_DIR" ]]; then
  echo ">>> Skipping pretrained: ${PRETRAINED_DIR} already exists"
else
  echo ">>> Downloading pretrained: ${PRETRAINED_ID}"
  python scripts/gemma9bit/save_model_param_folder.py --model "$PRETRAINED_ID" --output-dir "$PRETRAINED_DIR"
fi

# 2. Download finetuned models and symlink per-task pretrained → shared pretrained
for hf_id in "${FINETUNED_IDS[@]}"; do
  task="${hf_id##*_}"  # extract suffix after final underscore: instruction, math, coding, multilingual

  # Download finetuned
  ft_dir="${BASE}/${task}/finetuned"
  if [[ -d "$ft_dir" ]]; then
    echo ">>> Skipping ${task}/finetuned: already exists"
  else
    echo ">>> Downloading ${task}/finetuned: ${hf_id}"
    python scripts/gemma9bit/save_model_param_folder.py --model "$hf_id" --output-dir "$ft_dir"
  fi

  # Symlink per-task pretrained → shared pretrained (always refresh to handle re-downloads)
  pre_link="${BASE}/${task}/pretrained"
  rm -f "$pre_link"
  ln -s "$(realpath "$PRETRAINED_DIR")" "$pre_link"
  echo ">>> Symlinked ${task}/pretrained → $(realpath "$PRETRAINED_DIR")"
done
