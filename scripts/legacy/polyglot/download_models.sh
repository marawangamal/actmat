#!/bin/bash
# Download the Polyglot-Teachers OLMo3-7B student models and organize into the
# same param-folder checkpoint layout the OLMo pipeline uses:
#
#   artifacts/checkpoints/Olmo-3-7b-polyglot/
#     pretrained/          (param folder — shared base: allenai/Olmo-3-1025-7B)
#     ar/
#       pretrained/        (symlink -> ../pretrained)
#       finetuned/         (param folder — Polyglot-OLMo3-7B-SFT-ar)
#     cs/ de/ es/          (same structure)
#
# These are full-finetunes of allenai/Olmo-3-1025-7B on per-language synthetic
# SFT data (paper: "Polyglot Teachers", arXiv 2604.11290). Sharing one base
# makes them mergeable via task arithmetic.
#
# Usage:
#   bash scripts/polyglot/download_models.sh
set -euo pipefail

PRETRAINED_ID="allenai/Olmo-3-1025-7B"
# lang -> finetuned HF id
LANGS=(ar cs de es)
FT_PREFIX="ljvmiranda921/Polyglot-OLMo3-7B-SFT"

BASE="artifacts/checkpoints/Olmo-3-7b-polyglot"
PRETRAINED_DIR="${BASE}/pretrained"

# 1. Download shared pretrained base
if [[ -d "$PRETRAINED_DIR" ]]; then
  echo ">>> Skipping pretrained: ${PRETRAINED_DIR} already exists"
else
  echo ">>> Downloading pretrained: ${PRETRAINED_ID}"
  python scripts/olmo/save_model_param_folder.py --model "$PRETRAINED_ID" --output-dir "$PRETRAINED_DIR"
fi

# 2. Download per-language finetuned experts + symlink pretrained
for lang in "${LANGS[@]}"; do
  hf_id="${FT_PREFIX}-${lang}"
  ft_dir="${BASE}/${lang}/finetuned"
  if [[ -d "$ft_dir" ]]; then
    echo ">>> Skipping ${lang}/finetuned: already exists"
  else
    echo ">>> Downloading ${lang}/finetuned: ${hf_id}"
    python scripts/olmo/save_model_param_folder.py --model "$hf_id" --output-dir "$ft_dir"
  fi

  pre_link="${BASE}/${lang}/pretrained"
  rm -f "$pre_link"
  ln -s "$(realpath "$PRETRAINED_DIR")" "$pre_link"
  echo ">>> Symlinked ${lang}/pretrained -> $(realpath "$PRETRAINED_DIR")"
done

echo ">>> Done. Checkpoints under ${BASE}"
