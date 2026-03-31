#!/bin/bash
set -e

MODEL_DIR="${1:?Usage: bash scripts/vision/fix_checkpoints.sh checkpoints/ViT-B-16}"

for d in Cars DTD EuroSAT GTSRB MNIST RESISC45 SUN397 SVHN; do
  if [ -d "$MODEL_DIR/$d" ] && [ ! -d "$MODEL_DIR/${d}Val" ]; then
    mv "$MODEL_DIR/$d" "$MODEL_DIR/${d}Val"
  fi
  if [ -d "$MODEL_DIR/${d}Val" ] && [ ! -e "$MODEL_DIR/${d}Val/zeroshot.pt" ]; then
    ln -s ../zeroshot.pt "$MODEL_DIR/${d}Val/zeroshot.pt"
  fi
done

echo "Done. Checkpoint layout fixed in $MODEL_DIR"
