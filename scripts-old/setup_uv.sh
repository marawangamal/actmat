#!/bin/bash
set -e

rm -f pyproject.toml uv.lock
rm -rf .venv
echo "Removed .venv and pyproject.toml"

module load python/3.10
uv init --python 3.10 --no-readme

uv add torch torchvision numpy scipy tqdm Pillow \
  transformers datasets huggingface-hub peft evaluate trl \
  sentencepiece protobuf \
  open-clip-torch
