#!/bin/bash
set -e

rm -f pyproject.toml uv.lock
uv python install 3.12
uv init --python 3.12 --no-readme
sed -i 's/requires-python = ">=3.12"/requires-python = ">=3.11,<3.13"/' pyproject.toml

# promptsource PyPI sdist is broken — install from git
cat >> pyproject.toml << 'EOF'

[tool.uv.sources]
promptsource = { git = "https://github.com/bigscience-workshop/promptsource.git" }
EOF

uv add torch torchvision numpy scipy tqdm Pillow \
  transformers datasets huggingface-hub safetensors "peft>=0.6.0" evaluate trl \
  sentencepiece protobuf promptsource \
  "open-clip-torch==2.0.2" timm ftfy regex \
  filelock fsspec packaging pyyaml wcwidth

uv run python -c "import torch; import transformers; import open_clip; print('OK')"
