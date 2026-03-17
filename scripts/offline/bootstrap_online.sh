#!/usr/bin/env bash
set -euo pipefail

# Build an offline transfer bundle on an internet-connected server.
#
# Usage:
#   bash scripts/offline/bootstrap_online.sh [REPO_DIR] [BUNDLE_DIR]
#
# Example:
#   bash scripts/offline/bootstrap_online.sh \
#     /home/dtam/scratch/eigcov \
#     /home/dtam/scratch/eigcov/offline_bundle
#
# After this finishes, copy the bundle and repo to offline server:
#   rsync -aH --info=progress2 "$BUNDLE_DIR/" \
#     offline:/home/dtam/links/dtam/eigcov/offline_bundle/
#   rsync -aH --info=progress2 "$REPO_DIR/" \
#     offline:/home/dtam/links/dtam/eigcov/

REPO_DIR="${1:-/home/dtam/scratch/eigcov}"
BUNDLE_DIR="${2:-/home/dtam/scratch/eigcov/offline_bundle}"

mkdir -p "$BUNDLE_DIR"/{wheelhouse,hf_home,meta}
cd "$REPO_DIR"

# Some cluster Python installs do not ship pip by default.
if ! python3 -m pip --version >/dev/null 2>&1; then
  echo "pip not found for python3; bootstrapping with ensurepip..."
  python3 -m ensurepip --upgrade
fi

python3 -m pip install -U pip "huggingface_hub[cli]"

# 1) Download wheel files for offline pip install.
python -m pip install -U pip setuptools wheel
python -m pip download --only-binary=:all: "pyyaml==6.0.2" -d "$BUNDLE_DIR/wheelhouse"
sed -e 's/^pyyaml==6.0$/pyyaml==6.0.2/' \
    -e 's/^regex==2026\\.2\\.28$/regex==2026.1.15+computecanada/' \
    requirements.txt > /tmp/requirements.offline.txt
python3 -m pip download -r /tmp/requirements.offline.txt -d "$BUNDLE_DIR/wheelhouse"

# Runtime deps often required by eval stack even if not fully pinned.
python3 -m pip download \
  transformers datasets accelerate evaluate tokenizers \
  sentencepiece peft safetensors fsspec huggingface_hub \
  -d "$BUNDLE_DIR/wheelhouse"

# 2) Populate Hugging Face cache with required models.
export HF_HOME="$BUNDLE_DIR/hf_home"
export HF_HUB_DISABLE_XET=1
export HF_DATASETS_CACHE="$HF_HOME/datasets"

if command -v hf >/dev/null 2>&1; then
  HF_DL_CMD=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_DL_CMD=(huggingface-cli download)
else
  HF_DL_CMD=(python3 -m huggingface_hub.commands.huggingface_cli download)
fi

MODELS=(
  # "meta-llama/Llama-3.1-8B"
  "pmahdavi/Llama-3.1-8B-math-reasoning"
  "pmahdavi/Llama-3.1-8B-coding"
  "pmahdavi/Llama-3.1-8B-coding-tulu3-ebs128-lr5e6-wsdcr0p4"
  "pmahdavi/Llama-3.1-8B-precise-if"
  "pmahdavi/Llama-3.1-8B-general"
  "pmahdavi/Llama-3.1-8B-knowledge-recall"
)

for m in "${MODELS[@]}"; do
  echo "Downloading model: $m"
  echo "HF_DL_CMD: ${HF_DL_CMD[*]}"
  "${HF_DL_CMD[@]}" "$m" --cache-dir "$HF_HOME"
done

# 3) Prime task-specific caches via olmes for Tulu-3-dev.
# Set RUN_OLMES_PRIME=0 to skip.
if [[ "${RUN_OLMES_PRIME:-1}" == "1" ]]; then
  if command -v olmes >/dev/null 2>&1; then
    olmes \
      --model pmahdavi/Llama-3.1-8B-math-reasoning \
      --task tulu_3_dev \
      --limit 1 \
      --output-dir /tmp/olmes_prime
  else
    echo "Warning: 'olmes' not in PATH, skipping Tulu-3-dev priming."
  fi
fi

python3 -V > "$BUNDLE_DIR/meta/python_version.txt"
uname -a > "$BUNDLE_DIR/meta/uname.txt"

echo "Offline bundle ready at: $BUNDLE_DIR"
