#!/usr/bin/env bash
set -euo pipefail

# Configure and install on an offline server using a prebuilt transfer bundle.
#
# Usage:
#   bash scripts/offline/bootstrap_offline.sh [REPO_DIR] [BUNDLE_DIR]
#
# Example:
#   bash scripts/offline/bootstrap_offline.sh \
#      ~/links/dtam/eigcov/eigcov \
#      ~/links/dtam/eigcov/offline_bundle

REPO_DIR="${1:-/home/dtam/links/dtam/eigcov}"
BUNDLE_DIR="${2:-/home/dtam/links/dtam/eigcov/offline_bundle}"

if [[ ! -d "$BUNDLE_DIR/wheelhouse" ]]; then
  echo "Missing wheelhouse: $BUNDLE_DIR/wheelhouse"
  exit 1
fi

if [[ ! -d "$BUNDLE_DIR/hf_home" ]]; then
  echo "Missing HF cache dir: $BUNDLE_DIR/hf_home"
  exit 1
fi

cd "$REPO_DIR"

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install -U pip
sed -e 's/^pyyaml==6.0$/pyyaml==6.0.2/' \
    -e 's/^regex==2026\\.2\\.28$/regex==2026.1.15+computecanada/' \
    requirements.txt > /tmp/requirements.offline.txt
python -m pip install --no-index --find-links="$BUNDLE_DIR/wheelhouse" -r /tmp/requirements.offline.txt
python -m pip install --no-index --find-links="$BUNDLE_DIR/wheelhouse" \
  transformers datasets accelerate evaluate tokenizers \
  sentencepiece peft safetensors fsspec huggingface_hub

# Offline runtime config.
export HF_HOME="$BUNDLE_DIR/hf_home"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_XET=1

cat <<EOM
Offline bootstrap complete.

Activate env each session:
  source "$REPO_DIR/.venv/bin/activate"

Export offline env vars each session:
  export HF_HOME="$BUNDLE_DIR/hf_home"
  export HF_HUB_CACHE="\$HF_HOME/hub"
  export HF_DATASETS_CACHE="\$HF_HOME/datasets"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  export HF_HUB_DISABLE_XET=1
EOM
