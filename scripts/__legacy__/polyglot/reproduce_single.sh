#!/bin/bash
#SBATCH --job-name=polyglot_repro
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=artifacts/logs/%x_%j.out
#SBATCH --error=artifacts/logs/%x_%j.err
# Step 2: reproduce the paper's numbers for ONE released student model by
# running its exact Lighteval setup straight from the HF Hub (no merge).
# Defaults to the German expert; override with REPRO_MODEL / REPRO_NAME.
#
# Usage:
#   sbatch scripts/polyglot/reproduce_single.sh
#   REPRO_MODEL=ljvmiranda921/Polyglot-OLMo3-7B-SFT-es REPRO_NAME=sft-es \
#     sbatch scripts/polyglot/reproduce_single.sh
set -euo pipefail
mkdir -p artifacts/logs

source "$SCRATCH/actmat/.venv-polyglot/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export HF_HOME=$SCRATCH/huggingface
export SSL_CERT_DIR=/etc/ssl/certs

source scripts/polyglot/lib_lighteval.sh

REPRO_MODEL="${REPRO_MODEL:-ljvmiranda921/Polyglot-OLMo3-7B-SFT-de}"
REPRO_NAME="${REPRO_NAME:-sft-de}"
OUT_DIR="artifacts/results-polyglot/repro-${REPRO_NAME}"

echo ">>> Reproducing ${REPRO_MODEL} -> ${OUT_DIR}"
run_lighteval "$REPRO_MODEL" "$OUT_DIR"
echo ">>> Done. Inspect ${OUT_DIR} and compare against the paper / HF details_* datasets."
