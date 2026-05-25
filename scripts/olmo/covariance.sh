#!/bin/bash
#SBATCH --job-name=cov_olmo
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=artifacts/logs/%x_%j.out
#SBATCH --error=artifacts/logs/%x_%j.err
# Collect per-layer activation covariances for each Olmo-3-7B capability expert.
#
# Prerequisites:
#   - scripts/olmo/download_models.sh has populated
#     artifacts/checkpoints/${MODEL}/{Math,Code,IF}/finetuned (param-folder layout).
#
# Usage:
#   sbatch scripts/olmo/covariance.sh
set -euo pipefail
mkdir -p artifacts/logs

# 0. Setup environment
source "$SCRATCH/actmat/.venv-olmo/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export HF_HOME="$SCRATCH/huggingface"
export SSL_CERT_DIR=/etc/ssl/certs

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL="Olmo-3-7b"
COV_NUM_BATCHES=10
COV_BATCH_SIZE=8
COV_TYPE="sm"          # uncentered second moment (matches regmean/actmat)
COV_ESTIMATOR="full"

# ── Run ───────────────────────────────────────────────────────────────────────
python scripts/olmo/covariance.py \
  --capability all \
  --save "artifacts/checkpoints/${MODEL}" \
  --cov-num-batches "$COV_NUM_BATCHES" \
  --cov-batch-size "$COV_BATCH_SIZE" \
  --cov-type "$COV_TYPE" \
  --cov-estimator "$COV_ESTIMATOR" \
  --overwrite
