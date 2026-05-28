#!/bin/bash
#SBATCH --job-name=t5base_cond
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=artifacts/logs/%x_%j.out
#SBATCH --error=artifacts/logs/%x_%j.err
# Per-layer condition numbers for t5-base RegMean (covariance_old.pt) vs
# ACTMat d.T d estimators, per task and summed. CPU-only.
set -euo pipefail
mkdir -p artifacts/logs

source "$SCRATCH/actmat/.venv-vl/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

python -u scripts/analysis/condition_numbers_t5base.py
