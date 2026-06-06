#!/bin/bash
#SBATCH --job-name=mags_combined
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=artifacts/logs/%x_%j.out
#SBATCH --error=artifacts/logs/%x_%j.err
# Walk every (model, dataset) task-vector dir under artifacts/checkpoints,
# compute the global Frobenius norm across all 2-D layer deltas, and append
# one row per (model, dataset) to artifacts/analysis/magnitudes/magnitudes_combined.csv.
#
# Usage:
#   sbatch scripts/mags_combined.sh
set -euo pipefail
mkdir -p artifacts/logs artifacts/csvs

source "$SCRATCH/actmat/.venv-vl/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export HF_HOME="$SCRATCH/huggingface"

python scripts/mags_combined.py
