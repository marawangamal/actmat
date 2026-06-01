#!/bin/bash
#SBATCH --job-name=mags
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=artifacts/logs/%x_%j.out
#SBATCH --error=artifacts/logs/%x_%j.err
# Walk every (model, dataset) task-vector dir under artifacts/checkpoints,
# compute the Frobenius norm of each 2-D layer delta, and append to
# artifacts/csvs/magnitudes.csv. CPU-only; vision/language/OLMo all share
# the .venv-vl venv here since ParamFolderTaskVector only needs torch + safetensors.
#
# Usage:
#   sbatch scripts/mags.sh
set -euo pipefail
mkdir -p artifacts/logs artifacts/csvs

source "$SCRATCH/actmat/.venv-vl/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export HF_HOME="$SCRATCH/huggingface"

python scripts/mags.py
