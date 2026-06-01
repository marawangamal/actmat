#!/bin/bash
#SBATCH --job-name=download_gemma9bit
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=artifacts/logs/%x_%j.out
#SBATCH --error=artifacts/logs/%x_%j.err
# One-shot download wrapper. The 5 param folders (1 base + 4 experts, each
# ~18GB for gemma-2-9b-it) take ~30-60 min to fetch + write.
set -euo pipefail
mkdir -p artifacts/logs

source "$SCRATCH/actmat/.venv-gemma/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export HF_HOME="$SCRATCH/huggingface"

bash scripts/gemma9bit/download_models.sh
