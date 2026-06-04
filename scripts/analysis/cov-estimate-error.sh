#!/bin/bash
#SBATCH --job-name=cov-est-err
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=artifacts/logs/%x_%j.out
#SBATCH --error=artifacts/logs/%x_%j.err

set -euo pipefail
cd /network/scratch/m/marawan.gamal/actmat
mkdir -p artifacts/logs
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export HF_HOME=$SCRATCH/huggingface
# .venv-vl handles both: it can unpickle the t5/ViT .pt models AND read OLMo's
# safetensors/mmap'd covariance (the OLMo path instantiates no model).
source .venv-vl/bin/activate

python scripts/analysis/cov-estimate-error.py
