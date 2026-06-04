#!/bin/bash
#SBATCH --job-name=rmloss-gen
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j-rmloss-gen.out

set -euo pipefail
cd /network/scratch/m/marawan.gamal/actmat
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export HF_HOME=$SCRATCH/huggingface
# .venv-vl handles both: it can unpickle the t5 .pt models AND read OLMo's
# safetensors/mmap'd covariance (the OLMo path instantiates no model).
source .venv-vl/bin/activate

python scripts/analysis/rm-loss-general.py
