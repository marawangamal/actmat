#!/bin/bash
#SBATCH --job-name=rmloss-olmo
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j-rmloss-olmo.out

set -euo pipefail
cd /network/scratch/m/marawan.gamal/actmat
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export HF_HOME=$SCRATCH/huggingface
source .venv-olmo/bin/activate

python scripts/analysis/rm-loss-olmo.py
python scripts/analysis/rm-loss-olmo-plot.py
