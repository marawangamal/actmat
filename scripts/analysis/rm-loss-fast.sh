#!/bin/bash
#SBATCH --job-name=rmloss-vl
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j-rmloss-vl.out

set -euo pipefail
cd /network/scratch/m/marawan.gamal/actmat
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export HF_HOME=$SCRATCH/huggingface
source .venv-vl/bin/activate

python scripts/analysis/rm-loss-fast.py
python scripts/analysis/rm-loss-fast-plot.py
