#!/bin/bash
#SBATCH --job-name=polyglot_dl
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=artifacts/logs/%x_%j.out
#SBATCH --error=artifacts/logs/%x_%j.err
# SLURM wrapper around download_models.sh. Loading a 7B model on CPU to export
# its param folder needs ~14-28G RAM, so give it a dedicated allocation (the
# login/interactive node OOM-kills it). Skips any expert already downloaded.
#
# Usage: sbatch scripts/polyglot/download.sh
set -euo pipefail
mkdir -p artifacts/logs
source "$SCRATCH/actmat/.venv-olmo/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export HF_HOME=$SCRATCH/huggingface
bash scripts/polyglot/download_models.sh
