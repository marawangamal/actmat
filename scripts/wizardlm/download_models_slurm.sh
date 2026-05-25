#!/bin/bash
#SBATCH --job-name=dl_wizardlm
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=artifacts/logs/%x_%j.out
#SBATCH --error=artifacts/logs/%x_%j.err
# Run scripts/wizardlm/download_models.sh under a SLURM job whose cgroup
# allows enough memory for torch.load on 13B .bin shards. Login-node /
# `mila-code` interactive cgroups (≤16G) get the converter OOM-killed.
set -euo pipefail
mkdir -p artifacts/logs

source "$SCRATCH/actmat/.venv-olmo/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export HF_HOME="$SCRATCH/huggingface"

bash scripts/wizardlm/download_models.sh
