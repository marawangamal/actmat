#!/bin/bash
#SBATCH --job-name=ft_mtl_lang
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=artifacts/logs/%x_%j.out
#SBATCH --error=artifacts/logs/%x_%j.err

set -euo pipefail
mkdir -p artifacts/logs

# 0. Setup environment
source "$SCRATCH/actmat/.venv-vl/bin/activate"
export HF_HOME=$SCRATCH/huggingface
export NLTK_DATA=$SCRATCH/nltk_data
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

if [ ! -d "$SLURM_TMPDIR/data" ]; then
  cp downloads/data.tar.gz "$SLURM_TMPDIR/"
  tar -xzf "$SLURM_TMPDIR/data.tar.gz" -C "$SLURM_TMPDIR/"
fi
ln -sfn "$SLURM_TMPDIR/data" data

MODEL=${MODEL:-t5-base}
FT_MODE=${FT_MODE:-standard}

echo "[BASH] Running finetune_mtl.py | model: $MODEL | ft mode: $FT_MODE"
python scripts/language/finetune_mtl.py \
  --model="$MODEL" \
  --finetuning-mode="$FT_MODE" \
  --checkpoint-every=1000 \
  --patience=10
