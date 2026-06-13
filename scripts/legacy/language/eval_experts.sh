#!/bin/bash
#SBATCH --job-name=eval_lang_experts
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err

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

# Array dispatch: one task per (MODEL, FT_MODE).
#   ft=none      → writes artifacts/results/<model>-zeroshot/metrics.json
#   ft=standard  → writes artifacts/results/<model>-experts/metrics.json
#   ft=lora      → writes artifacts/results/<model>-experts/lora_metrics.json
MODELS=(t5-base t5-large)
FT_MODES=(none standard lora)

TID=$SLURM_ARRAY_TASK_ID
model_idx=$(( TID / ${#FT_MODES[@]} ))
ft_idx=$(( TID % ${#FT_MODES[@]} ))

MODEL=${MODELS[$model_idx]}
FT_MODE=${FT_MODES[$ft_idx]}

echo "[BASH] array task $TID → model=$MODEL ft=$FT_MODE"
python scripts/language/eval_experts.py \
  --model="$MODEL" \
  --finetuning-mode="$FT_MODE"
