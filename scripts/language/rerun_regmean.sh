#!/bin/bash
#SBATCH --job-name=rerun_regmean_t5
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=artifacts/logs/%x_%j.out
#SBATCH --error=artifacts/logs/%x_%j.err

set -euo pipefail
mkdir -p artifacts/logs

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

MODELS=(t5-base t5-large)
FT_MODES=(standard lora)
MERGE_MODE=d
# Using default --cov-num-batches (10) so this is an apples-to-apples baseline
# against the pre-prefix-fix runs; we'll re-test with 10x later.

for FT_MODE in "${FT_MODES[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    echo "[BASH] covariance.py | model=$MODEL ft=$FT_MODE (default batches)"
    python scripts/language/covariance.py \
      --model="$MODEL" \
      --finetuning-mode="$FT_MODE" \
      --overwrite

    echo "[BASH] eval_task_addition.py regmean | model=$MODEL ft=$FT_MODE"
    python scripts/language/eval_task_addition.py \
      --model="$MODEL" \
      --finetuning-mode="$FT_MODE" \
      --merge-mode="$MERGE_MODE" \
      --merge-func=regmean
  done
done
