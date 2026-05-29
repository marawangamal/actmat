#!/bin/bash
#SBATCH --job-name=regmean_lora_cov
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err

# Re-run LoRA RegMean with covariance collected in LoRA mode.
# Prior LoRA RegMean results consumed FFT-derived covariance.pt via the
# unprefixed discovery fallback (no lora_covariance.pt existed). With the
# prefix-aware writer fix in covariance.py, this collects lora_covariance.pt
# from the LoRA-finetuned encoder and re-evaluates RegMean against it.

set -euo pipefail
mkdir -p artifacts/logs

# 0. Setup environment
source "$SCRATCH/actmat/.venv-vl/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export HF_HOME=$SCRATCH/huggingface
export SSL_CERT_DIR=/etc/ssl/certs

DATA_DIR="data/vision"

if [ ! -d "$SLURM_TMPDIR/data" ]; then
  cp downloads/data.tar.gz "$SLURM_TMPDIR/"
  tar -xzf "$SLURM_TMPDIR/data.tar.gz" -C "$SLURM_TMPDIR/"
fi
ln -sfn "$SLURM_TMPDIR/data" data

FT_MODE=lora
MERGE_MODE=d   # matches the original ViT-*-regmean/lora_metrics.json runs

# Array dispatch: one model per task.
MODELS=(ViT-B-16 ViT-B-32 ViT-L-14)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

# 1. Collect LoRA-mode covariance -> writes lora_covariance.pt next to each
#    finetuned checkpoint (nb=10, bs=32, train split, split-MHA: same recipe
#    as the original FFT covariance, only the FT mode differs).
echo "[BASH] covariance.py | model: $MODEL | ft mode: $FT_MODE"
python scripts/vision/covariance.py \
  --model="$MODEL" \
  --finetuning-mode="$FT_MODE" \
  --data-location="$DATA_DIR" \
  --cov-split=train \
  --cov-num-batches=10 \
  --cov-batch-size=32 \
  --mha=split

# 2. Re-evaluate LoRA RegMean against the freshly collected lora_covariance.pt.
echo "[BASH] eval_task_addition.py | model: $MODEL | ft mode: $FT_MODE | regmean | mode: $MERGE_MODE"
python scripts/vision/eval_task_addition.py \
  --model="$MODEL" \
  --finetuning-mode="$FT_MODE" \
  --data-location="$DATA_DIR" \
  --merge-func=regmean \
  --merge-mode="$MERGE_MODE" \
  --mha=split \
  --overwrite
