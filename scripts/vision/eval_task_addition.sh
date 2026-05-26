#!/bin/bash
#SBATCH --job-name=eval_vision_8
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-4
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err

set -euo pipefail
mkdir -p artifacts/logs

# 0. Setup environment
source "$SCRATCH/actmat/.venv-vl/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

DATA_DIR="data/vision"
OPENCLIP_DIR="$SCRATCH/openclip"

if [ ! -d "$SLURM_TMPDIR/data" ]; then
  cp downloads/data.tar.gz "$SLURM_TMPDIR/"
  tar -xzf "$SLURM_TMPDIR/data.tar.gz" -C "$SLURM_TMPDIR/"
fi
ln -sfn "$SLURM_TMPDIR/data" data

# ===== LoRA experiments on the original 8-dataset benchmark =====
MODELS=(ViT-B-16 ViT-B-32 ViT-L-14)
FT_MODE="lora"
MERGE_MODE=d
HPO=''

# Array dispatch: one task per method. Keep --array=0-N in sync with len(METHODS)-1.
METHODS=(mean isoc tsv actmat wudi)
method="${METHODS[$SLURM_ARRAY_TASK_ID]}"

for MODEL in "${MODELS[@]}"; do
  # 2a. Run covariance if needed (actmat consumes covariance.pt).
  if [ "$method" = "actmat" ]; then
    echo "[BASH] Running covariance.py | model: $MODEL | ft mode: $FT_MODE | method: $method"
    python scripts/vision/covariance.py \
      --model="$MODEL" \
      --finetuning-mode="$FT_MODE" \
      --mha=split
  fi

  # 2b. Evaluate task addition.
  echo "[BASH] Running eval_task_addition.py | model: $MODEL | ft mode: $FT_MODE | method: $method | mode: $MERGE_MODE"
  python scripts/vision/eval_task_addition.py \
    --model="$MODEL" \
    --finetuning-mode="$FT_MODE" \
    --data-location="$DATA_DIR" \
    --merge-func="$method" \
    --merge-mode="$MERGE_MODE" \
    --mha=split \
    ${HPO:+--hpo="$HPO"}
done
