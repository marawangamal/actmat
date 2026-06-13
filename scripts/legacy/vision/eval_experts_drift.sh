#!/bin/bash
#SBATCH --job-name=eval_experts_drift
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-10
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err

set -euo pipefail
mkdir -p artifacts/logs

# 0. Setup environment
source ".venv-vl/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs
export HF_HOME=$SCRATCH/huggingface

DATA_DIR="data/vision"
OPENCLIP_DIR="$SCRATCH/openclip"

if [ ! -d "$SLURM_TMPDIR/data" ]; then
  cp downloads/data.tar.gz "$SLURM_TMPDIR/"
  tar -xzf "$SLURM_TMPDIR/data.tar.gz" -C "$SLURM_TMPDIR/"
fi
ln -sfn "$SLURM_TMPDIR/data" data

# ===== Per-step expert (single-task) eval over drift checkpoints =====
# For each training step S, load checkpoint_S.pt for each task and evaluate
# that expert on its own task's val + test splits. Yields the per-task
# upper-bound curve (no merging) to compare against the merged trajectory
# from eval_task_addition_drift.sh.
MODEL="ViT-B-16"
FT_MODE="standard"
CKPT_ROOT="artifacts/checkpoints-analysis-drift"
RESULTS_ROOT="artifacts/results-analysis-drift"

# Array dispatch: one task per training step. Step 2000 omitted (RESISC45 +
# SUN397 early-stopped at 1800). "final" maps to finetuned.pt for every task.
STEPS=(0 200 400 600 800 1000 1200 1400 1600 1800 final)
step="${STEPS[$SLURM_ARRAY_TASK_ID]}"

echo "[BASH] eval_experts.py | model: $MODEL | ft: $FT_MODE | step: $step"
python scripts/vision/eval_experts.py \
  --model="$MODEL" \
  --finetuning-mode="$FT_MODE" \
  --save="$CKPT_ROOT" \
  --data-location="$DATA_DIR" \
  --cache-dir="$OPENCLIP_DIR" \
  --checkpoint-step="$step" \
  --results-dir="$RESULTS_ROOT/step_$step"
