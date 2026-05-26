#!/bin/bash
#SBATCH --job-name=eval_vision_drift
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

# ===== Trajectory merge eval over intermediate drift checkpoints =====
# Sweeps over training steps: at each step S, merge `checkpoint_S.pt` from
# each task's drift dir via {mean, actmat, tsv} and evaluate. Yields a curve
# of merged-model accuracy vs. training step per method.
MODEL="ViT-B-16"
FT_MODE="standard"
MERGE_MODE="d"
MHA="split"
CKPT_ROOT="artifacts/checkpoints-analysis-drift"
RESULTS_ROOT="artifacts/results-analysis-drift"
METHODS=(mean actmat tsv isoc wudi)
DATASETS=(Cars DTD EuroSAT GTSRB MNIST RESISC45 SUN397 SVHN)

# Array dispatch: one task per training step. Keep --array=0-N in sync with len(STEPS)-1.
# Note: step 2000 dropped — RESISC45 and SUN397 hit early-stopping at 1800 so
# `checkpoint_2000.pt` does not exist for those tasks. The "final" step
# (finetuned.pt) is available for all tasks and serves as the trajectory endpoint.
STEPS=(0 200 400 600 800 1000 1200 1400 1600 1800 final)
step="${STEPS[$SLURM_ARRAY_TASK_ID]}"

# regmean is the only method in this repo that reads covariance_path (merge_actmat
# computes its own c = dᵀd from the deltas). Per-step covariances
# (covariance_checkpoint_S.pt) are already produced by covariance-drifts.sh for
# S in {0,200,...,1800}; at step "final" we need a matching covariance.pt —
# compute it on demand if missing.
if [[ " ${METHODS[*]} " =~ " regmean " ]] && [ "$step" = "final" ]; then
  for DATASET in "${DATASETS[@]}"; do
    cov_file="$CKPT_ROOT/$MODEL/${DATASET}Val/covariance.pt"
    if [ ! -f "$cov_file" ]; then
      echo "[BASH] covariance.py (final) | $MODEL | $DATASET"
      python scripts/vision/covariance.py \
        --model="$MODEL" \
        --finetuning-mode="$FT_MODE" \
        --save="$CKPT_ROOT" \
        --eval-datasets="$DATASET" \
        --data-location="$DATA_DIR" \
        --cache-dir="$OPENCLIP_DIR" \
        --mha="$MHA" \
        --cov-split=train \
        --cov-num-batches=10 \
        --cov-batch-size=32 \
        --cov-type=sm \
        --cov-estimator=full
    fi
  done
fi

for MERGE_FUNC in "${METHODS[@]}"; do
  echo "[BASH] eval_task_addition.py | model: $MODEL | ft: $FT_MODE | method: $MERGE_FUNC | step: $step"
  python scripts/vision/eval_task_addition.py \
    --model="$MODEL" \
    --finetuning-mode="$FT_MODE" \
    --save="$CKPT_ROOT" \
    --data-location="$DATA_DIR" \
    --cache-dir="$OPENCLIP_DIR" \
    --merge-func="$MERGE_FUNC" \
    --merge-mode="$MERGE_MODE" \
    --mha="$MHA" \
    --checkpoint-step="$step" \
    --results-dir="$RESULTS_ROOT/step_$step"
done
