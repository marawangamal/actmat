#!/bin/bash
#SBATCH --job-name=eval_task_addition_sgd
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-4
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err

set -euo pipefail
mkdir -p artifacts/logs

# 0. Setup environment
source ".venv-vl/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

DATA_DIR="data/vision"

# 1. Stage datasets to $SLURM_TMPDIR (mirrors finetune.sh)
if [ ! -d "$SLURM_TMPDIR/data" ]; then
  cp downloads/data.tar.gz "$SLURM_TMPDIR/"
  tar -xzf "$SLURM_TMPDIR/data.tar.gz" -C "$SLURM_TMPDIR/"
fi
ln -sfn "$SLURM_TMPDIR/data" data

# 2. Merge the SGD-trained experts (artifacts/checkpoints-sgd, lr 1e-4) with
#    tsv / actmat / mean / isoc / wudi. All are data-free (no covariance step).
#    Results go to a separate dir so the AdamW merge results aren't clobbered.
#    Already-computed methods skip instantly (eval_task_addition.py skips when
#    the results file exists and --overwrite is not passed).
MODEL="ViT-B-16"
FT_MODE="standard"
MERGE_MODE="d"
CKPT_ROOT="artifacts/checkpoints-sgd"
RESULTS_DIR="artifacts/results-sgd"

METHODS=(tsv actmat mean isoc wudi)
method="${METHODS[$SLURM_ARRAY_TASK_ID]}"

echo "[BASH] Running eval_task_addition.py | model: $MODEL | ft mode: $FT_MODE | method: $method | mode: $MERGE_MODE | ckpt: $CKPT_ROOT"
python scripts/vision/eval_task_addition.py \
  --model="$MODEL" \
  --finetuning-mode="$FT_MODE" \
  --save="$CKPT_ROOT" \
  --data-location="$DATA_DIR" \
  --results-dir="$RESULTS_DIR" \
  --merge-func="$method" \
  --merge-mode="$MERGE_MODE" \
  --mha=split
