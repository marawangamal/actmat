#!/bin/bash
#SBATCH --job-name=eval_lang_models
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err
#SBATCH --array=0

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

# ===== One-off: regmean on t5-base (FFT) with test-set covariance =====
MODELS=(t5-base)
METHODS=(regmean)
FT_MODES=(standard)
MERGE_MODE=d
HPO=""

# Array dispatch: one task per (FT_MODE, MODEL, METHOD).
#   len(METHODS)=1, len(MODELS)=1, len(FT_MODES)=1  → total 1 task
TID=$SLURM_ARRAY_TASK_ID
ft_idx=$(( TID / 1 ))
model_idx=$(( (TID % 1) / 1 ))
method_idx=$(( TID % 1 ))

FT_MODE=${FT_MODES[$ft_idx]}
MODEL=${MODELS[$model_idx]}
method=${METHODS[$method_idx]}

echo "[BASH] array task $TID → ft=$FT_MODE model=$MODEL method=$method"

# 1. Run covariance if needed.
if [ "$method" = "regmean" ]; then
  echo "[BASH] Running covariance.py | model: $MODEL | ft mode: $FT_MODE | method: $method"
  python scripts/language/covariance.py \
    --model="$MODEL" \
    --finetuning-mode="$FT_MODE"
fi

# 2. Run fisher collection if needed.
if [ "$method" = "fisher" ]; then
  echo "[BASH] Running fisher.py | model: $MODEL | ft mode: $FT_MODE | method: $method"
  python scripts/language/fisher.py \
    --model="$MODEL" \
    --finetuning-mode="$FT_MODE"
fi

# 3. Evaluate task addition.
echo "[BASH] Running eval_task_addition.py | model: $MODEL | ft mode: $FT_MODE | method: $method | merge mode: $MERGE_MODE"
python scripts/language/eval_task_addition.py \
  --model="$MODEL" \
  --finetuning-mode="$FT_MODE" \
  --merge-mode="$MERGE_MODE" \
  --merge-func="$method" \
  ${HPO:+--hpo="$HPO"}
