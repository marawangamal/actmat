#!/bin/bash
#SBATCH --job-name=eval_lang_models
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err
#SBATCH --array=0-39

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

# ===== Default experiments (no hyperparameter tuning) =====
MODELS=(t5-base t5-large)
METHODS=(ace wudi ties dare sum mean tsv isoc regmean actmat)
FT_MODES=(standard lora)
MERGE_MODE=d
HPO=""

# Array dispatch: one task per (FT_MODE, MODEL, METHOD). Same order as the
# original nested loop (FT_MODE outer, MODEL middle, METHOD inner).
#   len(METHODS)=10, len(MODELS)=2, len(FT_MODES)=2  → total 40 tasks
TID=$SLURM_ARRAY_TASK_ID
ft_idx=$(( TID / 20 ))
model_idx=$(( (TID % 20) / 10 ))
method_idx=$(( TID % 10 ))

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
