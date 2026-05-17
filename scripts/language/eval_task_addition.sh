#!/bin/bash
#SBATCH --job-name=eval_lang_models
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
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

# ===== Default experiments (no hyperparameter tuning) =====
MODELS=(t5-base t5-large)
METHODS=(sum mean tsv isoc regmean actmat)
FT_MODES=(standard lora)
MERGE_MODE=d
HPO=""

for FT_MODE in "${FT_MODES[@]}"; do
for MODEL in "${MODELS[@]}"; do
  for method in "${METHODS[@]}"; do

    # Run covariance script if needed
    if [ "$method" = "regmean" ]; then
      echo "[BASH] Running covariance.py | model: $MODEL | ft mode: $FT_MODE | method: $method"
      python scripts/language/covariance.py \
        --model="$MODEL" \
        --finetuning-mode="$FT_MODE" 
    fi

    # Run fisher collection if needed
    if [ "$method" = "fisher" ]; then
      echo "[BASH] Running fisher.py | model: $MODEL | ft mode: $FT_MODE | method: $method"
      python scripts/language/fisher.py \
        --model="$MODEL" \
        --finetuning-mode="$FT_MODE" 
    fi

    # Evaluate task addition
    echo "[BASH] Running eval_task_addition.py | model: $MODEL | ft mode: $FT_MODE | method: $method | merge mode: $MERGE_MODE"
    python scripts/language/eval_task_addition.py \
      --model="$MODEL" \
      --finetuning-mode="$FT_MODE" \
      --merge-mode="$MERGE_MODE" \
      --merge-func="$method" \
      ${HPO:+--hpo="$HPO"}

  done
done
done
