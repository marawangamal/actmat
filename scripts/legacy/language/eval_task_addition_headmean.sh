#!/bin/bash
#SBATCH --job-name=eval_lang_headmean
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err
#SBATCH --array=0-3

# Head-mean variant of eval_task_addition.sh: identical merge, except the FF
# down-projection layers (DenseReluDense.wo) are forced to a plain MEAN merge
# (--mean-keys) instead of the chosen method. These wide-input layers are where
# the unregularized ACTMat pseudo-inverse overshoots and dominates the summed
# RegMean loss (see artifacts/notes/regmean-loss-vs-accuracy.md). Writes to its
# OWN group (group-headmean) so it never clobbers the full-method group-main
# results; experts are symlinked group-headmean/experts -> group-main/experts.
# Confirm the override fired: grep "[mean_keys]" the merge logs.

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

MODELS=(t5-base t5-large)
METHODS=(actmat regmean)
FT_MODE=standard
MERGE_MODE=d
MEAN_KEYS="DenseReluDense.wo"

# Array dispatch: one task per (MODEL, METHOD).  len=2*2=4  → array 0-3
TID=$SLURM_ARRAY_TASK_ID
model_idx=$(( TID / ${#METHODS[@]} ))
method_idx=$(( TID % ${#METHODS[@]} ))
MODEL=${MODELS[$model_idx]}
METHOD=${METHODS[$method_idx]}

echo "[BASH] array task $TID → model=$MODEL method=$METHOD group=headmean mean_keys=$MEAN_KEYS"

python scripts/language/eval_task_addition.py \
  --model="$MODEL" \
  --finetuning-mode="$FT_MODE" \
  --merge-mode="$MERGE_MODE" \
  --merge-func="$METHOD" \
  --mean-keys "$MEAN_KEYS" \
  --group=headmean
