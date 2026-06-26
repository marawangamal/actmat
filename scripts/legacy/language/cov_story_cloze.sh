#!/bin/bash
#SBATCH --job-name=cov_storycloze
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err
#SBATCH --array=0-1

# Train-set covariance for the freshly re-finetuned story_cloze FFT expert.
# covariance.py loops all 7 datasets but skips any with an existing
# {prefix}covariance.pt; only story_cloze gets computed. Then rename + symlink
# to match the sibling experts (covariance.pt -> covariance-train-nb1000-bs32.pt).
set -euo pipefail
mkdir -p artifacts/logs

source "$SCRATCH/actmat/.venv-vl/bin/activate"
export HF_HOME=$SCRATCH/huggingface
export NLTK_DATA=$SCRATCH/nltk_data
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs
ln -sfn artifacts/data data

MODELS=(t5-base t5-large)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
NB=1000
BS=32
TRAIN_NAME="covariance-train-nb${NB}-bs${BS}.pt"

python scripts/language/covariance.py \
  --model="$MODEL" \
  --finetuning-mode=standard \
  --cov-split=train \
  --cov-num-batches="$NB" \
  --cov-batch-size="$BS"

dir="artifacts/checkpoints/$MODEL/story_cloze"
if [[ -f "$dir/covariance.pt" && ! -L "$dir/covariance.pt" ]]; then
  mv -v "$dir/covariance.pt" "$dir/$TRAIN_NAME"
  ln -sfn "$TRAIN_NAME" "$dir/covariance.pt"
  echo "  symlink: $dir/covariance.pt -> $TRAIN_NAME"
fi
