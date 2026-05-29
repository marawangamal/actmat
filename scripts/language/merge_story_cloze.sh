#!/bin/bash
#SBATCH --job-name=merge_sc
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err
#SBATCH --array=0-11

# Re-run the merges (FFT, merge-mode d) over actmat + baselines for both models.
# Each consumes the freshly re-finetuned story_cloze expert (and its covariance
# for actmat/regmean, produced by cov_story_cloze.sh). Writes a fresh
# artifacts/results/<model>-<method>/metrics.json per (model, method).
set -euo pipefail
mkdir -p artifacts/logs

source "$SCRATCH/actmat/.venv-vl/bin/activate"
export HF_HOME=$SCRATCH/huggingface
export NLTK_DATA=$SCRATCH/nltk_data
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs
ln -sfn artifacts/data data

MODELS=(t5-base t5-large)
METHODS=(actmat regmean tsv ties isoc mean)
TID=$SLURM_ARRAY_TASK_ID
MODEL=${MODELS[$(( TID / ${#METHODS[@]} ))]}
method=${METHODS[$(( TID % ${#METHODS[@]} ))]}

echo "[BASH] merge | model=$MODEL method=$method"
python scripts/language/eval_task_addition.py \
  --model="$MODEL" \
  --finetuning-mode=standard \
  --merge-mode=d \
  --merge-func="$method"
