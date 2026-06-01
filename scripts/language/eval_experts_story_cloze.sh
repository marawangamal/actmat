#!/bin/bash
#SBATCH --job-name=eval_exp_sc
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err
#SBATCH --array=0-3

# Regenerate the experts (FFT) and zeroshot result files for both models over
# all 7 tasks, now that story_cloze renders correctly.
#   ft=none     -> artifacts/results/<model>-zeroshot/metrics.json
#   ft=standard -> artifacts/results/<model>-experts/metrics.json
set -euo pipefail
mkdir -p artifacts/logs

source "$SCRATCH/actmat/.venv-vl/bin/activate"
export HF_HOME=$SCRATCH/huggingface
export NLTK_DATA=$SCRATCH/nltk_data
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs
ln -sfn artifacts/data data

MODELS=(t5-base t5-large)
FT_MODES=(none standard)
TID=$SLURM_ARRAY_TASK_ID
MODEL=${MODELS[$(( TID / 2 ))]}
FT_MODE=${FT_MODES[$(( TID % 2 ))]}

echo "[BASH] eval_experts | model=$MODEL ft=$FT_MODE"
python scripts/language/eval_experts.py --model="$MODEL" --finetuning-mode="$FT_MODE"
