#!/bin/bash
#SBATCH --job-name=mtl_sc
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err
#SBATCH --array=0-1

# Retrain the multitask (FFT) model on the joint mixture with the corrected
# story_cloze reader, then eval. The old multitask checkpoint was moved to
# multitask_faulty so the skip-if-exists guard does not fire.
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

echo "[BASH] finetune_mtl | model=$MODEL"
python scripts/language/finetune_mtl.py \
  --model="$MODEL" \
  --finetuning-mode=standard \
  --checkpoint-every=1000 \
  --patience=10

echo "[BASH] eval_multitask | model=$MODEL"
python scripts/language/eval_multitask.py --model="$MODEL" --finetuning-mode=standard
