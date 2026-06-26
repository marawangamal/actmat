#!/bin/bash
#SBATCH --job-name=ft_storycloze
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err
#SBATCH --array=0-1

# Re-finetune ONLY the story_cloze FFT expert for t5-base and t5-large.
# finetune.py loops over all 7 T5_DATASETS but skips any dataset whose
# checkpoint already exists; since story_cloze/ was moved to story_cloze_faulty,
# only story_cloze is (re)trained. Uses the corrected StoryClozeReader.
set -euo pipefail
mkdir -p artifacts/logs

source "$SCRATCH/actmat/.venv-vl/bin/activate"
export HF_HOME=$SCRATCH/huggingface
export NLTK_DATA=$SCRATCH/nltk_data
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

# story_cloze is the only task that reads a local CSV. Point `data` at the
# in-repo copy (verified correct columns) instead of unpacking the 33GB tarball.
ln -sfn artifacts/data data

MODELS=(t5-base t5-large)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "[BASH] Re-finetuning story_cloze (FFT) | model=$MODEL"
python scripts/language/finetune.py \
  --model="$MODEL" \
  --finetuning-mode=standard
