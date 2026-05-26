#!/bin/bash
#SBATCH --job-name=eval_lang_actmat_p05
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err
#SBATCH --array=0-3

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
FT_MODES=(standard lora)
METHOD=actmat_p05
MERGE_MODE=d

# 4 tasks: (FT_MODE outer, MODEL inner) — len(MODELS)=2, len(FT_MODES)=2
TID=$SLURM_ARRAY_TASK_ID
ft_idx=$(( TID / 2 ))
model_idx=$(( TID % 2 ))

FT_MODE=${FT_MODES[$ft_idx]}
MODEL=${MODELS[$model_idx]}

echo "[BASH] array task $TID → ft=$FT_MODE model=$MODEL method=$METHOD"

echo "[BASH] Running eval_task_addition.py | model: $MODEL | ft mode: $FT_MODE | method: $METHOD | merge mode: $MERGE_MODE"
python scripts/language/eval_task_addition.py \
  --model="$MODEL" \
  --finetuning-mode="$FT_MODE" \
  --merge-mode="$MERGE_MODE" \
  --merge-func="$METHOD"
