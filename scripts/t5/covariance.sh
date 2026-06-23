#!/bin/bash
#SBATCH --job-name=cov_t5
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-6
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err

set -euo pipefail
mkdir -p artifacts/logs

source ".venv-vl/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

NUM_TASKS="${NUM_TASKS:-7}"
FT_MODE="${FT_MODE:-fft}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-128}"
DATA_DIR="data"
CACHE_DIR="$SCRATCH/huggingface"
CKPT_ROOT="artifacts/checkpoints"

DATASETS_7=(qasc wiki_qa quartz paws story_cloze winogrande wsc)
case "$NUM_TASKS" in
  7) DATASETS=("${DATASETS_7[@]}") ;;
  *) echo "Unsupported NUM_TASKS=$NUM_TASKS"; exit 1 ;;
esac

DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
MODEL="${MODEL:-t5-base}"

OVERWRITE_ARGS=()
if [ "${OVERWRITE:-0}" = "1" ]; then
  OVERWRITE_ARGS=(--overwrite)
fi

# Mirror finetune.sh: non-default seq-lens live in their own group dir.
GROUP="$FT_MODE-$NUM_TASKS"
if [ "$MAX_SEQ_LEN" != "128" ]; then
  GROUP="$GROUP-seqlen$MAX_SEQ_LEN"
fi

EXPERT_DIR="$CKPT_ROOT/$MODEL/group-$GROUP/experts/$DATASET"
python scripts/t5/covariance.py \
  --model "$MODEL" \
  --expert-dir "$EXPERT_DIR" \
  --dataset "$DATASET" \
  --output-path "$EXPERT_DIR/covariance.pt" \
  --cache-dir "$CACHE_DIR" \
  --data-location "$DATA_DIR" \
  "${OVERWRITE_ARGS[@]}"
