#!/bin/bash
#SBATCH --job-name=eval_llama8b_single
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --output=artifacts/logs/%x_%j.out
#SBATCH --error=artifacts/logs/%x_%j.err
# Evaluate each MergeBench Llama-3.1-8B-Instruct expert on its corresponding
# task(s). Used as the "best individual expert" baseline.
#
# Mirrors scripts/gemma2bit/eval_single_task.sh — all evals via lm-eval to keep
# parity with eval_task_addition.sh.
#
# Usage:
#   sbatch scripts/llama8b/eval_single_task.sh
set -euo pipefail
mkdir -p artifacts/logs

source "$SCRATCH/actmat/.venv-gemma/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_ALLOW_CODE_EVAL=1
export HF_METRICS_CACHE="${SCRATCH}/huggingface/metrics_single_${SLURM_JOB_ID}"
mkdir -p "$HF_METRICS_CACHE"

MULTILINGUAL_TASKS="m_mmlu_fr,arc_fr,hellaswag_fr,m_mmlu_es,arc_es,hellaswag_es,m_mmlu_de,arc_de,hellaswag_de,m_mmlu_ru,arc_ru,hellaswag_ru"

# Capability → (tasks, batch_size)
declare -A EXPERT_TASKS
EXPERT_TASKS[MergeBench/Llama-3.1-8B-Instruct_instruction]="ifeval:64"
EXPERT_TASKS[MergeBench/Llama-3.1-8B-Instruct_math]="gsm8k_cot:64"
EXPERT_TASKS[MergeBench/Llama-3.1-8B-Instruct_coding]="humaneval_plus_mb,mbpp_plus_mb:64"
EXPERT_TASKS[MergeBench/Llama-3.1-8B-Instruct_multilingual]="${MULTILINGUAL_TASKS}:16"

for MODEL_ID in "${!EXPERT_TASKS[@]}"; do
  spec=${EXPERT_TASKS[$MODEL_ID]}
  TASKS="${spec%:*}"
  BS="${spec##*:}"
  MODEL_FNAME="$(basename "$MODEL_ID")"
  OUTPUT_DIR="artifacts/results/Llama-3.1-8B-Instruct-expert-${MODEL_FNAME}"

  if compgen -G "${OUTPUT_DIR}/**/results_*.json" > /dev/null; then
    echo ">>> Skipping ${MODEL_FNAME}: results already exist"
    continue
  fi

  mkdir -p "$OUTPUT_DIR"
  echo "============================================================"
  echo "Model      : ${MODEL_ID}"
  echo "Tasks      : ${TASKS}"
  echo "Batch size : ${BS}"
  echo "Output     : ${OUTPUT_DIR}"
  echo "============================================================"

  EXTRA=""
  case "$TASKS" in
    *humaneval_plus_mb*|*mbpp_plus_mb*) EXTRA="--confirm_run_unsafe_code" ;;
  esac

  lm_eval --model hf \
    --model_args "pretrained=${MODEL_ID},dtype=bfloat16" \
    --tasks "$TASKS" \
    --batch_size "$BS" \
    --output_path "$OUTPUT_DIR" \
    $EXTRA
done
