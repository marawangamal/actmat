#!/bin/bash
#SBATCH --job-name=eval_gemma2bit_single
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=artifacts/logs/%x_%j.out
#SBATCH --error=artifacts/logs/%x_%j.err
# Evaluate each MergeBench Gemma-2-2B-IT expert on its corresponding task(s).
# Used as a "best individual expert" baseline.
#
# Usage:
#   sbatch scripts/gemma2bit/eval_single_task.sh

set -euo pipefail
mkdir -p artifacts/logs

# 0. Setup environment
source "$SCRATCH/actmat/.venv-gemma/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

OLMES_MODEL_ARGS='{"gpu_memory_utilization": 0.8, "trust_remote_code": false, "max_length": 8192}'
GPUS=1
BATCH_SIZE=64
LM_EVAL_BATCH_SIZE=8
MULTILINGUAL_TASKS="m_mmlu_fr,arc_fr,hellaswag_fr,m_mmlu_es,arc_es,hellaswag_es,m_mmlu_de,arc_de,hellaswag_de,m_mmlu_ru,arc_ru,hellaswag_ru"

# Capability → (harness, tasks)
declare -A OLMES_TASKS
OLMES_TASKS[MergeBench/gemma-2-2b-it_instruction]="ifeval::tulu"
OLMES_TASKS[MergeBench/gemma-2-2b-it_math]="gsm8k::tulu"
OLMES_TASKS[MergeBench/gemma-2-2b-it_coding]="codex_humanevalplus::tulu mbppplus:0-shot-chat"

# 1. olmes evals (instruction, math, code experts)
for MODEL_ID in "${!OLMES_TASKS[@]}"; do
  TASKS=${OLMES_TASKS[$MODEL_ID]}
  MODEL_FNAME="$(basename "$MODEL_ID")"
  OUTPUT_DIR="artifacts/results/gemma-2-2b-it-expert-${MODEL_FNAME}/olmes"

  if [[ -f "$OUTPUT_DIR/metrics.json" ]]; then
    echo ">>> Skipping ${MODEL_FNAME}: ${OUTPUT_DIR}/metrics.json already exists"
    continue
  fi

  echo "============================================================"
  echo "Model      : ${MODEL_ID}"
  echo "Tasks      : ${TASKS}"
  echo "Output     : ${OUTPUT_DIR}"
  echo "============================================================"

  olmes \
    --model "$MODEL_ID" \
    --task $TASKS \
    --output-dir "$OUTPUT_DIR" \
    --gpus "$GPUS" \
    --model-type vllm \
    --model-args "$OLMES_MODEL_ARGS" \
    --batch-size "$BATCH_SIZE" \
    --num-workers 1
done

# 2. lm-eval for multilingual expert
MULTI_MODEL="MergeBench/gemma-2-2b-it_multilingual"
MULTI_OUTPUT_DIR="artifacts/results/gemma-2-2b-it-expert-$(basename $MULTI_MODEL)/lm-eval"

if compgen -G "${MULTI_OUTPUT_DIR}/results_*.json" > /dev/null; then
  echo ">>> Skipping multilingual: ${MULTI_OUTPUT_DIR}/results_*.json already exists"
else
  mkdir -p "$MULTI_OUTPUT_DIR"
  echo "============================================================"
  echo "Model      : ${MULTI_MODEL}"
  echo "Tasks      : ${MULTILINGUAL_TASKS}"
  echo "Output     : ${MULTI_OUTPUT_DIR}"
  echo "============================================================"
  lm_eval \
    --model hf \
    --model_args "pretrained=${MULTI_MODEL},dtype=bfloat16" \
    --tasks "$MULTILINGUAL_TASKS" \
    --batch_size "$LM_EVAL_BATCH_SIZE" \
    --output_path "$MULTI_OUTPUT_DIR"
fi
