#!/bin/bash
#SBATCH --job-name=polyglot_all_sanity
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=artifacts/logs/%x_%j.out
#SBATCH --error=artifacts/logs/%x_%j.err
#
# Sanity smoke for the lighteval (MMLU+MRB) env (.venv-pg-mmlu-mrb): run ONE
# task on a known model and eyeball the score against the references below.
#
# Usage:
#   bash scripts/polyglot-all/test.sh

set -euo pipefail
ACTMAT="$SCRATCH/actmat"
cd "$ACTMAT"

source "$ACTMAT/.venv-pg-mmlu-mrb/bin/activate"
export HF_HOME=$SCRATCH/huggingface
export NLTK_DATA=$SCRATCH/nltk_data
export SSL_CERT_DIR=/etc/ssl/certs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RESULTS_DIR="artifacts/results-polyglot-sanity"
# TASK="mrewardbench_mcf:ar"
TASK="global_mmlu_lite:ar"
MODEL_PATH=allenai/Olmo-3-7B-Instruct
echo "=== Evaluating ${MODEL_PATH} on: ${TASK} ==="
lighteval vllm "model_name=${MODEL_PATH},tensor_parallel_size=1,gpu_memory_utilization=0.9,max_model_length=8192,dtype=bfloat16,generation_parameters={max_new_tokens:4096,temperature:0.6,top_p:0.95}" "${TASK}" \
    --custom-tasks scripts/polyglot-all/lighteval_tasks.py \
    --output-dir "$RESULTS_DIR" \
    --results-path-template '{output_dir}/{org}___{model}' \
    --save-details

echo ">>> done: results=${RESULTS_DIR}"

# mrewardbench_mcf:ar
# "acc_norm_token": 0.4932032066922273,
# "acc_norm_token_stderr": 0.00933555842923963,
# "weighted_acc": 0.5028495236160122,
# "weighted_acc_stderr": 0.00035655346438800186,
# "weighted_acc_chat": 0.5743243243243243,
# "weighted_acc_chat_stderr": 0.0009412660465849412,
# "weighted_acc_chat_hard": 0.4668304668304668,
# "weighted_acc_chat_hard_stderr": 0.0008126131593780779,
# "weighted_acc_safety": 0.48641304347826086,
# "weighted_acc_safety_stderr": 0.000540571811258277,
# "weighted_acc_reasoning": 0.48383025983099714,
# "weighted_acc_reasoning_stderr": 0.00043616330972176705

# global_mmlu_lite:ar
# |        Task         |Version|Metric|Value |   |Stderr|
# |---------------------|-------|------|-----:|---|-----:|
# |all                  |       |acc   |0.3175|±  |0.0233|
# |global_mmlu_lite:ar:0|       |acc   |0.3175|±  |0.0233|