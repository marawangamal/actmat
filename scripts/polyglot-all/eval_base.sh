#!/bin/bash
#SBATCH --job-name=polyglot_all_base_eval
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=artifacts/logs/%x_%j.out
#SBATCH --error=artifacts/logs/%x_%j.err
#
# Evaluate the BASE model (S_phi = OLMo-3-1025-7B, the paper's PGR 0-point) on
# the full ar/cs/de/es task suite, so the base row is complete and apples-to-
# apples with the experts/merges. Same lighteval stack as tesh.sh.
#
# Usage:
#   sbatch scripts/polyglot-all/eval_base.sh
set -euo pipefail

ACTMAT="$SCRATCH/actmat"
cd "$ACTMAT"
mkdir -p artifacts/logs

PG="$ACTMAT/polyglot-teachers"
MODEL="allenai/Olmo-3-1025-7B"
RESULTS_DIR="artifacts/results-polyglot-all/base-Olmo-3-1025-7B"
TASKS=(
    "global_mmlu_lite:ar" "global_mmlu_lite:de" "global_mmlu_lite:es"
    "mrewardbench_mcf:ar" "mrewardbench_mcf:cs" "mrewardbench_mcf:de" "mrewardbench_mcf:es"
    "mgsm_custom:de|5" "mgsm_custom:es|5"
)
TASK=$(IFS=,; echo "${TASKS[*]}")

source "$PG/.venv/bin/activate"
export HF_HOME=$SCRATCH/huggingface
export NLTK_DATA=$SCRATCH/nltk_data
export SSL_CERT_DIR=/etc/ssl/certs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Base ${MODEL} on: ${TASK} ==="
lighteval vllm "model_name=${MODEL},tensor_parallel_size=1,gpu_memory_utilization=0.9,max_model_length=8192,dtype=bfloat16,generation_parameters={max_new_tokens:4096,temperature:0.6,top_p:0.95}" "${TASK}" \
    --custom-tasks "$PG/scripts/lighteval_tasks.py" \
    --output-dir "$RESULTS_DIR" \
    --results-path-template '{output_dir}/{org}___{model}' \
    --save-details

echo ">>> done: ${RESULTS_DIR}"
