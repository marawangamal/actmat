#!/bin/bash
#SBATCH --job-name=polyglot_base_eval
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Evaluate the base Olmo-3-7B on the paper's full polyglot task suite using the
# ljvmiranda921/lighteval fork + custom tasks (scripts/lighteval_tasks.py).
# See SETUP.md for how the .venv (PyPI stack + fork overlay) is built.
#
# Usage (submit from the repo root):
#   cd /network/scratch/m/marawan.gamal/actmat/polyglot-teachers
#   sbatch tesh.sh

set -euo pipefail

REPO="/network/scratch/m/marawan.gamal/actmat/polyglot-teachers"
cd "$REPO"
mkdir -p logs

source .venv/bin/activate
export PYTHONPATH="$PYTHONPATH:$REPO"
export HF_HOME=$SCRATCH/huggingface
export NLTK_DATA=$SCRATCH/nltk_data
export SSL_CERT_DIR=/etc/ssl/certs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Experts available: ar, cs, de, es. ja/id tasks commented out (no experts).
# TASKS=(
#     "global_mmlu_lite:de"
#     "global_mmlu_lite:es"
#     # "global_mmlu_lite:ja"
#     "mrewardbench_mcf:de"
#     "mrewardbench_mcf:es"
#     # "mrewardbench_mcf:ja"
#     "mgsm_custom:de|5"
#     "mgsm_custom:es|5"
#     # "mgsm_custom:ja|5"
#     "global_mmlu_lite:ar"
#     "mrewardbench_mcf:ar"
#     # "global_mmlu_lite:id"
#     # "mrewardbench_mcf:id"
#     "mrewardbench_mcf:cs"
# )

TASKS=(
    "mrewardbench_mcf:de"
    "mrewardbench_mcf:es"
    "mrewardbench_mcf:ar"
    "mrewardbench_mcf:cs"
)

# Join all tasks with commas so lighteval loads the model once and runs them sequentially.
TASK=$(IFS=,; echo "${TASKS[*]}")
MODEL="allenai/Olmo-3-1025-7B"

echo "Evaluating model: ${MODEL} on tasks: ${TASK}"

# Reference for gsm8k setup: https://github.com/huggingface/lighteval/issues/686
lighteval vllm "model_name=${MODEL},tensor_parallel_size=1,gpu_memory_utilization=0.9,max_model_length=8192,dtype=bfloat16,generation_parameters={max_new_tokens:4096,temperature:0.6,top_p:0.95}" "${TASK}" \
    --custom-tasks scripts/lighteval_tasks.py \
    --output-dir lighteval-results \
    --results-path-template '{output_dir}/{org}___{model}' \
    --save-details
