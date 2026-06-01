#!/bin/bash
#SBATCH --job-name=polyglot_lighteval
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=artifacts/logs/%x_%j.out
#SBATCH --error=artifacts/logs/%x_%j.err
# Evaluate a merged Polyglot model with the PAPER'S OWN eval stack: the
# ljvmiranda921/lighteval fork + their custom tasks (scripts/lighteval_tasks.py).
# This reproduces all THREE paper benchmarks, including M-RewardBench (CHAT),
# which is absent from lm-eval-harness.
#
#   global_mmlu_lite:{ar,de,es}   CULTURE   MCF acc        0-shot
#   mrewardbench_mcf:{ar,cs,de,es} CHAT     weighted acc   0-shot
#   mgsm_custom:{de,es}|5          MATH     extractive m.  5-shot
#
# Env: .glot (Python 3.11 + their lighteval fork 0.13.1dev0 + vllm 0.11 +
# transformers 4.57.6), built per their experiments/jobs/sync_isambard.sh.
# We deviate from their command in three ways: (1) local model path instead of
# an HF id, (2) GREEDY decoding (temperature=0) instead of temp=0.6 sampling,
# (3) no --push-to-hub / --results-org (keep results local).
#
# Usage:
#   MODEL_DIR=artifacts/checkpoints/Olmo-3-7b-polyglot/merged-mean \
#     sbatch scripts/polyglot/eval_lighteval.sh
#   MODEL_DIR=allenai/Olmo-3-1025-7B sbatch scripts/polyglot/eval_lighteval.sh   # base ref
set -euo pipefail
mkdir -p artifacts/logs

source "$SCRATCH/actmat/.glot/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export HF_HOME=$SCRATCH/huggingface
export SSL_CERT_DIR=/etc/ssl/certs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO="$SCRATCH/polyglot-teachers"          # their repo (has scripts/lighteval_tasks.py)
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to a local merged dir or an HF id}"
OUT="${OUT:-artifacts/results-lighteval}"
mkdir -p "$OUT"

# Their task suite (greedy: temperature 0). MGSM is the only 5-shot task (|5).
TASKS="global_mmlu_lite:ar,global_mmlu_lite:de,global_mmlu_lite:es,\
mrewardbench_mcf:ar,mrewardbench_mcf:cs,mrewardbench_mcf:de,mrewardbench_mcf:es,\
mgsm_custom:de|5,mgsm_custom:es|5"

MODEL_ARGS="model_name=${MODEL_DIR},tensor_parallel_size=1,gpu_memory_utilization=0.9,\
max_model_length=8192,dtype=bfloat16,generation_parameters={max_new_tokens:4096,temperature:0.0}"

echo "=== lighteval | model=${MODEL_DIR} ==="
lighteval vllm "$MODEL_ARGS" "$TASKS" \
  --custom-tasks "${REPO}/scripts/lighteval_tasks.py" \
  --output-dir "$OUT" \
  --results-path-template '{output_dir}/{org}___{model}' \
  --no-public-run \
  --save-details

echo ">>> done. results under ${OUT}/"
