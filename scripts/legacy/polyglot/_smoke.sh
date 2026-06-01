#!/bin/bash
#SBATCH --job-name=polyglot_smoke
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/logs/%x_%j.out
#SBATCH --error=artifacts/logs/%x_%j.err
# Cheap end-to-end validation of the lighteval + vllm-0.11 + OLMo3 stack:
# 8 samples on one multiple-choice and one generative task, from the HF hub
# (param-folder exports aren't vLLM-loadable). Throwaway results.
set -euo pipefail
source "$SCRATCH/actmat/.venv-polyglot/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export HF_HOME=$SCRATCH/huggingface
export SSL_CERT_DIR=/etc/ssl/certs

MODEL="ljvmiranda921/Polyglot-OLMo3-7B-SFT-ar"
ARGS="model_name=${MODEL},tensor_parallel_size=1,gpu_memory_utilization=0.9,max_model_length=8192,dtype=bfloat16,generation_parameters={max_new_tokens:4096,temperature:0.6,top_p:0.95}"

for T in "global_mmlu_lite:ar" "mgsm_custom:de|5"; do
  echo ">>> SMOKE task=$T"
  lighteval vllm "$ARGS" "$T" \
    --custom-tasks scripts/polyglot/lighteval_tasks.py \
    --output-dir artifacts/results-polyglot/_smoke \
    --no-public-run --max-samples 8
done
echo ">>> SMOKE DONE"
