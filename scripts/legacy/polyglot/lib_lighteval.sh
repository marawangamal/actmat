#!/bin/bash
# Shared Lighteval runner for the Polyglot-Teachers (OLMo3-7B) experiments.
# Source this from a SLURM script that has already activated .venv-polyglot.
#
# Replicates the paper's eval (arXiv 2604.11290, App. E): Lighteval 0.13.x,
# vLLM backend, bf16, custom tasks in scripts/polyglot/lighteval_tasks.py.
# M-RewardBench is SKIPPED here (it needs the authors' lighteval fork); the
# remaining two benchmarks run on a stock lighteval release. For OLMo3 langs:
#     global_mmlu_lite:{ar,de,es}   MCF, acc,      0-shot
#     mgsm_custom:{de,es}|5          generative EM, 5-shot
#   (cs absent from Global-MMLU-Lite; ar/cs absent from MGSM, so cs has no
#    stock-runnable task and is covered only by the merge itself.)

# Tasks restricted to the languages the OLMo3 experts cover (ar, cs, de, es).
POLYGLOT_TASKS=(
  "global_mmlu_lite:ar"
  "global_mmlu_lite:de"
  "global_mmlu_lite:es"
  "mgsm_custom:de|5"
  "mgsm_custom:es|5"
)

CUSTOM_TASKS="scripts/polyglot/lighteval_tasks.py"
# Match the paper's vLLM/generation settings verbatim.
GEN_PARAMS="generation_parameters={max_new_tokens:4096,temperature:0.6,top_p:0.95}"
MODEL_COMMON="tensor_parallel_size=1,gpu_memory_utilization=0.9,max_model_length=8192,dtype=bfloat16,${GEN_PARAMS}"

# run_lighteval <model_path_or_hf_id> <output_dir>
# Runs every POLYGLOT_TASKS task, skipping any whose results file already exists.
run_lighteval() {
  local model="$1"
  local out_dir="$2"
  mkdir -p "$out_dir"

  for task in "${POLYGLOT_TASKS[@]}"; do
    echo "=========================================================="
    echo ">>> lighteval | model=${model} | task=${task}"
    echo "=========================================================="
    lighteval vllm \
      "model_name=${model},${MODEL_COMMON}" \
      "${task}" \
      --custom-tasks "${CUSTOM_TASKS}" \
      --output-dir "${out_dir}" \
      --no-public-run \
      --save-details
    echo ">>> sleeping 15s before next task"
    sleep 15
  done
}
