#!/bin/bash
# Shared lm-eval-harness runner for the Polyglot-Teachers (OLMo3-7B) experiment.
# Source this from a script that has already activated .venv-olmo (which has
# lm-eval 0.4.x + vLLM 0.11 + transformers 4.57 + OLMo3 support).
#
# We use lm-eval-harness (a trusted, standard tool we already run) rather than
# the paper's custom lighteval fork. Same datasets (CohereForAI/Global-MMLU-Lite,
# juletxara/mgsm) and same benchmark protocol where it matters:
#   global_mmlu_{ar,de,es}  multiple-choice acc,  0-shot   (paper: CULTURE, MCF)
#   mgsm_direct_{de,es}      generative exact-match, 5-shot, GREEDY  (paper: MATH)
# Absolute numbers won't byte-match the paper's lighteval configs (prompt/norm
# differences), but every model is scored identically — valid for comparing
# merge methods. M-RewardBench (paper's CHAT) is not in lm-eval, so it's omitted.
#
# global_mmlu_<lang> is a GROUP of 6 subject categories; lm-eval reports the
# size-weighted aggregate acc, i.e. the overall Global-MMLU-Lite score.

GMMLU_TASKS="global_mmlu_ar,global_mmlu_de,global_mmlu_es"
MGSM_TASKS="mgsm_direct_de,mgsm_direct_es"
VLLM_COMMON="dtype=bfloat16,gpu_memory_utilization=0.9,max_model_len=8192,tensor_parallel_size=1"

# run_lmeval <model_path_or_hf_id> <out_dir>
# Two invocations because few-shot count differs (Global-MMLU 0-shot, MGSM 5-shot).
# lm-eval writes results_*.json under <out_dir>/{global_mmlu,mgsm}; both skip via
# its own caching only if --use_cache is set, so we guard on the output dir here.
run_lmeval() {
  local model="$1"
  local out="$2"
  mkdir -p "$out"

  if compgen -G "${out}/global_mmlu/**/results_*.json" > /dev/null; then
    echo ">>> skip global_mmlu (exists): ${out}/global_mmlu"
  else
    echo ">>> lm-eval Global-MMLU (0-shot) | model=${model}"
    lm_eval --model vllm \
      --model_args "pretrained=${model},${VLLM_COMMON}" \
      --tasks "$GMMLU_TASKS" --num_fewshot 0 \
      --batch_size auto \
      --output_path "${out}/global_mmlu"
  fi

  if compgen -G "${out}/mgsm/**/results_*.json" > /dev/null; then
    echo ">>> skip mgsm (exists): ${out}/mgsm"
  else
    echo ">>> lm-eval MGSM (5-shot, greedy) | model=${model}"
    lm_eval --model vllm \
      --model_args "pretrained=${model},${VLLM_COMMON}" \
      --tasks "$MGSM_TASKS" --num_fewshot 5 \
      --batch_size auto \
      --output_path "${out}/mgsm"
  fi
}
