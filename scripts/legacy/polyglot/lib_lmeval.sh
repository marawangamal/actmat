#!/bin/bash
# Shared lm-eval-harness runner for the Polyglot-Teachers (OLMo3-7B) experiment.
# Source this from a script that has already activated the lm-eval venv
# (.venv-pg-mgsm for polyglot-all; .venv-pg for the older polyglot path) —
# lm-eval 0.4.12 + transformers 4.57.6 + OLMo3 support. Pin to one venv so every
# model is scored under the SAME lm-eval version — task definitions change between
# releases (e.g. mgsm_direct's few-shot target was fixed in 3.0->4.0).
#
# We use lm-eval-harness (a trusted, standard tool we already run) rather than
# the paper's custom lighteval fork. Same datasets (CohereForAI/Global-MMLU-Lite,
# juletxara/mgsm) and same benchmark protocol where it matters:
#   global_mmlu_{ar,de,es}     multiple-choice acc,  0-shot   (paper: CULTURE, MCF)
#   mgsm_native_cot_{de,es}    generative exact-match, 5-shot, greedy, native-language
#                              chain-of-thought  (paper: MATH / M-GSM)
# NB: we use the CoT variant, not mgsm_direct — MGSM is conventionally chain-of-
# thought and that matches the paper's Lighteval setup. mgsm_direct (no reasoning)
# scores far lower (~0.16 vs ~0.48 for base OLMo-3) and is the wrong protocol here.
# Absolute numbers won't byte-match the paper's lighteval configs (prompt/norm
# differences), but every model is scored identically — valid for comparing
# merge methods. M-RewardBench (paper's CHAT) is not in lm-eval, so it's omitted.
#
# global_mmlu_<lang> is a GROUP of 6 subject categories; lm-eval reports the
# size-weighted aggregate acc, i.e. the overall Global-MMLU-Lite score.
#
# Backend: the plain HuggingFace transformers backend (--model hf), NOT vLLM.
# vLLM's engine-core subprocess proved fragile on interactive nodes (orphaned
# EngineCore procs holding the GPU); the hf backend is slower but has no
# subprocess and is robust. Our eval sets are small so the slowdown is fine.

GMMLU_TASKS="global_mmlu_ar,global_mmlu_de,global_mmlu_es"
MGSM_TASKS="mgsm_native_cot_de,mgsm_native_cot_es"
# max_length=4096 (not 8192): MCQ/math sequences are short, and 8192 at batch 32
# OOMs a 44GB L40S during Global-MMLU loglikelihood batching.
HF_COMMON="dtype=bfloat16,max_length=4096,trust_remote_code=False"
# Fixed batch size (not 'auto' — the auto search is slow/flaky). 16 is a safe fit
# for a 7B bf16 model on a 44GB L40S; bump via env on bigger GPUs.
BATCH_SIZE="${BATCH_SIZE:-16}"
# Per-benchmark toggles (default: run both). Set RUN_GMMLU=0 for MGSM-only, etc.
RUN_GMMLU="${RUN_GMMLU:-1}"
RUN_MGSM="${RUN_MGSM:-1}"

# run_lmeval <model_path_or_hf_id> <out_dir>
# Two invocations because few-shot count differs (Global-MMLU 0-shot, MGSM 5-shot).
# lm-eval writes results_*.json under <out_dir>/{global_mmlu,mgsm}; both skip via
# its own caching only if --use_cache is set, so we guard on the output dir here.
run_lmeval() {
  local model="$1"
  local out="$2"
  mkdir -p "$out"

  if [[ "$RUN_GMMLU" != "1" ]]; then
    echo ">>> skip global_mmlu (RUN_GMMLU=$RUN_GMMLU)"
  elif compgen -G "${out}/global_mmlu/**/results_*.json" > /dev/null; then
    echo ">>> skip global_mmlu (exists): ${out}/global_mmlu"
  else
    echo ">>> lm-eval Global-MMLU (0-shot) | model=${model}"
    lm_eval --model hf \
      --model_args "pretrained=${model},${HF_COMMON}" \
      --tasks "$GMMLU_TASKS" --num_fewshot 0 \
      --batch_size "$BATCH_SIZE" \
      --output_path "${out}/global_mmlu"
  fi

  if [[ "$RUN_MGSM" != "1" ]]; then
    echo ">>> skip mgsm (RUN_MGSM=$RUN_MGSM)"
  elif compgen -G "${out}/mgsm/**/results_*.json" > /dev/null; then
    echo ">>> skip mgsm (exists): ${out}/mgsm"
  else
    echo ">>> lm-eval MGSM (5-shot, greedy) | model=${model}"
    lm_eval --model hf \
      --model_args "pretrained=${model},${HF_COMMON}" \
      --tasks "$MGSM_TASKS" --num_fewshot 5 \
      --batch_size "$BATCH_SIZE" \
      --output_path "${out}/mgsm"
  fi
}
