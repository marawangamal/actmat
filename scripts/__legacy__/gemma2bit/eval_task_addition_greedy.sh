#!/bin/bash
#SBATCH --job-name=eval_gemma2bit_greedy
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --array=0-11
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err
# Greedy variant of eval_task_addition.sh — all five stages decoded with
# do_sample=false / temperature=0 / repeats=1.
#   • gsm8k_cot, ifeval, multilingual: already greedy via lm-eval defaults.
#   • code:  humaneval_plus_greedy (custom yaml, greedy pass@1).
#   • mbpp+: mbpp_plus_greedy       (custom yaml, greedy pass@1).
#
# Results go to artifacts/results/${MODEL}-${method}-greedy/ so they sit
# side-by-side with the sampling-based runs.
set -euo pipefail
mkdir -p artifacts/logs

source "$SCRATCH/actmat/.venv-gemma/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_ALLOW_CODE_EVAL=1
export HF_METRICS_CACHE="${SCRATCH}/huggingface/metrics_arr_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$HF_METRICS_CACHE"

MODEL="gemma-2-2b-it"
METHODS=(tsv actmat wudi mean isoc actmat_gd isoc2 isoc3 actmat_5k ace actmat_gd_5k actmat_mons)
method="${METHODS[$SLURM_ARRAY_TASK_ID]}"

MULTILINGUAL_TASKS="m_mmlu_fr,arc_fr,hellaswag_fr,m_mmlu_es,arc_es,hellaswag_es,m_mmlu_de,arc_de,hellaswag_de,m_mmlu_ru,arc_ru,hellaswag_ru"

MERGED_DIR="artifacts/checkpoints/${MODEL}/${method}"
RESULTS_DIR="artifacts/results/${MODEL}-${method}-greedy"

echo "============================================================"
echo "Array task: ${SLURM_ARRAY_TASK_ID}  Method: ${method}  (greedy)"
echo "Merged: ${MERGED_DIR}"
echo "Results: ${RESULTS_DIR}"
echo "============================================================"

# 1. Merge (reuse if already done — same checkpoint as non-greedy)
if [[ -d "$MERGED_DIR" ]]; then
  echo ">>> Skipping merge: ${MERGED_DIR} already exists"
else
  python scripts/gemma2bit/merge.py \
    --save "artifacts/checkpoints/${MODEL}" \
    --merge-func "$method" \
    --output-dir "$MERGED_DIR"
fi

MODEL_ARGS="pretrained=${MERGED_DIR},dtype=bfloat16"

# 2a. gsm8k_cot — already greedy by default
OUT="${RESULTS_DIR}/gsm8k_cot"
if compgen -G "${OUT}/**/results_*.json" > /dev/null; then
  echo ">>> Skipping gsm8k_cot: results exist"
else
  mkdir -p "$OUT"
  echo ">>> lm-eval gsm8k_cot"
  lm_eval --model hf --model_args "$MODEL_ARGS" \
    --tasks gsm8k_cot --batch_size 64 --output_path "$OUT"
fi

# 2b. multilingual (12 tasks) — loglikelihood / MC, decoding-agnostic
OUT="${RESULTS_DIR}/multilingual"
if compgen -G "${OUT}/**/results_*.json" > /dev/null; then
  echo ">>> Skipping multilingual: results exist"
else
  mkdir -p "$OUT"
  echo ">>> lm-eval multilingual"
  lm_eval --model hf --model_args "$MODEL_ARGS" \
    --tasks "$MULTILINGUAL_TASKS" --batch_size 8 --output_path "$OUT"
fi

# 2c. ifeval — already greedy by default
OUT="${RESULTS_DIR}/ifeval"
if compgen -G "${OUT}/**/results_*.json" > /dev/null; then
  echo ">>> Skipping ifeval: results exist"
else
  mkdir -p "$OUT"
  echo ">>> lm-eval ifeval"
  lm_eval --model hf --model_args "$MODEL_ARGS" \
    --tasks ifeval --batch_size 64 --output_path "$OUT"
fi

# 2d. humaneval_plus — GREEDY pass@1 (do_sample=false, repeats=1)
OUT="${RESULTS_DIR}/code"
if compgen -G "${OUT}/**/results_*.json" > /dev/null; then
  echo ">>> Skipping code (humaneval_plus_greedy): results exist"
else
  mkdir -p "$OUT"
  echo ">>> lm-eval humaneval_plus_greedy"
  lm_eval --model hf --model_args "$MODEL_ARGS" \
    --tasks humaneval_plus_greedy \
    --batch_size 64 --output_path "$OUT" \
    --confirm_run_unsafe_code
fi

# 2e. mbpp_plus — GREEDY pass@1
OUT="${RESULTS_DIR}/mbpp_plus"
if compgen -G "${OUT}/**/results_*.json" > /dev/null; then
  echo ">>> Skipping mbpp_plus: results exist"
else
  mkdir -p "$OUT"
  echo ">>> lm-eval mbpp_plus_greedy"
  lm_eval --model hf --model_args "$MODEL_ARGS" \
    --tasks mbpp_plus_greedy \
    --batch_size 64 --output_path "$OUT" \
    --confirm_run_unsafe_code
fi
