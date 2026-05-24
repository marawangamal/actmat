#!/bin/bash
#SBATCH --job-name=eval_gemma9bit
#SBATCH --partition=long
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=8:00:00
#SBATCH --array=0-4
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err
# Merge + evaluate Gemma-2-9B-IT (MergeBench) using EXACTLY MergeBench's
# evaluation setup (scripts/evaluate.sh in github.com/uiuctml/MergeBench).
#
# All evals via lm-eval (HF backend). Code tasks use custom yamls under
# configs/lm_eval_tasks/ (humaneval_plus_mb, mbpp_plus_mb) baking in
# MergeBench's bigcode-eval kwargs: max_gen_toks=512, temp=0.2, top_p=0.95,
# do_sample, repeats=10. The yamls are symlinked into the venv lm_eval/tasks/
# tree at setup time — see scripts/gemma2bit/README.md.
#
# Submitted as a SLURM job array: one array task per merge method.
# a100l (80GB) — L40S 44GB OOMs during merge step (per-layer SVD on stacked
# 9b tensors + 256k vocab needs >30GB intermediate).
set -euo pipefail
mkdir -p artifacts/logs

source "$SCRATCH/actmat/.venv-gemma/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_ALLOW_CODE_EVAL=1

MODEL="gemma-2-9b-it"
METHODS=(tsv actmat wudi mean isoc)
method="${METHODS[$SLURM_ARRAY_TASK_ID]}"

MULTILINGUAL_TASKS="m_mmlu_fr,arc_fr,hellaswag_fr,m_mmlu_es,arc_es,hellaswag_es,m_mmlu_de,arc_de,hellaswag_de,m_mmlu_ru,arc_ru,hellaswag_ru"

MERGED_DIR="artifacts/checkpoints/${MODEL}/${method}"
RESULTS_DIR="artifacts/results/${MODEL}-${method}"

echo "============================================================"
echo "Array task: ${SLURM_ARRAY_TASK_ID}  Method: ${method}"
echo "Merged: ${MERGED_DIR}"
echo "Results: ${RESULTS_DIR}"
echo "============================================================"

# 1. Merge (skip if already done)
if [[ -d "$MERGED_DIR" ]]; then
  echo ">>> Skipping merge: ${MERGED_DIR} already exists"
else
  python scripts/gemma9bit/merge.py \
    --save "artifacts/checkpoints/${MODEL}" \
    --merge-func "$method" \
    --output-dir "$MERGED_DIR"
fi

MODEL_ARGS="pretrained=${MERGED_DIR},dtype=bfloat16"

# 2a. gsm8k_cot — math
OUT="${RESULTS_DIR}/gsm8k_cot"
if compgen -G "${OUT}/**/results_*.json" > /dev/null; then
  echo ">>> Skipping gsm8k_cot: results exist"
else
  mkdir -p "$OUT"
  echo ">>> lm-eval gsm8k_cot"
  lm_eval --model hf --model_args "$MODEL_ARGS" \
    --tasks gsm8k_cot --batch_size 16 --output_path "$OUT"
fi

# 2b. multilingual (12 tasks)
OUT="${RESULTS_DIR}/multilingual"
if compgen -G "${OUT}/**/results_*.json" > /dev/null; then
  echo ">>> Skipping multilingual: results exist"
else
  mkdir -p "$OUT"
  echo ">>> lm-eval multilingual"
  lm_eval --model hf --model_args "$MODEL_ARGS" \
    --tasks "$MULTILINGUAL_TASKS" --batch_size 4 --output_path "$OUT"
fi

# 2c. ifeval — instruction
OUT="${RESULTS_DIR}/ifeval"
if compgen -G "${OUT}/**/results_*.json" > /dev/null; then
  echo ">>> Skipping ifeval: results exist"
else
  mkdir -p "$OUT"
  echo ">>> lm-eval ifeval"
  lm_eval --model hf --model_args "$MODEL_ARGS" \
    --tasks ifeval --batch_size 16 --output_path "$OUT"
fi

# 2d. code — humaneval_plus_mb, mbpp_plus_mb (MergeBench gen kwargs baked into yamls)
OUT="${RESULTS_DIR}/code"
if compgen -G "${OUT}/**/results_*.json" > /dev/null; then
  echo ">>> Skipping code: results exist"
else
  mkdir -p "$OUT"
  echo ">>> lm-eval humaneval_plus_mb,mbpp_plus_mb"
  lm_eval --model hf --model_args "$MODEL_ARGS" \
    --tasks humaneval_plus_mb,mbpp_plus_mb \
    --batch_size 16 --output_path "$OUT" \
    --confirm_run_unsafe_code
fi
