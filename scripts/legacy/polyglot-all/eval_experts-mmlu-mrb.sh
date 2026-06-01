#!/bin/bash
#SBATCH --job-name=polyglot_all_expert_eval
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err
#
# Evaluate each Polyglot OLMo3-7B language expert on ONLY its own language's
# MMLU + MRB tasks, via the lighteval fork (.venv-pg-mmlu-mrb). Experts are the
# HF-hub SFT models — the local param-folder checkpoints aren't vLLM-loadable.
# M-GSM is scored separately by eval_all-mgsm.sh (lm-eval native CoT).
#
#   ar -> global_mmlu_lite:ar, mrewardbench_mcf:ar
#   cs -> mrewardbench_mcf:cs
#   de -> global_mmlu_lite:de, mrewardbench_mcf:de
#   es -> global_mmlu_lite:es, mrewardbench_mcf:es
#
# Usage:
#   sbatch scripts/polyglot-all/eval_experts-mmlu-mrb.sh
#   EXPERT=de bash scripts/polyglot-all/eval_experts-mmlu-mrb.sh   # single expert, no array
set -euo pipefail

ACTMAT="$SCRATCH/actmat"
cd "$ACTMAT"
mkdir -p artifacts/logs

# Stagger concurrent array starts: transformers' lazy import scan races on a
# shared networked venv when several tasks cold-import at once.
sleep $(( ${SLURM_ARRAY_TASK_ID:-0} * 60 ))

LANGS=(ar cs de es)
declare -A TASKS_FOR=(
  [ar]="global_mmlu_lite:ar,mrewardbench_mcf:ar"
  [cs]="mrewardbench_mcf:cs"
  [de]="global_mmlu_lite:de,mrewardbench_mcf:de"
  [es]="global_mmlu_lite:es,mrewardbench_mcf:es"
)

# Pick expert: array index if set, else $EXPERT env, else default.
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  EXPERT="${LANGS[$SLURM_ARRAY_TASK_ID]}"
else
  EXPERT="${EXPERT:-ar}"
fi
MODEL="ljvmiranda921/Polyglot-OLMo3-7B-SFT-${EXPERT}"
TASK="${TASKS_FOR[$EXPERT]}"
RESULTS_DIR="artifacts/results-polyglot-all/expert-${EXPERT}"

source "$ACTMAT/.venv-pg-mmlu-mrb/bin/activate"
export HF_HOME=$SCRATCH/huggingface
export NLTK_DATA=$SCRATCH/nltk_data
export SSL_CERT_DIR=/etc/ssl/certs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Expert ${EXPERT}: ${MODEL} on ${TASK} ==="
lighteval vllm "model_name=${MODEL},tensor_parallel_size=1,gpu_memory_utilization=0.9,max_model_length=8192,dtype=bfloat16,generation_parameters={max_new_tokens:4096,temperature:0.6,top_p:0.95}" "${TASK}" \
    --custom-tasks scripts/polyglot-all/lighteval_tasks.py \
    --output-dir "$RESULTS_DIR" \
    --results-path-template '{output_dir}/{org}___{model}' \
    --save-details

echo ">>> done: ${RESULTS_DIR}"
