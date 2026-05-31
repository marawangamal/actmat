#!/bin/bash
#SBATCH --job-name=hf_eval_wizardlm
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err
# Merge WizardLM-13B / WizardMath-13B / llama-2-13b-code-alpaca directly from
# their HF checkpoints (src/hf/merge.py — no param-folder preprocessing), then
# evaluate the merged model with lm-eval-harness.
#
# Each array task runs one merge method end-to-end (merge + eval).
#
# Usage:
#   sbatch --array=0-$((N-1)) scripts/hf/eval_wizardlm.sh
set -euo pipefail
mkdir -p artifacts/logs

# 0. Setup environment
source "$SCRATCH/actmat/.venv-olmo/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export HF_HOME="$SCRATCH/huggingface"
export SSL_CERT_DIR=/etc/ssl/certs

# ── CONFIG ──────────────────────────────────────────────────────────────────
BASE_MODEL="meta-llama/Llama-2-13b-hf"
EXPERT_MODELS=(
  "WizardLMTeam/WizardLM-13B-V1.2"
  "vanillaOVO/WizardMath-13B-V1.0"
  "layoric/llama-2-13b-code-alpaca"
)
METHODS=(sum mean actmat tsv)
SAVE_ROOT="artifacts/checkpoints/hf"

# embed_tokens/lm_head vocab shapes differ across experts (32000 vs 32001) —
# keep the pretrained values for those keys.
IGNORE_KEEP_PT='embed_tokens|lm_head'

# lm-eval-harness tasks (Wizard suite: math + code).
LM_EVAL_TASKS="gsm8k,minerva_math,humaneval"
BATCH_SIZE="auto"
DTYPE="bfloat16"

# ── Select method for this array task ────────────────────────────────────────
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: submit as a job array:"
  echo "  sbatch --array=0-$((${#METHODS[@]}-1)) scripts/hf/eval_wizardlm.sh"
  exit 1
fi
if (( SLURM_ARRAY_TASK_ID >= ${#METHODS[@]} )); then
  echo "ERROR: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} out of range (${#METHODS[@]} methods)"
  exit 1
fi
method="${METHODS[$SLURM_ARRAY_TASK_ID]}"

# normalize(base) mirrors src/hf/merge.py: "/" -> "_"
BASE_NORM="${BASE_MODEL//\//_}"
MERGED_DIR="${SAVE_ROOT}/${BASE_NORM}/merged/${method}"

echo "============================================================"
echo "Array task ${SLURM_ARRAY_TASK_ID} / Method: ${method}"
echo "Merged dir: ${MERGED_DIR}"
echo "============================================================"

# 1. Merge (src/hf/merge.py self-skips if the dir already exists)
python src/hf/merge.py \
  --base-model "$BASE_MODEL" \
  --expert-models "${EXPERT_MODELS[@]}" \
  --merge-method "$method" \
  --ignore-keep-pt "$IGNORE_KEEP_PT" \
  --save "$SAVE_ROOT"

# 2. Evaluate with lm-eval-harness. Results land under the merged dir;
#    lm-eval writes results_{timestamp}.json itself, so we gate on any existing
#    results file rather than a fixed name.
RESULTS_DIR="${MERGED_DIR}/lm_eval"
if compgen -G "${RESULTS_DIR}/**/results_*.json" > /dev/null 2>&1; then
  echo ">>> Skipping eval: results already exist under ${RESULTS_DIR}"
else
  echo ">>> Evaluating ${MERGED_DIR} on: ${LM_EVAL_TASKS}"
  lm_eval \
    --model hf \
    --model_args "pretrained=${MERGED_DIR},dtype=${DTYPE},parallelize=True" \
    --tasks "$LM_EVAL_TASKS" \
    --batch_size "$BATCH_SIZE" \
    --output_path "$RESULTS_DIR"
fi
