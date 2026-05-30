#!/bin/bash
#SBATCH --job-name=polyglot_all_merge_eval
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err
#
# For one data-free method (mean/tsv/actmat) per array task:
#   1. merge the 4 Polyglot OLMo3-7B experts (ar, cs, de, es)   [CPU, .venv-pg]
#   2. eval the merged model with the lighteval fork            [GPU, polyglot-teachers/.venv]
# Eval mirrors polyglot-teachers/tesh.sh (same model args + custom tasks).
#
# Usage:
#   sbatch scripts/polyglot-all/merge.sh
#   METHOD=actmat bash scripts/polyglot-all/merge.sh   # single method, no array
set -euo pipefail

ACTMAT="$SCRATCH/actmat"
cd "$ACTMAT"
mkdir -p artifacts/logs

# Stagger concurrent array starts: transformers' lazy import scan races (TOCTOU
# ENOENT) when several tasks cold-import from the shared networked venv at once.
sleep $(( ${SLURM_ARRAY_TASK_ID:-0} * 60 ))

PG="$ACTMAT/polyglot-teachers"                 # lighteval fork + custom tasks
MODEL="Olmo-3-7b-polyglot-all"
BASE="artifacts/checkpoints/${MODEL}"
LANGS=(ar cs de es)
METHODS=(mean tsv actmat isoc wudi actmat_gd ties dare_ties)
TASKS=(
    "global_mmlu_lite:ar" "global_mmlu_lite:de" "global_mmlu_lite:es"
    "mrewardbench_mcf:ar" "mrewardbench_mcf:cs" "mrewardbench_mcf:de" "mrewardbench_mcf:es"
    "mgsm_custom:de|5" "mgsm_custom:es|5"
)

# Method = array index if set, else $METHOD env, else default.
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  METHOD="${METHODS[$SLURM_ARRAY_TASK_ID]}"
else
  METHOD="${METHOD:-mean}"
fi
MERGED_DIR="${BASE}/merged-${METHOD}"
RESULTS_DIR="artifacts/results-polyglot-all/${MODEL}-${METHOD}"

# ── 1. Merge (CPU, .venv-pg) ──────────────────────────────────────────────────
source "$ACTMAT/.venv-pg/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export HF_HOME=$SCRATCH/huggingface
if [[ -d "$MERGED_DIR" ]]; then
  echo ">>> Skipping merge: ${MERGED_DIR} already exists"
else
  echo "=== Merging [${LANGS[*]}] with '${METHOD}' -> ${MERGED_DIR} ==="
  python scripts/olmo/merge.py \
    --save "$BASE" \
    --merge-tasks "${LANGS[@]}" \
    --merge-func "$METHOD" \
    --output-dir "$MERGED_DIR"
fi
deactivate

# ── 2. Eval (GPU, lighteval fork) ─────────────────────────────────────────────
source "$PG/.venv/bin/activate"
export HF_HOME=$SCRATCH/huggingface
export NLTK_DATA=$SCRATCH/nltk_data
export SSL_CERT_DIR=/etc/ssl/certs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TASK=$(IFS=,; echo "${TASKS[*]}")
MODEL_PATH="$(realpath "$MERGED_DIR")"
echo "=== Evaluating ${MODEL_PATH} on: ${TASK} ==="
lighteval vllm "model_name=${MODEL_PATH},tensor_parallel_size=1,gpu_memory_utilization=0.9,max_model_length=8192,dtype=bfloat16,generation_parameters={max_new_tokens:4096,temperature:0.6,top_p:0.95}" "${TASK}" \
    --custom-tasks "$PG/scripts/lighteval_tasks.py" \
    --output-dir "$RESULTS_DIR" \
    --results-path-template '{output_dir}/{org}___{model}' \
    --save-details

echo ">>> done: merged=${MERGED_DIR} results=${RESULTS_DIR}"
