#!/bin/bash
#SBATCH --job-name=hf_eval_olmo_grid
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err
# Layer x method hybrid grid (docs/experiments.md). Each array task = one cell:
# merge one layer type with one method, everything else with mean, then eval.
# Cells are assembled on the fly from the per-layer expert splits
# (scripts/analysis/split_expert.py must have been run for mean + every METHOD first).
#
# Tasks split by chat template, exactly like eval_olmo_v2.sh:
#   ct-code view (Code/IF template) -> codex_humaneval(+), ifeval
#   ct-math view (Math template)    -> minerva_math_500, gsm8k
# Each view is one olmes load; we assemble both views (identical weight symlinks,
# only the copied chat template/tokenizer differ) and run each task group once.
# Results for a (layer, method) cell land under their own dir, so submitting just
# one method's column (e.g. the actmat_herm indices) won't touch the others.
#
# Grid = LAYER_TYPES x METHODS, flattened row-major (index = li*nMethods + mi).
# Submit with: sbatch --array=0-$((7*3-1)) scripts/hybrid/eval_olmo_grid.sh
set -euo pipefail

source "$SCRATCH/actmat/.venv-olmo/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export HF_HOME="$SCRATCH/huggingface"
export SSL_CERT_DIR=/etc/ssl/certs

LAYER_TYPES=(q_proj k_proj v_proj o_proj gate_proj up_proj down_proj)
METHODS=(isoc tsv actmat_herm)
BACKGROUND="mean"

N_METHODS=${#METHODS[@]}
LI=$(( SLURM_ARRAY_TASK_ID / N_METHODS ))
MI=$(( SLURM_ARRAY_TASK_ID % N_METHODS ))
LAYER="${LAYER_TYPES[$LI]}"
METHOD="${METHODS[$MI]}"

HYBRID_ROOT="artifacts/checkpoints/Olmo-3-7b/group-hybrid"
EXPERTS_ROOT="$HYBRID_ROOT/experts"
MERGED_ROOT="artifacts/checkpoints/Olmo-3-7b/group-rl-zero/merged"
# Reference merges = source of the key list + the per-template tokenizer/chat
# template. mean carries the Math template; mean-ct-code carries the Code/IF one.
REF_MATH="$MERGED_ROOT/$BACKGROUND"
REF_CODE="$MERGED_ROOT/${BACKGROUND}-ct-code"
CELL_MATH="$HYBRID_ROOT/merged/${LAYER}_${METHOD}-ct-math"
CELL_CODE="$HYBRID_ROOT/merged/${LAYER}_${METHOD}-ct-code"
RESULTS_BASE="artifacts/results-simpler-olmo-exps/Olmo-3-7b/group-hybrid/merged/${LAYER}_${METHOD}"

OLMES_MODEL_ARGS='{"gpu_memory_utilization": 0.8, "trust_remote_code": false, "max_length": 4096}'
CODE_TASKS=("codex_humaneval::tulu" "codex_humanevalplus::tulu" "ifeval::tulu")
MATH_TASKS=("minerva_math_500::tulu" "gsm8k::tulu")

echo ">>> cell: layer=$LAYER method=$METHOD (bg=$BACKGROUND)"

assemble() {  # $1=ref-merge  $2=out-dir
  python scripts/analysis/assemble_cell.py \
    --experts-root "$EXPERTS_ROOT" \
    --layer-type "$LAYER" --method "$METHOD" --background "$BACKGROUND" \
    --ref-merge "$1" --out-dir "$2"
}

# 1. Assemble both per-template views (symlinks + index + config); idempotent.
assemble "$REF_CODE" "$CELL_CODE"
assemble "$REF_MATH" "$CELL_MATH"

# 2. Code/IF tasks on the Code-template view (one model load).
olmes --model "$CELL_CODE" --task "${CODE_TASKS[@]}" \
  --output-dir "$RESULTS_BASE/ct-code" \
  --gpus 1 --model-type vllm --model-args "$OLMES_MODEL_ARGS" \
  --batch-size 64 --num-workers 1

# 3. gsm8k on the Math-template view (minerva already lives in ct-math/).
olmes --model "$CELL_MATH" --task "${MATH_TASKS[@]}" \
  --output-dir "$RESULTS_BASE/ct-math" \
  --gpus 1 --model-type vllm --model-args "$OLMES_MODEL_ARGS" \
  --batch-size 64 --num-workers 1
