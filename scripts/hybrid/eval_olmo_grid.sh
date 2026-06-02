#!/bin/bash
#SBATCH --job-name=hf_eval_olmo_grid
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err
# Layer x method hybrid grid (docs/experiments.md). Each array task = one cell:
# merge one layer type with one method, everything else with mean, then eval on
# minerva. Cells are assembled on the fly from the per-layer expert splits
# (scripts/hybrid/split_expert.py must have been run for mean + every METHOD first).
#
# Grid = LAYER_TYPES x METHODS, flattened row-major (index = li*nMethods + mi).
# Submit with: sbatch --array=0-$((7*3-1)) scripts/hybrid/eval_olmo_grid.sh
set -euo pipefail

source "$SCRATCH/actmat/.venv-olmo/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export HF_HOME="$SCRATCH/huggingface"
export SSL_CERT_DIR=/etc/ssl/certs

LAYER_TYPES=(q_proj k_proj v_proj o_proj gate_proj up_proj down_proj)
METHODS=(isoc tsv actmat_herm_10ki)
BACKGROUND="mean"

N_METHODS=${#METHODS[@]}
LI=$(( SLURM_ARRAY_TASK_ID / N_METHODS ))
MI=$(( SLURM_ARRAY_TASK_ID % N_METHODS ))
LAYER="${LAYER_TYPES[$LI]}"
METHOD="${METHODS[$MI]}"

HYBRID_ROOT="artifacts/checkpoints/Olmo-3-7b/group-hybrid"
EXPERTS_ROOT="$HYBRID_ROOT/experts"
REF_MERGE="artifacts/checkpoints/Olmo-3-7b/group-rl-zero/merged/$BACKGROUND"
CELL_DIR="$HYBRID_ROOT/merged/${LAYER}_${METHOD}-ct-math"
RESULTS_BASE="artifacts/results/Olmo-3-7b/group-hybrid/merged/${LAYER}_${METHOD}"

OLMES_MODEL_ARGS='{"gpu_memory_utilization": 0.8, "trust_remote_code": false, "max_length": 4096}'
MATH_TASKS=("minerva_math_500::tulu")

echo ">>> cell: layer=$LAYER method=$METHOD (bg=$BACKGROUND)"

# 1. Assemble the cell (symlinks + index + config); idempotent.
python scripts/hybrid/assemble_cell.py \
  --experts-root "$EXPERTS_ROOT" \
  --layer-type "$LAYER" --method "$METHOD" --background "$BACKGROUND" \
  --ref-merge "$REF_MERGE" \
  --out-dir "$CELL_DIR"

# 2. Eval on minerva.
olmes --model "$CELL_DIR" --task "${MATH_TASKS[@]}" \
  --output-dir "$RESULTS_BASE/ct-math" \
  --gpus 1 --model-type vllm --model-args "$OLMES_MODEL_ARGS" \
  --batch-size 64 --num-workers 1
