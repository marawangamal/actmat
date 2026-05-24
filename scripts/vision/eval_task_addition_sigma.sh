#!/bin/bash
#SBATCH --job-name=eval_vision_sigma
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err

set -euo pipefail
mkdir -p artifacts/logs

# 0. Setup environment
source "$SCRATCH/actmat/.venv-vl/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

DATA_DIR="data/vision"
OPENCLIP_DIR="$SCRATCH/openclip"

if [ ! -d "$SLURM_TMPDIR/data" ]; then
  cp downloads/data.tar.gz "$SLURM_TMPDIR/"
  tar -xzf "$SLURM_TMPDIR/data.tar.gz" -C "$SLURM_TMPDIR/"
fi
ln -sfn "$SLURM_TMPDIR/data" data

# ===== Sigma sweep =====
# Sweep std of per-task coefficient α_t ~ N(1.0, sigma) applied before merging.
# Results land in artifacts/results-sig{sigma}/{model}-{method}/metrics.json so
# the plotting code can read (sigma, method, model) -> mean perf directly.
MODELS=(ViT-B-16)
METHODS=(mean tsv actmat wudi ace)
FT_MODES=(standard)
SIGMAS=(0.1 0.2 0.5 1.0 2.0)
MERGE_MODE=d
HPO=''

# Build flat list of (sigma, ft_mode, model, method) combos so each array task
# runs exactly one configuration. Total combos = |SIGMAS|*|FT_MODES|*|MODELS|*|METHODS|.
COMBOS=()
for SIGMA in "${SIGMAS[@]}"; do
  for FT_MODE in "${FT_MODES[@]}"; do
    for MODEL in "${MODELS[@]}"; do
      for METHOD in "${METHODS[@]}"; do
        COMBOS+=("$SIGMA|$FT_MODE|$MODEL|$METHOD")
      done
    done
  done
done

NUM_COMBOS=${#COMBOS[@]}

# When run without SLURM (or interactively), iterate all combos sequentially.
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  echo "[BASH] No SLURM_ARRAY_TASK_ID set — running all $NUM_COMBOS combos sequentially."
  INDICES=( $(seq 0 $((NUM_COMBOS - 1))) )
else
  INDICES=("$SLURM_ARRAY_TASK_ID")
fi

for IDX in "${INDICES[@]}"; do
  IFS='|' read -r SIGMA FT_MODE MODEL METHOD <<< "${COMBOS[$IDX]}"

  RESULTS_DIR="artifacts/results-sig${SIGMA}"

  # Run covariance/fisher script if needed (sigma-independent — only once per model/ft mode/method).
  if [ "$METHOD" = "regmean" ]; then
    echo "[BASH] Running covariance.py | model: $MODEL | ft mode: $FT_MODE | method: $METHOD"
    python scripts/vision/covariance.py \
      --model="$MODEL" \
      --finetuning-mode="$FT_MODE" \
      --mha=split
  elif [ "$METHOD" = "fisher" ]; then
    echo "[BASH] Running fisher.py | model: $MODEL | ft mode: $FT_MODE | method: $METHOD"
    python scripts/vision/fisher.py \
      --model="$MODEL" \
      --finetuning-mode="$FT_MODE" \
      --mha=split
  fi

  echo "[BASH] [array=$IDX/$NUM_COMBOS] eval_task_addition.py | model: $MODEL | ft: $FT_MODE | method: $METHOD | mode: $MERGE_MODE | sigma: $SIGMA"
  python scripts/vision/eval_task_addition.py \
    --model="$MODEL" \
    --finetuning-mode="$FT_MODE" \
    --data-location="$DATA_DIR" \
    --merge-func="$METHOD" \
    --merge-mode="$MERGE_MODE" \
    --mha=split \
    --sigma="$SIGMA" \
    --results-dir="$RESULTS_DIR" \
    ${HPO:+--hpo="$HPO"}
done
