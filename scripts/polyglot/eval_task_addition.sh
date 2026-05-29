#!/bin/bash
#SBATCH --job-name=polyglot_merge_eval
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=artifacts/logs/%x_%j.out
#SBATCH --error=artifacts/logs/%x_%j.err
# Merge the 4 Polyglot OLMo3-7B language experts (ar, cs, de, es) into a single
# multilingual model with each method, then evaluate with Lighteval (same setup
# as the paper). Mirrors scripts/olmo/eval_task_addition.sh.
#
# All experts share base allenai/Olmo-3-1025-7B and one chat template, so no
# per-task chat-template view routing is needed (unlike the OLMo Math/Code/IF
# pipeline). actmat/mean/tsv are data-free — no covariance.py step.
#
# Prerequisites: bash scripts/polyglot/download_models.sh
#
# Usage:
#   sbatch scripts/polyglot/eval_task_addition.sh
#   METHODS="actmat" sbatch scripts/polyglot/eval_task_addition.sh
set -euo pipefail
mkdir -p artifacts/logs

source "$SCRATCH/actmat/.venv-polyglot/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export HF_HOME=$SCRATCH/huggingface
export SSL_CERT_DIR=/etc/ssl/certs

source scripts/polyglot/lib_lighteval.sh

# ── CONFIG ──────────────────────────────────────────────────────────────────
MODEL="Olmo-3-7b-polyglot"
BASE="artifacts/checkpoints/${MODEL}"
LANGS=(ar cs de es)
# Space-separated; override via env. Data-free methods only (no covariance).
read -r -a METHODS <<< "${METHODS:-mean tsv actmat}"
# Set EVAL_EXPERTS=1 to also eval each individual expert + base for reference.
# NOTE: the local pretrained/finetuned dirs are in our param-folder export
# format (hashed params/ + manifest) which vLLM cannot load. Reference evals
# therefore pull the standard HF-hub checkpoints instead; only the *merged*
# models (saved via save_pretrained) are evaluated from local disk.
EVAL_EXPERTS="${EVAL_EXPERTS:-0}"
PRETRAINED_ID="allenai/Olmo-3-1025-7B"
FT_PREFIX="ljvmiranda921/Polyglot-OLMo3-7B-SFT"

# ── Optional reference evals: base + each individual expert (from HF hub) ─────
if [[ "$EVAL_EXPERTS" == "1" ]]; then
  run_lighteval "$PRETRAINED_ID" "artifacts/results-polyglot/${MODEL}-base"
  for lang in "${LANGS[@]}"; do
    run_lighteval "${FT_PREFIX}-${lang}" "artifacts/results-polyglot/${MODEL}-expert-${lang}"
  done
fi

# ── Merge + evaluate each method ─────────────────────────────────────────────
for method in "${METHODS[@]}"; do
  MERGED_DIR="${BASE}/merged-${method}"
  RESULTS_DIR="artifacts/results-polyglot/${MODEL}-${method}"

  echo "============================================================"
  echo "Method: ${method} | merged: ${MERGED_DIR} | results: ${RESULTS_DIR}"
  echo "============================================================"

  if [[ -d "$MERGED_DIR" ]]; then
    echo ">>> Skipping merge: ${MERGED_DIR} already exists"
  else
    python scripts/olmo/merge.py \
      --save "$BASE" \
      --merge-tasks "${LANGS[@]}" \
      --merge-func "$method" \
      --output-dir "$MERGED_DIR"
  fi

  run_lighteval "$MERGED_DIR" "$RESULTS_DIR"
done

echo ">>> All methods done. Results under artifacts/results-polyglot/"
