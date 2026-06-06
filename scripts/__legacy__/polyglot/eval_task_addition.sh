#!/bin/bash
#SBATCH --job-name=polyglot_merge_eval
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=artifacts/logs/%x_%j.out
#SBATCH --error=artifacts/logs/%x_%j.err
# Merge the 4 Polyglot OLMo3-7B language experts (ar, cs, de, es) into one
# multilingual model with each method, then evaluate with lm-eval-harness
# (Global-MMLU-Lite 0-shot + MGSM 5-shot greedy). All experts share base
# allenai/Olmo-3-1025-7B, so merged models are standard HF checkpoints that
# vLLM/lm-eval load directly. actmat/mean/tsv are data-free (no covariance.py).
#
# Runs fine via sbatch OR directly on an interactive GPU node (just `bash` it).
# Prereq: bash scripts/polyglot/download_models.sh
#
# Usage:
#   sbatch scripts/polyglot/eval_task_addition.sh
#   METHODS="actmat" EVAL_EXPERTS=1 bash scripts/polyglot/eval_task_addition.sh
set -euo pipefail
mkdir -p artifacts/logs

source "$SCRATCH/actmat/.venv-pg/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export HF_HOME=$SCRATCH/huggingface
export SSL_CERT_DIR=/etc/ssl/certs
# Reduce CUDA fragmentation (eval OOM'd with ~3.8GB reserved-but-unallocated).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source scripts/polyglot/lib_lmeval.sh

# ── CONFIG ──────────────────────────────────────────────────────────────────
MODEL="Olmo-3-7b-polyglot"
BASE="artifacts/checkpoints/${MODEL}"
LANGS=(ar de es)   # cs dropped: not in lm-eval MGSM or Global-MMLU. ar+de+es have
                   # Global-MMLU; de+es also have MGSM. All merged langs are evaluable.
read -r -a METHODS <<< "${METHODS:-mean tsv actmat}"   # data-free methods only
EVAL_EXPERTS="${EVAL_EXPERTS:-0}"                       # 1 => also eval base + experts
RESULTS_ROOT="artifacts/results-polyglot"

# Reference checkpoints come from the HF hub (local pretrained/finetuned dirs are
# in our param-folder export format, which vLLM can't load; merged models are
# saved via save_pretrained and load fine).
PRETRAINED_ID="allenai/Olmo-3-1025-7B"
FT_PREFIX="ljvmiranda921/Polyglot-OLMo3-7B-SFT"

# ── Optional reference evals: base + each individual expert (from HF hub) ─────
if [[ "$EVAL_EXPERTS" == "1" ]]; then
  run_lmeval "$PRETRAINED_ID" "${RESULTS_ROOT}/${MODEL}-base"
  for lang in "${LANGS[@]}"; do
    run_lmeval "${FT_PREFIX}-${lang}" "${RESULTS_ROOT}/${MODEL}-expert-${lang}"
  done
fi

# ── Merge + evaluate each method ─────────────────────────────────────────────
for method in "${METHODS[@]}"; do
  MERGED_DIR="${BASE}/merged-${method}"
  RESULTS_DIR="${RESULTS_ROOT}/${MODEL}-${method}"

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

  run_lmeval "$MERGED_DIR" "$RESULTS_DIR"
done

echo ">>> All methods done. Results under ${RESULTS_ROOT}/"
