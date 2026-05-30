#!/bin/bash
#SBATCH --job-name=polyglot_all_mgsmcot
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --array=0-8
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err
#
# CoT MGSM (the proper math protocol) for the polyglot-all models, via lm-eval
# mgsm_native_cot_{de,es} (5-shot, greedy) — same runner as scripts/polyglot.
# number-only lighteval MGSM floors scores (~0.1) and unfairly penalizes the
# chat-tuned experts; native-CoT is ~0.5 and is the conventional MGSM setup.
# mmlu/mrb still come from the lighteval runs — this only adds the math column.
#
# Usage:
#   sbatch scripts/polyglot-all/eval_mgsm_cot.sh
#   ITEM=merge-actmat_gd bash scripts/polyglot-all/eval_mgsm_cot.sh   # single, no array
set -euo pipefail

ACTMAT="$SCRATCH/actmat"
cd "$ACTMAT"
mkdir -p artifacts/logs

sleep $(( ${SLURM_ARRAY_TASK_ID:-0} * 30 ))   # stagger shared-venv cold imports

source "$ACTMAT/.venv-pg/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export HF_HOME=$SCRATCH/huggingface
export SSL_CERT_DIR=/etc/ssl/certs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CK="artifacts/checkpoints/Olmo-3-7b-polyglot-all"
# name : model-path-or-hf-id
ITEMS=(
  "base:allenai/Olmo-3-1025-7B"
  "merge-mean:${CK}/merged-mean"
  "merge-tsv:${CK}/merged-tsv"
  "merge-actmat:${CK}/merged-actmat"
  "merge-isoc:${CK}/merged-isoc"
  "merge-wudi:${CK}/merged-wudi"
  "merge-actmat_gd:${CK}/merged-actmat_gd"
  "expert-de:ljvmiranda921/Polyglot-OLMo3-7B-SFT-de"
  "expert-es:ljvmiranda921/Polyglot-OLMo3-7B-SFT-es"
)

# Pick item: array index if set, else $ITEM env.
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  ENTRY="${ITEMS[$SLURM_ARRAY_TASK_ID]}"
else
  ENTRY=""
  for e in "${ITEMS[@]}"; do [[ "${e%%:*}" == "${ITEM:-}" ]] && ENTRY="$e"; done
  [[ -z "$ENTRY" ]] && { echo "set ITEM to one of: ${ITEMS[*]%%:*}"; exit 1; }
fi
NAME="${ENTRY%%:*}"; MODEL="${ENTRY#*:}"
OUT="artifacts/results-polyglot-all-mgsmcot/${NAME}"

# Local merged dirs may not exist yet (e.g. actmat_gd still merging) — skip cleanly.
if [[ "$MODEL" == artifacts/* && ! -d "$MODEL" ]]; then
  echo ">>> skip ${NAME}: ${MODEL} does not exist yet"; exit 0
fi

export RUN_GMMLU=0 RUN_MGSM=1          # CoT MGSM only; mmlu/mrb come from lighteval
source scripts/polyglot/lib_lmeval.sh
run_lmeval "$MODEL" "$OUT"
echo ">>> done: ${OUT}"
