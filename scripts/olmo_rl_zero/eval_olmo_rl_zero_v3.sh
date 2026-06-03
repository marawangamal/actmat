#!/bin/bash
#SBATCH --job-name=hf_eval_olmo_v3
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err
# Variant of eval_olmo_v2.sh that does NOT merge the vocab layers with the chosen
# method: lm_head.weight and model.embed_tokens.weight are instead averaged
# (--ignore-mean 'lm_head|embed_tokens'). Everything else (experts, tasks,
# two-view chat-template structure, gsm8k+minerva / code+ifeval) is identical to
# v2. Writes to its OWN merged-checkpoint and results dirs so it never reuses the
# v2 full-method merges (the merge-skip below keys off MERGED_DIR existing).
#
# To confirm the override fired, grep the merge logs for the marker that
# src/hf/merge.py prints per affected layer:
#   grep "\[IGNORE-MEAN\]" artifacts/logs/hf_eval_olmo_v3_*.out
#
# Submit with: sbatch --array=0-$((N-1)) scripts/olmo_rl_zero/eval_olmo_rl_zero_v3.sh
set -euo pipefail

source "$SCRATCH/actmat/.venv-olmo/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export HF_HOME="$SCRATCH/huggingface"
export SSL_CERT_DIR=/etc/ssl/certs

BASE_MODEL="allenai/Olmo-3-1025-7B"
MATH_EXPERT="allenai/Olmo-3-7B-RL-Zero-Math"
CODE_EXPERT="allenai/Olmo-3-7B-RL-Zero-Code"
IF_EXPERT="allenai/Olmo-3-7B-RL-Zero-IF"
METHODS=(sum mean actmat tsv isoc actmat_herm regmean wudi actmat_gd actmat_herm_10ki actmat_gd_10ki)
METHOD="${METHODS[$SLURM_ARRAY_TASK_ID]}"
# Own group (group-rl-zero-headmean) so v3 builds fresh head-mean merges instead
# of reusing v2's full-method ones (group-rl-zero).
MERGED_DIR="artifacts/checkpoints/Olmo-3-7b/group-rl-zero-headmean/merged/${METHOD}"
RESULTS_BASE="artifacts/results/Olmo-3-7b/group-rl-zero-headmean/merged/${METHOD}"

# Layers to mean-merge instead of method-merge (vocab head + input embeddings).
IGNORE_MEAN_RE='lm_head|embed_tokens'

# Task groups, split by which expert's chat template they need (Code and IF
# share a template; gsm8k uses the Math one).
CODE_TASKS=(
  "codex_humaneval::tulu"
  "codex_humanevalplus::tulu"
  "ifeval::tulu"
)
MATH_TASKS=(
  "gsm8k::tulu"
  "minerva_math_500::tulu"
)
OLMES_MODEL_ARGS='{"gpu_memory_utilization": 0.8, "trust_remote_code": false, "max_length": 4096}'

# Materialize a view of MERGED_DIR that uses the given expert's tokenizer (chat
# template included). Big weight files are symlinked (read-only, never written);
# everything else is copied so it's a real, independent file. save_pretrained
# then overwrites the tokenizer copies in place — safe, since nothing it writes
# is a symlink back to the canonical merge. Experts share the base vocab, so
# only the chat template differs.
make_view() {
  local view="$1" expert="$2"
  mkdir -p "$view"
  for f in "$MERGED_DIR"/*; do
    b="$(basename "$f")"
    case "$b" in
      *.safetensors | *.pt | *.bin) ln -sfn "$(realpath "$f")" "$view/$b" ;;
      *) cp "$f" "$view/$b" ;;
    esac
  done
  python - "$expert" "$view" <<'PY'
import sys
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained(sys.argv[1]).save_pretrained(sys.argv[2])
PY
}

# 1. Merge (skip if the merged checkpoint already exists — rm -rf "$MERGED_DIR"
# to force a re-merge). The vocab layers are forced to a plain mean via
# --ignore-mean; merge.py prints a [IGNORE-MEAN] line for each affected layer.
if [[ -d "$MERGED_DIR" ]]; then
  echo ">>> Skipping merge: $MERGED_DIR already exists"
else
  python src/hf/merge.py \
    --base-model-name-or-path "$BASE_MODEL" \
    --chat-template-name-or-path "$MATH_EXPERT" \
    --expert-model-names-or-paths "$MATH_EXPERT" "$CODE_EXPERT" "$IF_EXPERT" \
    --merge-method "$METHOD" \
    --ignore-mean "$IGNORE_MEAN_RE" \
    --output-dir "$MERGED_DIR"
fi

# 2. Materialize per-template views
make_view "${MERGED_DIR}-ct-code" "$CODE_EXPERT"
make_view "${MERGED_DIR}-ct-math" "$MATH_EXPERT"

# 3. Evaluate each task group against its matching view
olmes --model "${MERGED_DIR}-ct-code" --task "${CODE_TASKS[@]}" \
  --output-dir "${RESULTS_BASE}/ct-code" \
  --gpus 1 --model-type vllm --model-args "$OLMES_MODEL_ARGS" \
  --batch-size 64 --num-workers 1

olmes --model "${MERGED_DIR}-ct-math" --task "${MATH_TASKS[@]}" \
  --output-dir "${RESULTS_BASE}/ct-math" \
  --gpus 1 --model-type vllm --model-args "$OLMES_MODEL_ARGS" \
  --batch-size 64 --num-workers 1
