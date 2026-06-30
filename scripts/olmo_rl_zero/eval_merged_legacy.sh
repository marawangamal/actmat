#!/bin/bash
#SBATCH --job-name=eval_olmo_rl_zero_legacy
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err
# Merge Olmo-3-7B RL-Zero experts (Math + Code + IF) onto the base, then eval
# with olmes. Tasks mirror the current RL-Zero evaluation set (HumanEval(+),
# IFEval, AIME). Experts share the base's vocab, so no embed/lm_head masking.
#
# Chat template: the merge bakes in ONE template, but each olmes task group needs
# its matching expert's (Code/IF prompts differ from Math). So after merging once
# we materialize two lightweight VIEW dirs that symlink the merged weights and
# only override chat_template.jinja:
#   merged/${METHOD}-ct-code  -> Code/IF tasks
#   merged/${METHOD}-ct-math  -> AIME (math) tasks
# Weights are never duplicated; only chat_template.jinja differs per view.
#
# Submit with: sbatch --array=0-$((N-1)) scripts/olmo_rl_zero/eval_merged_legacy.sh
set -euo pipefail

source "$SCRATCH/actmat/.venv-olmo/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export HF_HOME="$SCRATCH/huggingface"
export SSL_CERT_DIR=/etc/ssl/certs

BASE_MODEL="allenai/Olmo-3-1025-7B"
MATH_EXPERT="allenai/Olmo-3-7B-RL-Zero-Math"
CODE_EXPERT="allenai/Olmo-3-7B-RL-Zero-Code"
IF_EXPERT="allenai/Olmo-3-7B-RL-Zero-IF"
METHODS=(sum mean actmat tsv)
NUM_METHODS="${#METHODS[@]}"
if [ "$SLURM_ARRAY_TASK_ID" -ge "$NUM_METHODS" ]; then
  echo "No method for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
  exit 0
fi
METHOD="${METHODS[$SLURM_ARRAY_TASK_ID]}"
# RL-Zero experts live under the group-rl-zero path level (polyglot is group-polyglot).
MERGED_DIR="artifacts/checkpoints/Olmo-3-7b/group-rl-zero/merged/${METHOD}"
RESULTS_BASE="artifacts/results/Olmo-3-7b/group-rl-zero/merged/${METHOD}"
EXPERT_STATS_DIR="artifacts/checkpoints/Olmo-3-7b/group-rl-zero/experts"

# Task groups, split by which expert's chat template they need (Code and IF
# share a template; AIME uses the Math one).
CODE_TASKS=(
  "codex_humaneval::tulu"
  "codex_humanevalplus::tulu"
  "ifeval::tulu"
)
MATH_TASKS=(
  "aime:zs_cot_r1::pass_at_32_2024_deepseek"
  "aime:zs_cot_r1::pass_at_32_2025_deepseek"
)
OLMES_MODEL_ARGS='{"gpu_memory_utilization": 0.8, "trust_remote_code": false, "max_length": 16384}'

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

# 1. Merge (skip if the merged checkpoint already exists — e.g. from a prior run
# or the migrated scripts/olmo outputs; rm -rf "$MERGED_DIR" to force a re-merge)
if [[ -f "$MERGED_DIR/model.safetensors.index.json" ]]; then
  echo ">>> Skipping merge: $MERGED_DIR already exists"
else
  python src/hf2/merge.py \
    --base-model-name-or-path "$BASE_MODEL" \
    --chat-template-name-or-path "$MATH_EXPERT" \
    --expert-model-names-or-paths "$MATH_EXPERT" "$CODE_EXPERT" "$IF_EXPERT" \
    --merge-method "$METHOD" \
    --expert-stats-dir "$EXPERT_STATS_DIR" \
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
