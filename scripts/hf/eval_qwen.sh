#!/bin/bash
#SBATCH --job-name=hf_eval_qwen
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err
# Merge Qwen2.5-Coder-1.5B + Qwen2.5-Math-1.5B (base: Qwen2.5-1.5B), then eval
# math (gsm8k + MATH-500) + code (humaneval) with the OLMES harness.
# Mirrors scripts/hf/eval_olmo_v2.sh: same merge + two-view (Code vs Math chat
# template) structure; tasks use the ::tulu (chat-templated) variants.
#
# Submit with: sbatch --array=0-$((N-1)) scripts/hf/eval_qwen.sh
set -euo pipefail

source "$SCRATCH/actmat/.venv-olmo/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export HF_HOME="$SCRATCH/huggingface"
export SSL_CERT_DIR=/etc/ssl/certs

BASE_MODEL="Qwen/Qwen2.5-1.5B"
MATH_EXPERT="Qwen/Qwen2.5-Math-1.5B"
CODE_EXPERT="Qwen/Qwen2.5-Coder-1.5B"
METHODS=(mean tsv actmat_herm_10ki)
METHOD="${METHODS[$SLURM_ARRAY_TASK_ID]}"
MERGED_DIR="artifacts/checkpoints/Qwen2.5-1.5B/group-main/merged/${METHOD}"
RESULTS_BASE="artifacts/results/Qwen2.5-1.5B/group-main/merged/${METHOD}"

# Task groups, split by which expert's chat template they need.
CODE_TASKS=(
  "codex_humaneval::tulu"
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

# 1. Merge (skip if the merged checkpoint already exists; rm -rf "$MERGED_DIR"
# to force a re-merge). Chat template seeded from the math expert.
if [[ -d "$MERGED_DIR" ]]; then
  echo ">>> Skipping merge: $MERGED_DIR already exists"
else
  python src/hf/merge.py \
    --base-model-name-or-path "$BASE_MODEL" \
    --chat-template-name-or-path "$MATH_EXPERT" \
    --expert-model-names-or-paths "$MATH_EXPERT" "$CODE_EXPERT" \
    --merge-method "$METHOD" \
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
