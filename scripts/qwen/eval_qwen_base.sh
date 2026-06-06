#!/bin/bash
#SBATCH --job-name=hf_eval_qwen_base
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=artifacts/logs/%x_%A.out
#SBATCH --error=artifacts/logs/%x_%A.err
# Sanity-check eval of the Qwen2.5-1.5B BASE model (no merge) on the base-model
# protocol: few-shot in-context completion with NO chat template (::olmes /
# ::none recipes), matching how Qwen2.5 (§5.1) and AI2's OlmoBaseEval evaluate
# base models. Establishes the reference numbers the merges should be compared
# against, and confirms the no-chat recipe gives sane scores (vs the garbage the
# ::tulu chat template produced on these base models).
#
# Submit with: sbatch scripts/qwen/eval_qwen_base.sh
set -euo pipefail

source "$SCRATCH/actmat/.venv-olmo/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export HF_HOME="$SCRATCH/huggingface"
export SSL_CERT_DIR=/etc/ssl/certs

MODEL="Qwen/Qwen2.5-1.5B"
RESULTS_DIR="artifacts/results/Qwen2.5-1.5B/group-main/pretrained"

# Base-model (non-chat) eval recipes: gsm8k/MATH-500 are 8/4-shot greedy
# exact_match; humaneval uses the starcoder_pass@1 recipe (0-shot raw-completion
# generative pass@1, temp 0.2) — the standard base-code-model setup. (NB: the
# ::none variant uses a bits-per-byte prompt_variant=inloop_bpb and is NOT a
# clean generative pass@1 — it scored ~5% on the base model.)
TASKS=(
  "gsm8k::olmes"
  "minerva_math_500::olmes"
  "codex_humaneval::starcoder_pass@1"
)
OLMES_MODEL_ARGS='{"gpu_memory_utilization": 0.8, "trust_remote_code": false, "max_length": 4096}'

olmes --model "$MODEL" --task "${TASKS[@]}" \
  --output-dir "$RESULTS_DIR" \
  --gpus 1 --model-type vllm --model-args "$OLMES_MODEL_ARGS" \
  --batch-size 64 --num-workers 1
