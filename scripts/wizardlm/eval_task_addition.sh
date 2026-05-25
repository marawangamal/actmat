#!/bin/bash
#SBATCH --job-name=eval_wizardlm
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err
# Merge + evaluate WizardLM-13B / WizardMath-13B / llama-2-13b-code-alpaca via
# olmes, then collect results. Reproduces the DARE paper Fig. 1 (right).
#
# Each array task runs one merge method, so methods run in parallel.
#
# Prerequisites:
#   - bash scripts/wizardlm/download_models.sh (HF_TOKEN required for Llama-2)
#   - OPENAI_API_KEY exported for alpaca_eval_v2 (GPT-4 judge)
#
# Usage:
#   sbatch --array=0-2 scripts/wizardlm/eval_task_addition.sh
set -euo pipefail
mkdir -p artifacts/logs

# 0. Setup environment
source "$SCRATCH/actmat/.venv-olmo/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export HF_HOME="$SCRATCH/huggingface"
export SSL_CERT_DIR=/etc/ssl/certs

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL="wizardlm"
METHODS=(tsv actmat wudi actmat_mons)
# Submit with: sbatch --array=0-$((${#METHODS[@]}-1)) scripts/wizardlm/eval_task_addition.sh
# Each array task runs one method end-to-end (merge + eval).

# Per-method extra merge kwargs (JSON). Empty string for none.
declare -A MERGE_KWARGS=(
  ["dare"]='{"drop_rate": 0.5, "seed": 0, "base_merge": "sum"}'
)

# Note: `actmat` in src/merging.py is the data-free variant (c = d^T @ d),
# so it does NOT need pre-collected activation covariances. Only regmean needs them.
STATS_METHODS=(regmean)

# Embedding/lm_head shapes differ across experts (32000 vs 32001 vocab from
# pad-token additions). Ignore those keys at merge time — they fall back to
# pretrained values via the ignore_keys path in src/merging.py.
IGNORE_KEYS=(embed_tokens lm_head)

# ── OLMES ─────────────────────────────────────────────────────────────────────
# DARE paper (arXiv:2311.03099) eval setup, following WizardCoder (Luo 2023b):
#   • T=0 greedy, max_gen_toks=1024 (GSM8K) / 2048 (others), pass@1.
#   • MBPP prompt = bigcode-evaluation-harness format: NL description + first
#     test assert wrapped in '"""..."""' so the model picks up the exact
#     function signature. (The default "inloop_bpb" variant strips the assert,
#     so the model invents its own function name and all tests fail — that's
#     what gave us ~3% pass@1 on the first run.)
#   • HumanEval = bare-prompt bigcode style (no chat), greedy pass@1.
# alpaca_eval_v2 dropped — requires OPENAI_API_KEY for the GPT-4 judge.
OLMES_TASKS=(
  '{"task_name": "gsm8k::tulu", "generation_kwargs": {"max_gen_toks": 1024}}'
  '{"task_name": "minerva_math_500::tulu", "generation_kwargs": {"max_gen_toks": 1024}}'
  '{"task_name": "codex_humaneval::starcoder_pass@1", "generation_kwargs": {"max_gen_toks": 2048, "do_sample": false, "temperature": 0.0, "repeats": 1}}'
  '{"task_name": "codex_humanevalplus::none", "generation_kwargs": {"max_gen_toks": 2048, "do_sample": false, "temperature": 0.0, "repeats": 1}, "metric_kwargs": {"pass_at_ks": [1]}}'
  '{"task_name": "mbpp:3shot::none", "context_kwargs": {"prompt_variant": "bcharness"}, "generation_kwargs": {"max_gen_toks": 2048, "do_sample": false, "temperature": 0.0, "repeats": 1}}'
  # '{"task_name": "alpaca_eval_v2::tulu", "generation_kwargs": {"max_gen_toks": 2048}}'
)
OLMES_MODEL_ARGS='{"gpu_memory_utilization": 0.8, "trust_remote_code": false, "max_length": 4096, "chat_template": "llama2"}'
GPUS=2
BATCH_SIZE=32
NUM_WORKERS=1

# ── Select method for this array task ───────────────────────────────────────
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: this script must be submitted as a job array. Use:"
  echo "  sbatch --array=0-$((${#METHODS[@]}-1)) scripts/wizardlm/eval_task_addition.sh"
  exit 1
fi
if (( SLURM_ARRAY_TASK_ID >= ${#METHODS[@]} )); then
  echo "ERROR: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} out of range for METHODS (${#METHODS[@]} entries)"
  exit 1
fi
method="${METHODS[$SLURM_ARRAY_TASK_ID]}"

# ── Statistics collection (regmean only — actmat variants are data-free) ───
need_stats=0
for sm in "${STATS_METHODS[@]}"; do
  if [[ "$method" == "$sm" ]]; then need_stats=1; fi
done
if [[ $need_stats -eq 1 ]]; then
  echo ">>> Collecting covariances for ${MODEL}"
  python scripts/wizardlm/covariance.py \
    --capability all \
    --save "artifacts/checkpoints/${MODEL}"
fi

# ── Merge + Evaluate ────────────────────────────────────────────────────────
MERGED_DIR="artifacts/checkpoints/${MODEL}/${method}"
RESULTS_DIR="artifacts/results/${MODEL}-${method}"

echo "============================================================"
echo "Array task: ${SLURM_ARRAY_TASK_ID} / Method: ${method}"
echo "Merged:  ${MERGED_DIR}"
echo "Results: ${RESULTS_DIR}"
echo "============================================================"

# 1. Merge (skip if already done)
if [[ -d "$MERGED_DIR" ]]; then
  echo ">>> Skipping merge: ${MERGED_DIR} already exists"
else
  extra_args=()
  if [[ -n "${MERGE_KWARGS[$method]:-}" ]]; then
    extra_args+=(--merge-kwargs "${MERGE_KWARGS[$method]}")
  fi
  python scripts/wizardlm/merge.py \
    --save "artifacts/checkpoints/${MODEL}" \
    --merge-func "$method" \
    --output-dir "$MERGED_DIR" \
    --ignore-keys "${IGNORE_KEYS[@]}" \
    "${extra_args[@]}"
fi

# 2. Evaluate (skip if metrics.json already exists)
if [[ -f "$RESULTS_DIR/metrics.json" ]]; then
  echo ">>> Skipping eval: ${RESULTS_DIR}/metrics.json already exists"
else
  echo ">>> Evaluating: Batch size = $BATCH_SIZE, Workers = $NUM_WORKERS, GPUs = $GPUS"
  echo ">>> Model: $MERGED_DIR, tasks: ${OLMES_TASKS[@]}"
  olmes \
    --model "$MERGED_DIR" \
    --task "${OLMES_TASKS[@]}" \
    --output-dir "$RESULTS_DIR" \
    --gpus "$GPUS" \
    --model-type vllm \
    --model-args "$OLMES_MODEL_ARGS" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS"
fi
