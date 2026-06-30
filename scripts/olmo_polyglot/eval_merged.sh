#!/bin/bash
#SBATCH --job-name=eval_olmo_polyglot
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err
# Merge the 4 multilingual experts ljvmiranda921/Polyglot-OLMo3-7B-SFT-{ar,cs,de,es}
# (base: Olmo-3-1025-7B), then eval. The two harnesses need conflicting venvs, so
# we switch: lm-eval in .venv-pg-mgsm, lighteval fork in .venv-pg-mmlu-mrb.
#
# Eval settings follow Polyglot Teachers (arXiv 2604.11290) Table 10:
#   Lib        Benchmark          Formulation  Metric        N-shot
#   lighteval  Global-MMLU Lite   MCF          accuracy      0
#   lighteval  M-RewardBench      MCF          weighted_acc  0
#   lm_eval    M-GSM              generative   exact-match   5
# Note: we use lm_eval mgsm_native_cot, where the few-shot examples are reasoning
# chains rather than a single number.
#
# Submit with: sbatch --array=0-6 scripts/olmo_polyglot/eval_merged.sh
set -euo pipefail

BASE_MODEL="allenai/Olmo-3-1025-7B"
EXPERTS=(ljvmiranda921/Polyglot-OLMo3-7B-SFT-{ar,cs,de,es})
METHODS=(sum mean actmat tsv isoc wudi actmat_gd)
NUM_METHODS="${#METHODS[@]}"
if [ "$SLURM_ARRAY_TASK_ID" -ge "$NUM_METHODS" ]; then
  echo "No method for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
  exit 0
fi
METHOD="${METHODS[$SLURM_ARRAY_TASK_ID]}"
TASKS_LMEVAL="mgsm_native_cot_de,mgsm_native_cot_es"
TASKS_LIGHTEVAL="global_mmlu_lite:ar,global_mmlu_lite:de,global_mmlu_lite:es,mrewardbench_mcf:ar,mrewardbench_mcf:cs,mrewardbench_mcf:de,mrewardbench_mcf:es"
# Polyglot experts live under group-polyglot of the shared Olmo-3-7b model dir
# (RL-Zero is group-rl-zero).
MERGED_DIR="artifacts/checkpoints/Olmo-3-7b/group-polyglot/merged/${METHOD}"
RESULTS_BASE="artifacts/results/Olmo-3-7b/group-polyglot/merged/${METHOD}"
EXPERT_STATS_DIR="artifacts/checkpoints/Olmo-3-7b/group-polyglot/experts"
export HF_HOME=$SCRATCH/huggingface NLTK_DATA=$SCRATCH/nltk_data SSL_CERT_DIR=/etc/ssl/certs

# 1. Merge (lm-eval venv)
source "$SCRATCH/actmat/.venv-pg-mgsm/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
if [[ -f "$MERGED_DIR/model.safetensors.index.json" ]]; then
  echo ">>> Skipping merge: $MERGED_DIR already exists"
else
  python src/hf/merge.py \
    --base-model-name-or-path "$BASE_MODEL" \
    --chat-template-name-or-path "${EXPERTS[2]}" \
    --expert-model-names-or-paths "${EXPERTS[@]}" \
    --merge-method "$METHOD" \
    --expert-stats-dir "$EXPERT_STATS_DIR" \
    --output-dir "$MERGED_DIR"
fi

# 2. M-GSM (CoT, 5-shot) — lm-eval, same venv. lm-eval nests the json under a
# sanitized-model-name subdir; flatten it up to lmeval/ to mirror lighteval.
lm_eval --model hf --model_args "pretrained=$MERGED_DIR,dtype=bfloat16,max_length=4096" \
  --tasks "$TASKS_LMEVAL" --num_fewshot 5 \
  --batch_size 16 --output_path "$RESULTS_BASE/lmeval"
for inner in "$RESULTS_BASE"/lmeval/*/; do
  [ -d "$inner" ] && { mv "$inner"* "$RESULTS_BASE/lmeval/"; rmdir "$inner"; }
done
deactivate

# 3. MMLU + M-RewardBench — lighteval fork venv
source "$SCRATCH/actmat/.venv-pg-mmlu-mrb/bin/activate"
lighteval vllm "model_name=$(realpath "$MERGED_DIR"),tensor_parallel_size=1,gpu_memory_utilization=0.9,max_model_length=8192,dtype=bfloat16,generation_parameters={max_new_tokens:4096,temperature:0.6,top_p:0.95}" \
  "$TASKS_LIGHTEVAL" \
  --custom-tasks scripts/olmo_polyglot/lighteval_tasks.py \
  --output-dir "$RESULTS_BASE/lighteval" \
  --results-path-template '{output_dir}' --save-details
