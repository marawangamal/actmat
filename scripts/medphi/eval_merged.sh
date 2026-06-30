#!/bin/bash
#SBATCH --job-name=eval_medphi
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err
# Merge Phi-3.5-mini-instruct + the 5 MediPhi clinical experts, then eval on the
# 2 openly-available CLUE tasks (MeQSum + LongHealth) with the patched CLUE harness.
#
# The other 4 CLUE tasks need PhysioNet/MIMIC credentials and the CLUE+ extras are
# unreleased, so only these 2 tasks run. The harness (greedy decoding,
# max_model_len, <|end|> stops) lives in its own .venv-med — install it first:
#   bash scripts/medphi/setup.sh
#
# Submit with: sbatch --array=0-$((N-1)) scripts/medphi/eval_merged.sh
set -euo pipefail

CLUE_DIR="$SCRATCH/clue-eval/CLUE"
source "$CLUE_DIR/.venv-med/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export HF_HOME="$SCRATCH/huggingface"
export SSL_CERT_DIR=/etc/ssl/certs

BASE_MODEL="microsoft/Phi-3.5-mini-instruct"
EXPERTS=(
  microsoft/MediPhi-PubMed
  microsoft/MediPhi-Clinical
  microsoft/MediPhi-MedCode
  microsoft/MediPhi-MedWiki
  microsoft/MediPhi-Guidelines
)
METHODS=(sum mean actmat tsv)
NUM_METHODS="${#METHODS[@]}"
if [ "$SLURM_ARRAY_TASK_ID" -ge "$NUM_METHODS" ]; then
  echo "No method for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
  exit 0
fi
METHOD="${METHODS[$SLURM_ARRAY_TASK_ID]}"
MERGED_DIR="$PWD/artifacts/checkpoints/Phi-3.5-mini-instruct/group-mediphi/merged/${METHOD}"
RESULTS_BASE="$PWD/artifacts/results/Phi-3.5-mini-instruct/group-mediphi/merged/${METHOD}"
EXPERT_STATS_DIR="$PWD/artifacts/checkpoints/Phi-3.5-mini-instruct/group-mediphi/experts"

# Stagger array starts: concurrent vLLM cold-starts race on the shared venv's
# jsonschema files (transient ENOENT on the network FS).
sleep $(( SLURM_ARRAY_TASK_ID * 90 ))

# 1. Merge (skip if exists). Chat template from base so the merged model can be
# evaluated with vLLM .chat().
if [[ -f "$MERGED_DIR/model.safetensors.index.json" ]]; then
  echo ">>> Skipping merge: $MERGED_DIR already exists"
else
  python src/hf2/merge.py \
    --base-model-name-or-path "$BASE_MODEL" \
    --chat-template-name-or-path "$BASE_MODEL" \
    --expert-model-names-or-paths "${EXPERTS[@]}" \
    --merge-method "$METHOD" \
    --expert-stats-dir "$EXPERT_STATS_DIR" \
    --output-dir "$MERGED_DIR"
fi

# 2. Evaluate the merged checkpoint (run from the harness dir for its relative imports)
cd "$CLUE_DIR"
python eval/eval_meqsum.py --model "$MERGED_DIR" --num_few_shot_examples 3 \
  --data_path data/MeQSum/MeQSum_ACL2019_BenAbacha_Demner-Fushman.xlsx \
  --log_path "$RESULTS_BASE/meqsum"
python eval/eval_LongHealth.py --model "$MERGED_DIR" --max_len 8140 \
  --data_path data/LongHealth/benchmark_v5.json \
  --log_path "$RESULTS_BASE/longhealth8k"
