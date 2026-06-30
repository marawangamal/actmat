#!/bin/bash
#SBATCH --job-name=eval_medphi_experts
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err
# Evaluate the 5 MediPhi clinical experts on the 2 open CLUE tasks (MeQSum +
# LongHealth). Install the CLUE harness first:
#   bash scripts/medphi/setup.sh
#
# Submit with: sbatch --array=0-$((N-1)) scripts/medphi/eval_experts.sh
set -euo pipefail

CLUE_DIR="$SCRATCH/clue-eval/CLUE"
source "$CLUE_DIR/.venv-med/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export HF_HOME="$SCRATCH/huggingface"
export SSL_CERT_DIR=/etc/ssl/certs

EXPERTS=(
  MediPhi-PubMed
  MediPhi-Clinical
  MediPhi-MedCode
  MediPhi-MedWiki
  MediPhi-Guidelines
)
EXPERT="${EXPERTS[$SLURM_ARRAY_TASK_ID]}"
MODEL="microsoft/$EXPERT"
RESULTS_BASE="$PWD/artifacts/results/Phi-3.5-mini-instruct/group-mediphi/experts/$EXPERT"

# Stagger array starts: concurrent vLLM cold-starts race on the shared venv's
# jsonschema files (transient ENOENT on the network FS).
sleep $(( SLURM_ARRAY_TASK_ID * 90 ))

cd "$CLUE_DIR"
python eval/eval_meqsum.py --model "$MODEL" --num_few_shot_examples 3 \
  --data_path data/MeQSum/MeQSum_ACL2019_BenAbacha_Demner-Fushman.xlsx \
  --log_path "$RESULTS_BASE/meqsum"
python eval/eval_LongHealth.py --model "$MODEL" --max_len 8140 \
  --data_path data/LongHealth/benchmark_v5.json \
  --log_path "$RESULTS_BASE/longhealth8k"
