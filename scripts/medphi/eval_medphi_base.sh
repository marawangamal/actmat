#!/bin/bash
#SBATCH --job-name=eval_medphi_base
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=artifacts/logs/%x_%A.out
#SBATCH --error=artifacts/logs/%x_%A.err
# Evaluate the pretrained base model (Phi-3.5-mini-instruct) on the 2 open CLUE
# tasks (MeQSum + LongHealth). Install the CLUE harness first:
#   bash scripts/medphi/setup.sh
#
# Submit with: sbatch scripts/medphi/eval_medphi_base.sh
set -euo pipefail

CLUE_DIR="$SCRATCH/clue-eval/CLUE"
source "$CLUE_DIR/.venv-med/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export HF_HOME="$SCRATCH/huggingface"
export SSL_CERT_DIR=/etc/ssl/certs

MODEL="microsoft/Phi-3.5-mini-instruct"
RESULTS_BASE="$PWD/artifacts/results/Phi-3.5-mini-instruct/group-mediphi/pretrained"

cd "$CLUE_DIR"
python eval/eval_meqsum.py --model "$MODEL" --num_few_shot_examples 3 \
  --data_path data/MeQSum/MeQSum_ACL2019_BenAbacha_Demner-Fushman.xlsx \
  --log_path "$RESULTS_BASE/meqsum"
python eval/eval_LongHealth.py --model "$MODEL" --max_len 8140 \
  --data_path data/LongHealth/benchmark_v5.json \
  --log_path "$RESULTS_BASE/longhealth8k"
