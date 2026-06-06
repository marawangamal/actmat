#!/bin/bash
# Minimal single-task lm-eval run for a quick sanity check of the eval stack.
# Defaults to OLMo-3 base on MGSM German (multilingual GSM8K): 5-shot, greedy,
# generative exact-match. hf backend (no vLLM), batch 32.
#
# Usage:
#   bash scripts/polyglot/simple.sh
#   TASK=global_mmlu_de NUM_FEWSHOT=0 bash scripts/polyglot/simple.sh
#   MODEL=ljvmiranda921/Polyglot-OLMo3-7B-SFT-de bash scripts/polyglot/simple.sh
set -euo pipefail

source "$SCRATCH/actmat/.venv-pg/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export HF_HOME=$SCRATCH/huggingface
export SSL_CERT_DIR=/etc/ssl/certs

# MODEL="${MODEL:-allenai/Olmo-3-1025-7B}"
MODEL="${MODEL:-ljvmiranda921/Polyglot-OLMo3-7B-SFT-de}"
# Expert override, e.g.: MODEL=ljvmiranda921/Polyglot-OLMo3-7B-SFT-de bash scripts/polyglot/simple.sh
TASK="${TASK:-mgsm_native_cot_de}"  # paper's MATH benchmark (M-GSM), German, CoT
NUM_FEWSHOT="${NUM_FEWSHOT:-5}"     # MGSM is 5-shot in the paper

lm_eval --model hf \
  --model_args "pretrained=${MODEL},dtype=bfloat16,max_length=2048" \
  --tasks "$TASK" --num_fewshot "$NUM_FEWSHOT" --batch_size 32 \
  --output_path "artifacts/results-polyglot/simple_${TASK}"
