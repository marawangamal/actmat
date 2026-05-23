#!/bin/bash
#SBATCH --job-name=smoke_gemma_tok
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=artifacts/logs/%x_%j.out
#SBATCH --error=artifacts/logs/%x_%j.err
# Smoke test: does setting tokenizer_mode="slow" in vllm fix the U+2581 leak
# in Gemma-2-2B-IT code generation?
#
# Generates a humaneval-style completion from the existing mean-merged checkpoint
# with both fast and slow tokenizers, prints both outputs and whether ▁ appears.

set -euo pipefail
mkdir -p artifacts/logs
source "$SCRATCH/actmat/.venv-gemma/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

python scripts/gemma2bit/smoke_tokenizer.py
