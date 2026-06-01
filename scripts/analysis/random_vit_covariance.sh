#!/usr/bin/env bash
#SBATCH --job-name=rand_vit_cov
#SBATCH --partition=main
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=artifacts/logs/%x_%j.out
#SBATCH --error=artifacts/logs/%x_%j.err
# Collect covariances for a randomly-initialized ViT-L-14 on SVHN train,
# matching the trained-expert collection (split-MHA, sm, 10x32 samples).

set -euo pipefail
mkdir -p artifacts/logs

export PYTHONPATH="$PYTHONPATH:$(pwd)"
export HF_HOME=$SCRATCH/huggingface
export NLTK_DATA=$SCRATCH/nltk_data
source "$SCRATCH/actmat/.venv-vl/bin/activate"

OUT="artifacts/checkpoints-analysis/ViT-L-14-random/SVHNVal/covariance.pt"

python scripts/analysis/random_vit_covariance.py \
    --model=ViT-L-14 \
    --data-location=artifacts/data/vision \
    --cache-dir="$SCRATCH/openclip" \
    --mha=split \
    --cov-split=train \
    --cov-num-batches=10 \
    --cov-batch-size=32 \
    --cov-type=sm \
    --cov-estimator=full \
    --num-workers=4 \
    --seed=0 \
    --out="$OUT"

echo "DONE -> $OUT"
