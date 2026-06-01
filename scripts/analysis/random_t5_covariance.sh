#!/usr/bin/env bash
#SBATCH --job-name=rand_t5_cov
#SBATCH --partition=main
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --output=artifacts/logs/%x_%j.out
#SBATCH --error=artifacts/logs/%x_%j.err
# Collect covariances for a randomly-initialized t5-large on paws train,
# matching the trained-expert collection (sm, 10x32 samples).

set -euo pipefail
mkdir -p artifacts/logs

export PYTHONPATH="$PYTHONPATH:$(pwd)"
export HF_HOME=$SCRATCH/huggingface
export NLTK_DATA=$SCRATCH/nltk_data
export SSL_CERT_DIR=/etc/ssl/certs
source "$SCRATCH/actmat/.venv-vl/bin/activate"

# story_cloze etc. expect a local data/ symlink; paws comes from HF cache.
if [ -d "$SLURM_TMPDIR" ] && [ -f downloads/data.tar.gz ] && [ ! -d "$SLURM_TMPDIR/data" ]; then
  cp downloads/data.tar.gz "$SLURM_TMPDIR/" && tar -xzf "$SLURM_TMPDIR/data.tar.gz" -C "$SLURM_TMPDIR/"
  ln -sfn "$SLURM_TMPDIR/data" data
fi

OUT="artifacts/checkpoints-analysis/t5-large-random/paws/covariance.pt"

python scripts/analysis/random_t5_covariance.py \
    --model=t5-large \
    --cov-split=train \
    --cov-num-batches=10 \
    --cov-batch-size=32 \
    --cov-type=sm \
    --cov-estimator=full \
    --seed=0 \
    --dataset=paws \
    --out="$OUT"

echo "DONE -> $OUT"
