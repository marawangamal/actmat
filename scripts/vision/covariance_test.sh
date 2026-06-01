#!/bin/bash
#SBATCH --job-name=vit16_cov_test
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=artifacts/logs/%x_%j.out
#SBATCH --error=artifacts/logs/%x_%j.err
# One-off: collect test-set covariance for ViT-B-16 (FFT) on the canonical
# 8-task benchmark and install as covariance.pt via symlink so regmean eval
# consumes test stats. Uses --mha=split to match the existing cov file
# convention (q/k/v/o per-block keys).
set -euo pipefail
mkdir -p artifacts/logs

source "$SCRATCH/actmat/.venv-vl/bin/activate"
export HF_HOME=$SCRATCH/huggingface
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

if [ ! -d "$SLURM_TMPDIR/data" ]; then
  cp downloads/data.tar.gz "$SLURM_TMPDIR/"
  tar -xzf "$SLURM_TMPDIR/data.tar.gz" -C "$SLURM_TMPDIR/"
fi
ln -sfn "$SLURM_TMPDIR/data" data

MODEL=ViT-B-16
NB=100
BS=32
DATASETS=(Cars DTD EuroSAT GTSRB MNIST RESISC45 SUN397 SVHN)
TEST_NAME="covariance-test-nb${NB}-bs${BS}.pt"

# Unlink existing covariance.pt SYMLINK so covariance.py's self-skip doesn't
# fire. Target files (covariance-nb*-bs*.pt / covariance-test-*.pt) preserved.
for d in "${DATASETS[@]}"; do
  link="artifacts/checkpoints/${MODEL}/${d}Val/covariance.pt"
  if [[ -L "$link" ]]; then
    echo "  unlinking symlink: $link -> $(readlink "$link")"
    rm "$link"
  fi
done

python scripts/vision/covariance.py \
  --model="$MODEL" \
  --finetuning-mode=standard \
  --data-location=data/vision \
  --cov-split=test \
  --cov-num-batches="$NB" \
  --cov-batch-size="$BS" \
  --mha=split

for d in "${DATASETS[@]}"; do
  dir="artifacts/checkpoints/${MODEL}/${d}Val"
  if [[ -f "$dir/covariance.pt" && ! -L "$dir/covariance.pt" ]]; then
    mv -v "$dir/covariance.pt" "$dir/$TEST_NAME"
    ln -sfn "$TEST_NAME" "$dir/covariance.pt"
    echo "  symlink: $dir/covariance.pt -> $TEST_NAME"
  else
    echo "  WARNING: $dir/covariance.pt not a regular file (skipped)"
  fi
done
