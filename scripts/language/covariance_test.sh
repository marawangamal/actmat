#!/bin/bash
#SBATCH --job-name=t5base_cov_test
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=artifacts/logs/%x_%j.out
#SBATCH --error=artifacts/logs/%x_%j.err
# One-off: collect test-set covariance for T5-base (FFT) and install as the
# canonical covariance.pt via symlink, so the regmean eval consumes test stats.
set -euo pipefail
mkdir -p artifacts/logs

source "$SCRATCH/actmat/.venv-vl/bin/activate"
export HF_HOME=$SCRATCH/huggingface
export NLTK_DATA=$SCRATCH/nltk_data
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

if [ ! -d "$SLURM_TMPDIR/data" ]; then
  cp downloads/data.tar.gz "$SLURM_TMPDIR/"
  tar -xzf "$SLURM_TMPDIR/data.tar.gz" -C "$SLURM_TMPDIR/"
fi
ln -sfn "$SLURM_TMPDIR/data" data

NB=1000
BS=32
DATASETS=(qasc wiki_qa quartz paws story_cloze winogrande wsc)
TEST_NAME="covariance-test-nb${NB}-bs${BS}.pt"

# Unlink any existing covariance.pt SYMLINK so covariance.py's self-skip
# doesn't fire. Target files (covariance-{train,test}-nb*-bs*.pt) are
# preserved — only the symlink is removed.
for d in "${DATASETS[@]}"; do
  link="artifacts/checkpoints/t5-base/$d/covariance.pt"
  if [[ -L "$link" ]]; then
    echo "  unlinking existing symlink: $link -> $(readlink "$link")"
    rm "$link"
  fi
done

python scripts/language/covariance.py \
  --model=t5-base \
  --finetuning-mode=standard \
  --cov-split=test \
  --cov-num-batches=$NB \
  --cov-batch-size=$BS

for d in "${DATASETS[@]}"; do
  dir="artifacts/checkpoints/t5-base/$d"
  if [[ -f "$dir/covariance.pt" && ! -L "$dir/covariance.pt" ]]; then
    mv -v "$dir/covariance.pt" "$dir/$TEST_NAME"
    ln -sfn "$TEST_NAME" "$dir/covariance.pt"
    echo "  symlink: $dir/covariance.pt -> $TEST_NAME"
  else
    echo "  WARNING: $dir/covariance.pt not a regular file (skipped)"
  fi
done
