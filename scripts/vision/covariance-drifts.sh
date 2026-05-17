#!/bin/bash
#SBATCH --job-name=covariance-drifts
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=artifacts/logs/%x_%j.out
#SBATCH --error=artifacts/logs/%x_%j.err

# ─── Configuration ────────────────────────────────────────────────────────────
MODEL="ViT-B-16"
ROOTDIR="$SCRATCH/actmat"   # repo root on the cluster
RUN_DIR="artifacts/checkpoints-analysis-drift"   # subdir under ROOTDIR containing drift snapshots

DATASETS=(
    "Cars"
    "DTD"
    "EuroSAT"
    "GTSRB"
    "MNIST"
    "RESISC45"
    "SUN397"
    "SVHN"
)

NUM_BATCHES=10
BATCH_SIZE=32
SPLIT="train"
MHA="split"
COV_TYPE="sm"
COV_ESTIMATOR="full"
# ──────────────────────────────────────────────────────────────────────────────

source ".venv-vl/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs
export HF_HOME=$SCRATCH/huggingface

if [ ! -d "$SLURM_TMPDIR/data" ]; then
  cp downloads/data.tar.gz "$SLURM_TMPDIR/"
  tar -xzf "$SLURM_TMPDIR/data.tar.gz" -C "$SLURM_TMPDIR/"
fi
ln -sfn "$SLURM_TMPDIR/data" data

for DATASET in "${DATASETS[@]}"; do
    CKPT_DIR="$ROOTDIR/$RUN_DIR/${MODEL}/${DATASET}Val"

    if [ ! -d "$CKPT_DIR" ]; then
        echo "WARNING: checkpoint directory not found: $CKPT_DIR, skipping $DATASET"
        continue
    fi

    for ckpt in "$CKPT_DIR"/checkpoint_*.pt; do
        [ -f "$ckpt" ] || continue
        ckpt_name=$(basename "$ckpt" .pt)   # e.g. checkpoint_0, checkpoint_200, ...

        echo "[BASH] covariance.py | model: $MODEL | dataset: $DATASET | ckpt: $ckpt_name"
        python scripts/vision/covariance.py \
            --model="$MODEL" \
            --save="$ROOTDIR/$RUN_DIR" \
            --load="$ckpt" \
            --eval-datasets="$DATASET" \
            --cov-split="$SPLIT" \
            --cov-num-batches="$NUM_BATCHES" \
            --cov-batch-size="$BATCH_SIZE" \
            --mha="$MHA" \
            --cov-type="$COV_TYPE" \
            --cov-estimator="$COV_ESTIMATOR" \
            --cache-dir="$SCRATCH/openclip" \
            --data-location="data/vision"
    done
done
