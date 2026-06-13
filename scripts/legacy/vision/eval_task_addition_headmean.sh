#!/bin/bash
#SBATCH --job-name=eval_vision_headmean
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-1
#SBATCH --time=12:00:00
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err

# Head-mean variant of eval_task_addition.sh (8-task suite): identical merge,
# except the FF up-projection layers (mlp.c_fc) are forced to a plain MEAN merge
# (--mean-keys) instead of the chosen method. c_fc carries 81-87% of ViT's
# summed RegMean loss and is where the unregularized ACTMat pseudo-inverse
# overshoots (see artifacts/notes/regmean-loss-vs-accuracy.md). Writes to its OWN
# group (group-8-headmean) so it never clobbers the full-method group-8 results;
# experts are symlinked group-8-headmean/experts -> group-20/experts (same farm
# as group-8). Confirm the override fired: grep "[mean_keys]" the merge logs.

set -euo pipefail
mkdir -p artifacts/logs

# 0. Setup environment
source "$SCRATCH/actmat/.venv-vl/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

CKPT_ROOT="artifacts/checkpoints"
DATA_DIR="$PWD/artifacts/data/vision"

# 1. Stage datasets to $SLURM_TMPDIR (mirrors eval_task_addition.sh)
if [ ! -d "$SLURM_TMPDIR/data" ]; then
  cp downloads/data.tar.gz "$SLURM_TMPDIR/"
  tar -xzf "$SLURM_TMPDIR/data.tar.gz" -C "$SLURM_TMPDIR/"
fi
ln -sfn "$SLURM_TMPDIR/data" data

KMNIST_RAW_DST="$SLURM_TMPDIR/data/vision/KMNIST/KMNIST/raw"
if [ ! -f "$KMNIST_RAW_DST/train-images-idx3-ubyte.gz" ] && [ -d downloads/kmnist ]; then
  mkdir -p "$KMNIST_RAW_DST"
  cp downloads/kmnist/*.gz "$KMNIST_RAW_DST/"
fi
PCAM_DST="$SLURM_TMPDIR/data/vision/PCAM/pcam"
PCAM_SRC="$PWD/artifacts/data/vision/PCAM/pcam"
if [ ! -f "$PCAM_DST/camelyonpatch_level_2_split_test_y.h5" ] && [ -d "$PCAM_SRC" ]; then
  mkdir -p "$PCAM_DST"
  for f in "$PCAM_SRC"/*.h5 "$PCAM_SRC"/*.h5.gz; do
    [ -f "$f" ] && ln -sfn "$f" "$PCAM_DST/$(basename "$f")"
  done
fi

# 8-task suite (matches the rm-loss / group-8 analysis).
EVAL_DATASETS="Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN"
RESULTS_DIR="artifacts/results"
# NOTE: FFT (standard), not LoRA. LoRA adapts only the attention projections, so
# the c_fc delta is zero under LoRA and mean-merging it is a no-op (verified:
# ||Δ c_fc||=0 for lora_finetuned, 0.45 for finetuned). The c_fc layers only
# carry a delta — and dominate the RegMean loss — in the full finetune.
MODELS=(ViT-B-32 ViT-L-14)
FT_MODE="standard"
MERGE_MODE=d
MEAN_KEYS="mlp.c_fc"

# actmat/regmean consume covariance.pt (FFT), already present under the shared
# group-20 experts, so no covariance recompute is needed here.
METHODS=(actmat regmean)
method="${METHODS[$SLURM_ARRAY_TASK_ID]}"

for MODEL in "${MODELS[@]}"; do
  echo "[BASH] eval_task_addition.py | model=$MODEL method=$method group=8-headmean mean_keys=$MEAN_KEYS"
  python scripts/vision/eval_task_addition.py \
    --model="$MODEL" \
    --finetuning-mode="$FT_MODE" \
    --save="$CKPT_ROOT" \
    --data-location="$DATA_DIR" \
    --merge-func="$method" \
    --merge-mode="$MERGE_MODE" \
    --results-dir="$RESULTS_DIR" \
    --group=8-headmean \
    --eval-datasets="$EVAL_DATASETS" \
    --mean-keys "$MEAN_KEYS" \
    --mha=split
done
