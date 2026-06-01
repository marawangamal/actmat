#!/bin/bash
#SBATCH --job-name=eval_vision
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-4
#SBATCH --output=artifacts/logs/%x_%A_%a.out
#SBATCH --error=artifacts/logs/%x_%A_%a.err

set -euo pipefail
mkdir -p artifacts/logs

# 0. Setup environment
source "$SCRATCH/actmat/.venv-vl/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

# Bucket bases. The 8/14/20 suite is carried by the `group-{N}` path level
# (--group=$NUM_TASKS below), uniform across checkpoints and results:
#   results:  artifacts/results/{model}/group-{N}/merged/{method}/[lora_]metrics.json
#   ckpts:    artifacts/checkpoints/{model}/group-{N}/experts/{dataset}Val/
# Vision expert checkpoints physically live in group-20 (the superset); group-8
# and group-14 experts are symlinks into it, so eval at any N reads one store.
CKPT_ROOT="artifacts/checkpoints"
DATA_DIR="$PWD/artifacts/data/vision"
OPENCLIP_DIR="$SCRATCH/openclip"

# 1. Stage datasets to $SLURM_TMPDIR (mirrors finetune.sh)
if [ ! -d "$SLURM_TMPDIR/data" ]; then
  cp downloads/data.tar.gz "$SLURM_TMPDIR/"
  tar -xzf "$SLURM_TMPDIR/data.tar.gz" -C "$SLURM_TMPDIR/"
fi
ln -sfn "$SLURM_TMPDIR/data" data

# Stage KMNIST raw files (torchvision's KMNIST mirror is unreliable from compute nodes).
KMNIST_RAW_DST="$SLURM_TMPDIR/data/vision/KMNIST/KMNIST/raw"
if [ ! -f "$KMNIST_RAW_DST/train-images-idx3-ubyte.gz" ] && [ -d downloads/kmnist ]; then
  mkdir -p "$KMNIST_RAW_DST"
  cp downloads/kmnist/*.gz "$KMNIST_RAW_DST/"
fi

# Stage PCAM h5 files (torchvision pulls from Google Drive and easily hits rate-limits).
PCAM_DST="$SLURM_TMPDIR/data/vision/PCAM/pcam"
PCAM_SRC="$PWD/artifacts/data/vision/PCAM/pcam"
if [ ! -f "$PCAM_DST/camelyonpatch_level_2_split_test_y.h5" ] && [ -d "$PCAM_SRC" ]; then
  mkdir -p "$PCAM_DST"
  for f in "$PCAM_SRC"/*.h5 "$PCAM_SRC"/*.h5.gz; do
    [ -f "$f" ] && ln -sfn "$f" "$PCAM_DST/$(basename "$f")"
  done
fi

# Common parameters
NUM_BATCHES=10
BATCH_SIZE=32

# ===== Benchmark configuration =====
# NUM_TASKS selects the dataset suite (override via `sbatch --export=ALL,NUM_TASKS=14 ...`
# or by editing the default) and the results suffix (results-{N}tasks), so the three
# former scripts collapse into this one.
#   8 : Cars, DTD, EuroSAT, GTSRB, MNIST, RESISC45, SUN397, SVHN
#   14: 8 + CIFAR100, STL10, Flowers102, OxfordIIITPet, PCAM, FER2013
#   20: 14 + EMNIST, CIFAR10, Food101, FashionMNIST, RenderedSST2, KMNIST
NUM_TASKS="${NUM_TASKS:-20}"

DATASETS_8="Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN"
DATASETS_14="${DATASETS_8},CIFAR100,STL10,Flowers102,OxfordIIITPet,PCAM,FER2013"
DATASETS_20="${DATASETS_14},EMNIST,CIFAR10,Food101,FashionMNIST,RenderedSST2,KMNIST"
case "$NUM_TASKS" in
  8)  EVAL_DATASETS="$DATASETS_8" ;;
  14) EVAL_DATASETS="$DATASETS_14" ;;
  20) EVAL_DATASETS="$DATASETS_20" ;;
  *)  echo "Unsupported NUM_TASKS=$NUM_TASKS (expected 8|14|20)"; exit 1 ;;
esac

# Task-count is the `group-{N}` path level (--group below); results_dir is bare.
RESULTS_DIR="artifacts/results"

MODELS=(ViT-B-32 ViT-L-14)
FT_MODE="lora"
MERGE_MODE=d
HPO=''

# Array dispatch: one array task per method. Keep --array=0-N in sync with len(METHODS)-1.
METHODS=(mean isoc tsv actmat wudi)
method="${METHODS[$SLURM_ARRAY_TASK_ID]}"

for MODEL in "${MODELS[@]}"; do
  # 2a. Run covariance/fisher script if needed (regmean + actmat consume covariance.pt; fisher consumes fisher.pt)
  if [ "$method" = "regmean" ] || [ "$method" = "actmat" ]; then
    echo "[BASH] Running covariance.py | model: $MODEL | ft mode: $FT_MODE | tasks: $NUM_TASKS"
    python scripts/vision/covariance.py \
      --model="$MODEL" \
      --finetuning-mode="$FT_MODE" \
      --save="$CKPT_ROOT" \
      --group="$NUM_TASKS" \
      --data-location="$DATA_DIR" \
      --eval-datasets="$EVAL_DATASETS" \
      --mha=split
  elif [ "$method" = "fisher" ]; then
    echo "[BASH] Running fisher.py | model: $MODEL | ft mode: $FT_MODE | tasks: $NUM_TASKS"
    python scripts/vision/fisher.py \
      --model="$MODEL" \
      --finetuning-mode="$FT_MODE" \
      --save="$CKPT_ROOT" \
      --group="$NUM_TASKS" \
      --data-location="$DATA_DIR" \
      --eval-datasets="$EVAL_DATASETS" \
      --mha=split
  fi

  # 2b. Evaluate task addition
  echo "[BASH] Running eval_task_addition.py | model: $MODEL | ft mode: $FT_MODE | method: $method | mode: $MERGE_MODE | tasks: $NUM_TASKS"
  python scripts/vision/eval_task_addition.py \
    --model="$MODEL" \
    --finetuning-mode="$FT_MODE" \
    --save="$CKPT_ROOT" \
    --data-location="$DATA_DIR" \
    --merge-func="$method" \
    --merge-mode="$MERGE_MODE" \
    --results-dir="$RESULTS_DIR" \
    --group="$NUM_TASKS" \
    --eval-datasets="$EVAL_DATASETS" \
    --mha=split \
    ${HPO:+--hpo="$HPO"}
done
