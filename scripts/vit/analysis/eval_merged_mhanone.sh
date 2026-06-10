#!/bin/bash
#SBATCH --job-name=eval_vit_actmat_mhanon
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=artifacts/logs/%x_%A.out
#SBATCH --error=artifacts/logs/%x_%A.err

set -euo pipefail
mkdir -p artifacts/logs

source ".venv-vl/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

NUM_TASKS="${NUM_TASKS:-8}"
FT_MODE="${FT_MODE:-fft}"
METHOD="actmat"
MODEL="${MODEL:-ViT-B-16}"

# Stage datasets to $SLURM_TMPDIR (mirrors scripts/vit/eval_merged.sh)
if [ ! -d "$SLURM_TMPDIR/data" ]; then
  cp downloads/data.tar.gz "$SLURM_TMPDIR/"
  tar -xzf "$SLURM_TMPDIR/data.tar.gz" -C "$SLURM_TMPDIR/"
fi
DATA_DIR="$SLURM_TMPDIR/data/vision"
OPENCLIP_DIR="$SCRATCH/openclip"
CKPT_ROOT="artifacts/checkpoints"
RESULTS_ROOT="artifacts/results"

DATASETS_8="Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN"
DATASETS_14="${DATASETS_8},CIFAR100,STL10,Flowers102,OxfordIIITPet,PCAM,FER2013"
DATASETS_20="${DATASETS_14},EMNIST,CIFAR10,Food101,FashionMNIST,RenderedSST2,KMNIST"
case "$NUM_TASKS" in
  8) EVAL_DATASETS="$DATASETS_8" ;;
  14) EVAL_DATASETS="$DATASETS_14" ;;
  20) EVAL_DATASETS="$DATASETS_20" ;;
  *) echo "Unsupported NUM_TASKS=$NUM_TASKS"; exit 1 ;;
esac

EXPERTS_DIR="$CKPT_ROOT/$MODEL/group-legacy-$FT_MODE-$NUM_TASKS/experts"
OUT="$RESULTS_ROOT/$MODEL/group-legacy-$FT_MODE-$NUM_TASKS/merged/$METHOD"
python scripts/vit/eval_merged.py \
  --model "$MODEL" \
  --experts-dir "$EXPERTS_DIR" \
  --eval-datasets "$EVAL_DATASETS" \
  --merge-method "$METHOD" \
  --output-dir "$OUT" \
  --data-location "$DATA_DIR" \
  --cache-dir "$OPENCLIP_DIR" \
  --expert-kwargs '{"mha": "none"}'
