#!/bin/bash
#SBATCH --job-name=eval_vision_tact
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=artifacts/logs/%x_%j.out
#SBATCH --error=artifacts/logs/%x_%j.err

set -euo pipefail
mkdir -p artifacts/logs

# 0. Setup environment
source "$SCRATCH/actmat/.venv-vl/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

DATA_DIR="data/vision"
OPENCLIP_DIR="$SCRATCH/openclip"

if [ ! -d "$SLURM_TMPDIR/data" ]; then
  cp downloads/data.tar.gz "$SLURM_TMPDIR/"
  tar -xzf "$SLURM_TMPDIR/data.tar.gz" -C "$SLURM_TMPDIR/"
fi
ln -sfn "$SLURM_TMPDIR/data" data

# ===== TACT experiments =====
MODELS=(ViT-B-32 ViT-L-14)
METHODS=(tact)
FT_MODES=(standard)
MERGE_MODE=d

# Per-layer trim keep fraction. Sweep reproduces the Appendix C grid; the script
# picks the best-on-val combo and reports its test accuracy.
HPO='{"tact_k": [0.1, 0.2, 0.3, 0.5, 0.7]}'


for FT_MODE in "${FT_MODES[@]}"; do
for MODEL in "${MODELS[@]}"; do
  for method in "${METHODS[@]}"; do
    echo "[BASH] Running eval_task_addition.py | model: $MODEL | ft mode: $FT_MODE | method: $method | mode: $MERGE_MODE | hpo: $HPO"
    python scripts/vision/eval_task_addition.py \
      --model="$MODEL" \
      --finetuning-mode="$FT_MODE" \
      --data-location="$DATA_DIR" \
      --merge-func="$method" \
      --merge-mode="$MERGE_MODE" \
      --mha=split \
      --hpo="$HPO"
  done
done
done
