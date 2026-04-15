set -euo pipefail

# Setup environment
export PYTHONPATH="$PYTHONPATH:$(pwd)" # Add src to python path
export HF_HOME=$SCRATCH/huggingface
export NLTK_DATA=$SCRATCH/nltk_data
source "$SCRATCH/eigcov/.venv-vl/bin/activate"

# Set vars
MODEL="ViT-B-16"
FT_MODE="standard"
CHECKPOINT_EVERY=200
CACHE_DIR="$SCRATCH/openclip"
DATA_DIR="$SLURM_TMPDIR/datasets"
SAVE_DIR="$SCRATCH/eigcov/analysis-checkpoints"
EPOCHS=2
MAX_STEPS=10

# Prepare datasets
if [ ! -d "$DATA_DIR" ]; then
  cp vit_datasets_08.zip "$SLURM_TMPDIR/"
  unzip -q "$SLURM_TMPDIR/vit_datasets_08.zip" -d "$SLURM_TMPDIR/"
fi

# 1. Finetune model
echo "[BASH] Running finetune.py | model: $MODEL | ft mode: $FT_MODE | checkpoint-every: $CHECKPOINT_EVERY | save: $SAVE_DIR"
python scripts/vision/finetune.py \
    --finetuning-mode="$FT_MODE" \
    --model="$MODEL" \
    --world-size=1 \
    --num-workers=1 \
    --checkpoint-every="$CHECKPOINT_EVERY" \
    --cache-dir="$CACHE_DIR" \
    --data-location="$DATA_DIR" \
    --save="$SAVE_DIR" \
    --grad-cross-matrix \
    --max-steps="$MAX_STEPS"