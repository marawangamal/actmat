#!/usr/bin/env bash
#SBATCH --job-name=test_language_e2e
#SBATCH --partition=main
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/logs/%x_%j.out
#SBATCH --error=artifacts/logs/%x_%j.err
# Minimal end-to-end smoke test for the LANGUAGE (T5) pipeline: briefly finetune a
# couple datasets, then eval experts + merge. Mirrors test_vision_e2e.sh; writes to
# dedicated test dirs so it doesn't collide with real runs. Language quirks vs vision:
# bare dataset dirs (no `Val` suffix), no `head.pt`, and the default group is `main`.

set -euo pipefail
mkdir -p artifacts/logs

# Setup environment
export PYTHONPATH="$PYTHONPATH:$(pwd)" # Add src to python path
export HF_HOME=$SCRATCH/huggingface
export NLTK_DATA=$SCRATCH/nltk_data
export SSL_CERT_DIR=/etc/ssl/certs
source "$SCRATCH/actmat/.venv-vl/bin/activate"

# Set vars
TEST_CKPT_DIR="artifacts/testing-checkpoints"
TEST_RESULTS_DIR="artifacts/testing-results"
MODEL="t5-base"
DATASETS="paws,wiki_qa"   # two small HF datasets
MAX_STEPS=2

# Prepare datasets (mirrors scripts/language/eval_experts.sh)
if [ ! -d "$SLURM_TMPDIR/data" ]; then
  cp downloads/data.tar.gz "$SLURM_TMPDIR/"
  tar -xzf "$SLURM_TMPDIR/data.tar.gz" -C "$SLURM_TMPDIR/"
fi
ln -sfn "$SLURM_TMPDIR/data" data

echo "=== [1/4] finetune (max-steps=$MAX_STEPS on $DATASETS) ==="
python scripts/language/finetune.py \
    --model="$MODEL" \
    --finetuning-mode=standard \
    --train-dataset="$DATASETS" \
    --save="$TEST_CKPT_DIR" \
    --max-steps="$MAX_STEPS"

echo "=== [2/4] eval_experts (mode=none, writes pretrained) ==="
python scripts/language/eval_experts.py \
    --model="$MODEL" \
    --finetuning-mode=none \
    --eval-datasets="$DATASETS" \
    --save="$TEST_CKPT_DIR" \
    --max-steps="$MAX_STEPS" \
    --results-dir="$TEST_RESULTS_DIR"

echo "=== [3/4] eval_experts (mode=standard, writes experts) ==="
python scripts/language/eval_experts.py \
    --model="$MODEL" \
    --finetuning-mode=standard \
    --eval-datasets="$DATASETS" \
    --save="$TEST_CKPT_DIR" \
    --max-steps="$MAX_STEPS" \
    --results-dir="$TEST_RESULTS_DIR"

echo "=== [4/4] eval_task_addition (merge=sum) ==="
python scripts/language/eval_task_addition.py \
    --model="$MODEL" \
    --finetuning-mode=standard \
    --eval-datasets="$DATASETS" \
    --save="$TEST_CKPT_DIR" \
    --max-steps="$MAX_STEPS" \
    --results-dir="$TEST_RESULTS_DIR" \
    --merge-func=sum \
    --merge-mode=d \
    --overwrite

# Structured layout (standard FT => no lora_ prefix). No --group is passed, so
# everything nests under the default group-main path level.
for f in \
    "$TEST_RESULTS_DIR/$MODEL/group-main/pretrained/metrics.json" \
    "$TEST_RESULTS_DIR/$MODEL/group-main/experts/metrics.json" \
    "$TEST_RESULTS_DIR/$MODEL/group-main/merged/sum/metrics.json"; do
    if [[ ! -f "$f" ]]; then
        echo "FAIL: expected $f to exist"
        exit 1
    fi
    echo "PASS: $f"
done
