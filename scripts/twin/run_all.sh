#!/bin/bash
# Twin-Merging experiments: Qwen-14B with 4 LoRA adapters.
#
# Merges adapters with eigcov, mean, tsv, isoc and evaluates on
# BBQ (EM), CNN/DailyMail (ROUGE-2), MMLU (EM), TruthfulQA (EM).
#
# Prerequisites:
#   pip install lm-eval peft safetensors
#   export PYTHONPATH="$PYTHONPATH:$PWD"
#
# Usage:
#   bash scripts/twin/run_all.sh

set -euo pipefail

export HF_HOME=${HF_HOME:-${SCRATCH:-$HOME}/.cache/huggingface}

BASE_MODEL="Qwen/Qwen-14B"
CKPT_DIR="checkpoints/twin"
RESULTS_DIR="results/twin"

ADAPTERS=(
    "lu-vae/qwen-bbq-merged"
    "lu-vae/qwen-cnn-merged"
    "lu-vae/qwen-mmlu-merged"
    "lu-vae/qwen-truthfulqa-merged"
)

METHODS="eigcov mean tsv isoc"

# ──────────────────────────────────────────────────────────────
# Step 0: Convert base model to param-folder
# ──────────────────────────────────────────────────────────────
echo "=== Step 0: Convert base model to param-folder ==="
if [ ! -f "${CKPT_DIR}/Qwen-14B/param_manifest.json" ]; then
    python scripts/nlg/save_model_param_folder.py \
        --model "${BASE_MODEL}" \
        --output-dir "${CKPT_DIR}/Qwen-14B"
else
    echo "  Skipping — ${CKPT_DIR}/Qwen-14B already exists"
fi

# ──────────────────────────────────────────────────────────────
# Step 1: Convert each LoRA adapter to param-folder
# ──────────────────────────────────────────────────────────────
echo "=== Step 1: Convert LoRA adapters to param-folder ==="
for adapter in "${ADAPTERS[@]}"; do
    # lu-vae/qwen-bbq-merged -> qwen-bbq-merged
    name=$(echo "${adapter}" | cut -d'/' -f2)
    outdir="${CKPT_DIR}/${name}"
    if [ ! -f "${outdir}/param_manifest.json" ]; then
        echo "  Converting: ${adapter}"
        python scripts/twin/save_lora_param_folder.py \
            --base-model "${BASE_MODEL}" \
            --adapter "${adapter}" \
            --output-dir "${outdir}"
    else
        echo "  Skipping — ${outdir} already exists"
    fi
done

# ──────────────────────────────────────────────────────────────
# Step 2: Merge with each method
# ──────────────────────────────────────────────────────────────
echo "=== Step 2: Merge task vectors ==="

# Build finetuned-dirs list
FT_DIRS=""
for adapter in "${ADAPTERS[@]}"; do
    name=$(echo "${adapter}" | cut -d'/' -f2)
    FT_DIRS="${FT_DIRS} ${CKPT_DIR}/${name}"
done

for method in ${METHODS}; do
    outdir="${CKPT_DIR}/Qwen-14B-${method}"
    if [ ! -d "${outdir}" ]; then
        echo "  Merging with: ${method}"
        python scripts/nlg/merge.py \
            --pretrained-dir "${CKPT_DIR}/Qwen-14B" \
            --finetuned-dirs ${FT_DIRS} \
            --merge-func "${method}" \
            --output-dir "${outdir}" \
            --trust-remote-code
    else
        echo "  Skipping — ${outdir} already exists"
    fi
done

# ──────────────────────────────────────────────────────────────
# Step 3: Evaluate merged models
# ──────────────────────────────────────────────────────────────
echo "=== Step 3: Evaluate merged models ==="
for method in ${METHODS}; do
    echo "  Evaluating: ${method}"
    bash scripts/twin/eval.sh \
        "${CKPT_DIR}/Qwen-14B-${method}" \
        "${RESULTS_DIR}/${method}"
done

# ──────────────────────────────────────────────────────────────
# Step 4: Evaluate individual adapters (upper bound)
# ──────────────────────────────────────────────────────────────
echo "=== Step 4: Evaluate individual adapters ==="
for adapter in "${ADAPTERS[@]}"; do
    name=$(echo "${adapter}" | cut -d'/' -f2)
    echo "  Evaluating: ${name}"
    bash scripts/twin/eval.sh \
        "${CKPT_DIR}/${name}" \
        "${RESULTS_DIR}/individual-${name}"
done

# ──────────────────────────────────────────────────────────────
# Step 5: Collect results
# ──────────────────────────────────────────────────────────────
echo "=== Step 5: Collect results ==="

# Build dirs list for collect_results.py
RESULT_DIRS=""
for method in ${METHODS}; do
    RESULT_DIRS="${RESULT_DIRS} ${RESULTS_DIR}/${method}"
done
for adapter in "${ADAPTERS[@]}"; do
    name=$(echo "${adapter}" | cut -d'/' -f2)
    RESULT_DIRS="${RESULT_DIRS} ${RESULTS_DIR}/individual-${name}"
done

python scripts/twin/collect_results.py --dirs ${RESULT_DIRS}

echo "=== Done ==="
