#!/bin/bash
# Set up the CLUE clinical-eval harness for the MediPhi reproduction experiment.
# Clones TIO-IKIM/CLUE (pinned), applies our patch, builds the .venv-med env, and
# fetches the 2 OPENLY-available datasets (MeQSum, LongHealth).
#
# The other 4 CLUE tasks (MedNLI, Problem List Summary, MeDiSumQA, MeDiSumCode)
# require credentialed PhysioNet / MIMIC-IV access, and the CLUE+ 12-task extras
# from the MediPhi paper (arXiv 2505.10717) are not publicly released. So this
# reproduces the 2 open tasks only.
#
# Usage:  bash scripts/medphi/setup.sh
# Then:   sbatch scripts/medphi/eval_base.sh   (etc. — see README.md)
set -euo pipefail

CLUE_COMMIT=2de4314cfaf2651a362a2df2ac8336ca0937ea52
CLUE_DIR="${CLUE_DIR:-$SCRATCH/clue-eval/CLUE}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/.uv-cache}"   # keep wheels off the home quota
export HF_HOME="${HF_HOME:-$SCRATCH/huggingface}"

# 1. Clone at the pinned commit
if [[ ! -d "$CLUE_DIR/.git" ]]; then
  mkdir -p "$(dirname "$CLUE_DIR")"
  git clone https://github.com/TIO-IKIM/CLUE.git "$CLUE_DIR"
fi
cd "$CLUE_DIR"
git checkout -q "$CLUE_COMMIT" 2>/dev/null || { git fetch origin && git checkout -q "$CLUE_COMMIT"; }

# 2. Apply our patch: python 3.11, datasets>=3.0 + transformers==4.45.2 pins,
#    max_model_len=16384, greedy decoding + <|end|> stop tokens. Idempotent.
if git apply --check "$HERE/clue.patch" 2>/dev/null; then
  git apply "$HERE/clue.patch" && echo ">>> patch applied"
else
  echo ">>> patch already applied (or does not apply cleanly — check manually)"
fi

# 3. Build the env (Python 3.11; vllm 0.6.6, scispacy, custom bert_score fork)
UV_PROJECT_ENVIRONMENT=.venv-med uv sync

# 4. Fetch the 2 OPEN datasets (the buggy upstream download_datasets.sh points at
#    GitHub blob HTML pages and needs a PhysioNet user for the gated ones)
mkdir -p data/MeQSum data/LongHealth
[[ -f data/MeQSum/MeQSum_ACL2019_BenAbacha_Demner-Fushman.xlsx ]] || \
  curl -sL -o data/MeQSum/MeQSum_ACL2019_BenAbacha_Demner-Fushman.xlsx \
    https://raw.githubusercontent.com/abachaa/MeQSum/master/MeQSum_ACL2019_BenAbacha_Demner-Fushman.xlsx
[[ -f data/LongHealth/benchmark_v5.json ]] || \
  curl -sL -o data/LongHealth/benchmark_v5.json \
    https://raw.githubusercontent.com/kbressem/LongHealth/main/data/benchmark_v5.json

echo
echo "Setup complete in $CLUE_DIR"
echo "Run from the repo root:"
echo "  sbatch        scripts/medphi/eval_base.sh"
echo "  sbatch --array=0-4 scripts/medphi/eval_experts.sh"
echo "  sbatch --array=0-3 scripts/medphi/eval_merged.sh"
