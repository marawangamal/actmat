#!/bin/bash
# Build the dedicated .venv-polyglot environment for the Polyglot-Teachers
# (OLMo3-7B) eval, replicating the paper's Lighteval setup on a stock release.
#
# Why a separate venv (not .venv-olmo): keeps the rebuttal's olmes/vllm stack
# untouched. Why these exact pins:
#   * lighteval 0.13.0   — the paper's eval framework (their 0.13.1.dev0 fork
#                          only adds M-RewardBench + a relaxed vllm pin; we skip
#                          M-RewardBench and relax the pin ourselves below).
#   * vllm 0.11.0        — required for Olmo3ForCausalLM (0.10.x lacks it).
#                          lighteval[vllm] would force vllm<0.10.2, so we install
#                          vllm separately and patch the runtime guard.
#   * transformers 4.57.6 — matches .venv-olmo (known-good with vllm 0.11.0 +
#                          OLMo3). transformers 5.x breaks vllm's tokenizer cache
#                          (TokenizersBackend lacks all_special_tokens_extended).
#
# Usage:  bash scripts/polyglot/setup_env.sh
set -euo pipefail

export UV_PROJECT_ENVIRONMENT=.venv-polyglot
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/uv-cache}"   # $HOME has a quota
export HF_HOME="${HF_HOME:-$SCRATCH/huggingface}"
mkdir -p "$UV_CACHE_DIR"

uv venv .venv-polyglot --python 3.11

# lighteval core (NOT the [vllm] extra — that pins vllm<0.10.2) + vllm 0.11.0 +
# the vllm-backend runtime deps lighteval expects + langcodes for task tags.
uv pip install --python .venv-polyglot \
  "lighteval==0.13.0" \
  "vllm==0.11.0" \
  "transformers==4.57.6" \
  ray more_itertools langcodes

# Relax lighteval's hard vllm<0.10.2 runtime guard for vllm only (same change
# the authors' fork makes). Idempotent.
.venv-polyglot/bin/python - <<'PY'
import pathlib, re
f = pathlib.Path(".venv-polyglot/lib/python3.11/site-packages/lighteval/utils/imports.py")
src = f.read_text()
needle = "        return installed in package.specifier"
patch = (
    "        # actmat patch: lighteval 0.13.0 pins vllm<0.10.2, but OLMo3 needs\n"
    "        # vllm>=0.11 (Olmo3ForCausalLM). Relax the upper bound for vllm only.\n"
    '        if package.name == "vllm":\n'
    '            return installed >= Version("0.10.0")\n'
    "\n"
    "        return installed in package.specifier"
)
if 'package.name == "vllm"' in src:
    print("vllm-pin patch already applied")
elif needle in src:
    f.write_text(src.replace(needle, patch, 1))
    print("applied vllm-pin patch to lighteval/utils/imports.py")
else:
    raise SystemExit("ERROR: could not locate patch site in imports.py")
PY

echo ">>> Validate (CPU, no GPU): source .venv-polyglot/bin/activate && python scripts/polyglot/cpu_precheck.py"
echo ">>> Done. .venv-polyglot ready."
