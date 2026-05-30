# Polyglot lighteval eval — documentation snapshot

These are **read-only snapshots** (copied 2026-05-30) of files that live in the
`polyglot-teachers/` **git submodule** (a separate repo, `ljvmiranda921/...`), so
they wouldn't otherwise be committed into this repo. Kept here for documentation /
reproducibility of the Polyglot-OLMo3-7B lighteval evaluation used by
`scripts/polyglot-all/`.

The live, executed copies are under `polyglot-teachers/`; edit those, not these.

## Files

| Snapshot | Live source | What it is |
|---|---|---|
| `SETUP.md` | `polyglot-teachers/SETUP.md` | how to build the eval venv (PyPI stack + the `lighteval` fork overlay) |
| `tesh.sh` | `polyglot-teachers/tesh.sh` | single-model lighteval run (base Olmo-3-7B on the mrb/mmlu/mgsm tasks) |
| `lighteval_tasks.py` | `polyglot-teachers/scripts/lighteval_tasks.py` | the paper's custom tasks (global_mmlu_lite, mrewardbench_mcf, mgsm_custom) |

## Key changes we made to the submodule (documented here)

1. **lighteval fork install** — the submodule ships `ljvmiranda921/lighteval`
   (0.13.1.dev0) but `pyproject.toml` pulls PyPI `lighteval==0.10.0`, which lacks
   `LogLikelihoodAccMetric`. Fix (see `SETUP.md`): `uv sync --extra eval` then
   overlay the fork editable with `--no-deps`, plus `uv pip install inspect-ai`.
   The fork can't be a managed uv source (its `numpy>=2`/`datasets>=4` deps clash
   with the project's pinned vllm stack).

2. **mgsm dataset swap** — `lighteval_tasks.py` hard-coded
   `hf_repo="ljvmiranda921/mgsm"`, which 404s (the author's fork was removed).
   Changed to `hf_repo="juletxara/mgsm"` (parquet, identical schema, de/es present).

## Eval protocol (paper Table 10)

- Global-MMLU-Lite — MCF, accuracy, 0-shot (key `acc`)
- M-RewardBench — MCF, **weighted** accuracy, 0-shot (key `weighted_acc`)
- M-GSM — generative exact-match, 5-shot (key `extractive_match`)

Number-only MGSM floors scores (~0.1) and penalizes chat-tuned models; the merge
analysis instead uses **CoT MGSM** via lm-eval `mgsm_native_cot_{de,es}` (see
`scripts/polyglot-all/eval_mgsm_cot.sh`), reported as `CMGSM` in the tables.
