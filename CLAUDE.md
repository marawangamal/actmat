# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Source code for the paper **Model Merging via Data-Free Covariance Estimation** (ACTMat). Implements and benchmarks task-vector merging methods (sum/TA, mean, TSV, IsoC, RegMean, Fisher, ACTMat) across three pipelines: vision (OpenCLIP ViTs), language (T5), and OLMo (RL-Zero 7B experts).

## Environments

Vision/language and OLMo dependency groups **conflict** (declared in `pyproject.toml`); they live in separate uv venvs and must be created/activated separately:

```sh
UV_PROJECT_ENVIRONMENT=.venv-vl   uv sync --group vision-language
UV_PROJECT_ENVIRONMENT=.venv-olmo uv sync --group olmo
```

The **polyglot** (multilingual OLMo3) experiment adds two more conflicting groups,
each its own venv (their numpy/datasets pins are mutually incompatible):

```sh
# M-GSM (lm-eval) — plain sync
UV_PROJECT_ENVIRONMENT=.venv-pg-mgsm uv sync --group polyglot-mgsm
# MMLU+MRB (lighteval fork) — sync the PyPI base, then OVERLAY the ./lighteval
# submodule with --no-deps (a managed source would fail to resolve numpy>=2/datasets>=4)
UV_PROJECT_ENVIRONMENT=.venv-pg-mmlu-mrb uv sync --group polyglot-mmlu-mrb
UV_PROJECT_ENVIRONMENT=.venv-pg-mmlu-mrb uv pip install --python .venv-pg-mmlu-mrb -e ./lighteval --no-deps
```
Re-running `uv sync --group polyglot-mmlu-mrb` clobbers the fork — re-run the
overlay line. See the repo README for full details.

Every shell that runs a script needs:

```sh
export PYTHONPATH="$PYTHONPATH:$(pwd)"   # repo root is the src root
export HF_HOME=$SCRATCH/huggingface
export NLTK_DATA=$SCRATCH/nltk_data
source .venv-vl/bin/activate              # or .venv-olmo / .venv-pg-mmlu-mrb / .venv-pg-mgsm
```

`olmes/` and `lighteval/` are git submodules (olmes path-installed via `ai2-olmes`; lighteval overlaid `--no-deps` into `.venv-pg-mmlu-mrb`). Use `git submodule update --init --recursive` after a fresh clone.

Vision experiments expect `vit_datasets_08.zip` (symlinked from `~/scratch/actmat-2026-05-04/`) to be copied to `$SLURM_TMPDIR/datasets`; this is done by the SLURM scripts.

## Common commands

End-to-end smoke tests (2 training steps, then eval + merge; verify the grouped
`group-main` result paths). Vision uses MNIST+SVHN/ViT-B-32; language uses
paws+wiki_qa/t5-base:

```sh
sbatch scripts/tests/test_vision_e2e.sh
sbatch scripts/tests/test_language_e2e.sh
```

Run a single vision merge eval directly (skip SLURM):

```sh
python scripts/vision/eval_task_addition.py \
  --model=ViT-B-32 --finetuning-mode=lora \
  --merge-func=actmat --merge-mode=d --mha=split
```

Driver SLURM scripts loop over `MODELS × METHODS × FT_MODES`:
- `scripts/vision/finetune.sh`, `scripts/vision/eval_task_addition.sh`
- `scripts/language/eval_task_addition.sh`
- `scripts/olmo/eval_task_addition.sh`

When a method needs statistics, the driver runs `covariance.py` (regmean, actmat) or `fisher.py` (fisher) for the model first, then `eval_task_addition.py`. Statistics live next to checkpoints as `covariance.pt` / `covariance/*.pt` / `fisher.pt` and are auto-discovered by `_TaskVector`.

## Architecture

### Three pipelines, one CLI

`src/args.py` is a **single shared argument parser** for all three pipelines. Pipeline-specific flags (`--capability` for OLMo, `--mha` for vision, `--max-seq-len` for language) live in the same parser; scripts ignore the flags they don't use. When adding a flag, put it in `src/args.py`, not in a per-script parser.

### Task-vector core

`src/task_vectors.py` defines the abstract `_TaskVector` (a dict of per-parameter tensors with arithmetic + lazy disk loading). Each pipeline subclasses it:
- `src/vision/task_vectors.py` — OpenCLIP ViT checkpoints
- `src/language/task_vectors.py` — HF T5 checkpoints
- `src/nlg/task_vectors.py` — folder-of-safetensors checkpoints for OLMo (`ParamFolderTaskVector`)

A `_TaskVector` is built from a `checkpoint_dir` containing `pretrained.pt` + `{prefix}finetuned.pt` (prefix is `""` for FFT, `"lora_"` for LoRA). OLMo/WizardLM use param-folder checkpoints — `pretrained/` + `finetuned/` subdirectories instead of `.pt` files. It auto-discovers `covariance.pt`/`fisher.pt` siblings.

### Merging

`src/merging.py::combine_task_vectors(vectors, merge_fn_name, **kwargs)` is the single entry point. It looks up `merge_<name>` in the same module — the `--merge-func` CLI value maps directly to a function name (`sum`, `mean`, `tsv`, `isoc`, `regmean`, `fisher`, `actmat`, `ace`, `actmat_general`, `actmat_gd`, `sum04`). To add a method, define `merge_<name>(taus, **kwargs)` in `src/merging.py`; nothing else needs registration. `--merge-mode d` merges differences (Δ = θ_ft − θ_pre); `--merge-mode w` merges raw weights (used by RegMean variants — triggers `save_pt=True` on the task vector).

### Statistics collection

`covariance.py` / `fisher.py` in each pipeline's `scripts/` directory walk the model's linear/attention layers, accumulate (un)centered input second moments or diagonal Fisher over a few batches of training data, and write `covariance.pt` / `fisher.pt` next to the corresponding finetuned checkpoint. Knobs: `--cov-num-batches`, `--cov-batch-size`, `--cov-type {sm,cov}`, `--cov-estimator {full,sampled,avg}`, `--mha {split,packed}` (vision: replaces `nn.MultiheadAttention` with a custom module so Q/K/V cov can be collected per-head).

### Artifacts layout (vision + language)

All pipelines follow a structured, nested convention whose path builders are the
single source of truth in `src/utils.py` (`group_dir`, `resolve_run_dir`, `expert_dir`,
`head_path`, `merged_results_path`, `experts_results_path`, `pretrained_results_path`,
`multitask_results_path`). `[lora_]` is the `get_prefix()` filename prefix; `{mode}`
is `-w` for weight-space merges (omitted for the default difference merge). `group-{g}`
is the **experiment-suite path level** (`--group`, default `main`) between `{model}` and
the `experts|multitask|merged|pretrained` subdirs — vision `group-{8,14,20}`, OLMo
`group-{rl-zero,polyglot}`, everything else `group-main`. `resolve_run_dir` injects it
for checkpoints; the `*_results_path` builders take it as the `group=` kwarg.

```
checkpoints/{model}/group-{g}/experts/{dataset}[Val]/ pretrained.pt, [lora_]finetuned.pt, [lora_]covariance.pt, fisher.pt[, head.pt]
checkpoints/{model}/group-{g}/multitask/             MTL checkpoint
checkpoints/{model}/pretrained.pt                    shared base, model-level (above the group)

results/{model}/group-{g}/merged/{method}[-{mode}]/[lora_]metrics.json
results/{model}/group-{g}/experts/[lora_]metrics.json
results/{model}/group-{g}/pretrained/[lora_]metrics.json   (zero-shot baseline)
results/{model}/group-{g}/multitask/[lora_]metrics.json
```

Vision-only quirks: dataset dirs carry a `Val` split suffix (`expert_dir(..., val_suffix=True)`,
the default) and each holds a co-located `head.pt`; language passes `val_suffix=False`
and has neither. The vision task-count (8/14/20) is the `group` value, passed as
`--group=$NUM_TASKS` by a single `eval_task_addition.sh` / `eval_experts.sh`. Vision
expert **checkpoints are shared across suites**: they live physically in `group-20`
(the superset), and `group-8` / `group-14` expert dirs are **symlinks** into it (so
finetuning runs once; `finetune_mtl.sh` passes `--group=20`).
Language uses `group-main` throughout. Named experiment buckets (`results-wang`,
`results-sgd`, `results-mixed`) are exotic and stay on `group-main` by default.

Two migration passes, both dry-run by default with reversible, move-only undo scripts:
`scripts/vision/migrate_artifacts.py` (old **flat → nested**; `--pipeline {vision,language,olmo}`,
`--canonical`, undo `artifacts/migrate_undo.sh`) and `scripts/migrate_to_groups.py`
(**nested → grouped**; `--pipeline {vision,language,olmo,all}`, undo
`artifacts/migrate_groups_undo.sh`). The grouped pass builds the vision symlink farms and
**unifies OLMo** `Olmo-3-7b` (rl-zero) + `Olmo-3-7b-polyglot-all` (polyglot) into one
`Olmo-3-7b` model dir, repointing the merged-view symlinks. Exotic vision buckets
(sgd/wang/ilharco/mixed/analysis, the dashless `results14`/`results20`), the `t5-*-faulty`
results, and the `results-polyglot*` buckets are deferred (still pre-group).

An append-only JSON-lines DB (`src/results_db.py`) hashes `(script, args)` into a
16-char run id so identical configurations are deduped; pass `--results-db <path>`
to use it, `--overwrite` to ignore the cache.

### Vision MHA quirk

`--mha=split` (or `packed`) **replaces** `nn.MultiheadAttention` modules with a custom split-QKV implementation so per-head covariances are collectible. Vision covariance collection and any merge that consumes those stats must pass the **same** `--mha` flag as `covariance.py` used; mismatches cause key-set errors during merging.

## Repository conventions

- `src/modeling.py` is a compatibility shim that re-exports `src/vision/modeling.py` — old pickled vision checkpoints reference the original module path. Don't remove it without re-pickling all checkpoints.
- Reproducing the paper plots: `notebooks/analysis.ipynb` reads from `artifacts/results/` and `artifacts/results-analysis/`.
- The repo uses `uv` for env management and Python 3.10–3.13 (`.python-version` pin). Don't pull `requirements.txt` — it's a frozen export, the source of truth is `pyproject.toml` + `uv.lock`.

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/): `<Type>: <short description>`. Do **not** use parenthesized scopes in this repo (use `Fix: ...`, not `Fix(test): ...`).

Types used in this repo (Capital case — e.g. `Feat: add x` or `Feat/create-branch-for-x`):
- `Feat:` — new feature/capability
- `Fix:` — bug fix
- `Refactor:` — restructure without behavior change
- `Perf:` — performance improvement
- `Test:` — add/update tests
- `Docs:` — documentation only
- `Chore:` — tooling, ignores, cleanup, deps
- `Revert:` — revert a previous commit

Keep the subject ≤ 72 chars, imperative mood ("add X", not "added X"), lowercase after the colon. Add a body only when the *why* isn't obvious from the diff. No Claude co-author trailers.
